"""Single-model, single-request local HF inference server (run outside ygbench)."""
from __future__ import annotations
import argparse
import base64
import importlib.metadata
import io
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from vlm_messages import text_messages, hf_messages, validate_context_budget, context_generation_exhausted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,help='profile name in configs/models/local_vlm.json')
    p.add_argument('--profiles',type=Path,default=Path('configs/models/local_vlm.json'))
    p.add_argument('--port',type=int,default=8091)
    p.add_argument('--quantization',choices=['nf4','none'],default=None)
    p.add_argument('--download-only',action='store_true')
    p.add_argument('--attention-out',type=Path,help='Qwen3-VL only: save read-only per-token image attention')
    a=p.parse_args()
    official_qwen = a.model == 'qwen3-vl-4b'
    profile=json.loads(a.profiles.read_text())['models'][a.model]
    a.quantization=a.quantization or profile.get('quantization','nf4')
    from huggingface_hub import snapshot_download
    path=snapshot_download(profile['model_id'],revision=profile['revision'],
                           ignore_patterns=['*.bin','*.pt','original/*','*.gguf'])
    if a.download_only:
        print(path);return
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this local evaluation profile')
    kwargs={'torch_dtype':torch.bfloat16,'device_map':{'':'cuda:0'},
            'trust_remote_code':profile['trust_remote_code']}
    # Molmo derives vision activation dtype from this layer's weight. Keep it
    # floating point: a packed NF4 weight would cast images to uint8.
    skip_modules=['patch_embedding'] if a.model=='molmo2-4b' else []
    if a.quantization=='nf4':
        kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=skip_modules or None)
    processor=AutoProcessor.from_pretrained(path,trust_remote_code=profile['trust_remote_code'])
    model=AutoModelForImageTextToText.from_pretrained(path,**kwargs).eval()
    context_limit = getattr(getattr(model.config, "text_config", model.config),
                            "max_position_embeddings", None)
    recorder=None
    if a.attention_out:
        from capture_qwen_attention import QwenAttentionRecorder
        recorder=QwenAttentionRecorder(model,processor,a.attention_out)
    metadata={**profile,'profile':a.model,'quantization':a.quantization,
              'quantization_skip_modules':skip_modules if a.quantization=='nf4' else [],
              'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','accelerate','bitsandbytes']},
              'gpu':torch.cuda.get_device_name(), 'sampling':'greedy', 'message_protocol':'multiturn-public-v1', 'image_protocol':'panorama-plus-detail-v4',
              'weight_memory_bytes':torch.cuda.memory_allocated()}
    metadata['attention_recording']=bool(recorder)
    metadata['context_limit_tokens']=context_limit
    metadata['cuda_allocator_config']=os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
    if official_qwen:
        from qwen_decoding import SETTINGS, generation_kwargs
        metadata.update(sampling='qwen3-vl-instruct-official',
                        decoding=SETTINGS,
                        decoding_source='https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction')

    class Handler(BaseHTTPRequestHandler):
        def reply(self,body,status=200):
            raw=json.dumps(body).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if self.path=='/health':self.reply(metadata)
            else:self.reply({'error':'not found'},404)
        def do_POST(self):
            if self.path not in ('/generate','/count_tokens'):self.reply({'error':'not found'},404);return
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<32*1024*1024:raise ValueError('invalid request size')
                data=json.loads(self.rfile.read(length))
                if data['model']!=profile['model_id']:raise ValueError('model mismatch')
                tokens=int(data['max_new_tokens'])
                limit=32768 if official_qwen else 4096
                if not 1<=tokens<=limit:raise ValueError(f'max_new_tokens must be 1..{limit}')
                decoding_seed=int(data.get('decoding_seed',3407)) if official_qwen else int(data['seed'])
                set_seed(decoding_seed)
                items=data.get('images') or [dict(label='Image',image_png_b64=data['image_png_b64'])]
                if not isinstance(items,list) or not 1 <= len(items) <= 4:
                    raise ValueError('expected 1..4 images (primary scene + optional CHECK card + optional visual memory)')
                pictures=[Image.open(io.BytesIO(base64.b64decode(x['image_png_b64'],validate=True))).convert('RGB') for x in items]
                messages=hf_messages(text_messages(data),pictures,[x['label'] for x in items])
                inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                                     return_dict=True,return_tensors='pt')
                inputs={k:v.to(model.device) if hasattr(v,'to') else v for k,v in inputs.items()}
                n=inputs['input_ids'].shape[-1]
                if self.path=='/count_tokens':
                    self.reply(dict(model=profile['model_id'],input_tokens=n));return
                if data.get('stage')=='compaction':
                    if context_limit is None:raise ValueError('Native context capacity unavailable for memory generation')
                    tokens=context_limit
                effective_tokens=validate_context_budget(n,tokens,context_limit)
                start=time.monotonic();torch.cuda.reset_peak_memory_stats()
                if recorder and data.get('stage')!='compaction':recorder.begin(inputs)
                decoding=generation_kwargs(n) if official_qwen else dict(do_sample=False)
                with torch.inference_mode():
                    output=model.generate(**inputs,max_new_tokens=effective_tokens,**decoding)
                generated=output[0,n:]
                attention=recorder.finish(generated) if recorder and data.get('stage')!='compaction' else None
                self.reply({'model':profile['model_id'],'text':processor.decode(generated,skip_special_tokens=True),
                            'attention':attention,
                            'decoding':dict(metadata.get('decoding',{'do_sample':False}),
                                            seed=decoding_seed,max_new_tokens=tokens),
                            'input_tokens':n,'output_tokens':len(generated),
                            'effective_max_new_tokens':effective_tokens,
                            'memory_generation_budget':'native_context_remaining' if data.get('stage')=='compaction' else None,
                            'context_budget_exhausted':context_generation_exhausted(generated,effective_tokens,tokens,model.generation_config.eos_token_id),
                            'generation_s':time.monotonic()-start,
                            'peak_memory_bytes':torch.cuda.max_memory_allocated(),
                            'output_budget_reached':len(generated)>=tokens})
            except Exception as error:
                if recorder:recorder.active=False
                import traceback
                traceback.print_exc()
                self.reply({'error':f'{type(error).__name__}: {error}'},500)
                torch.cuda.empty_cache()
        def log_message(self,fmt,*args):
            print(fmt % args,flush=True)
    print(json.dumps(metadata),flush=True)
    print(f'Local VLM ready at http://127.0.0.1:{a.port}',flush=True)
    HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__':main()
