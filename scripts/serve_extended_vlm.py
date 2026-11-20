"""Single-model, single-request local HF inference server (run outside ygbench)."""
from __future__ import annotations
import argparse
import base64
import importlib.metadata
import io
import hashlib
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from vlm_messages import text_messages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,help='profile name in configs/models/local_vlm.json')
    p.add_argument('--profiles',type=Path,default=Path('configs/models/local_vlm.json'))
    p.add_argument('--port',type=int,default=8091)
    p.add_argument('--quantization',choices=['nf4','none'],default=None)
    p.add_argument('--download-only',action='store_true')
    a=p.parse_args()
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
    from extended_vlm_adapters import load_backend
    model,generate,backend_metadata=load_backend(profile,path,a.quantization)
    metadata={**profile,'profile':a.model,'quantization':a.quantization,
              **backend_metadata,
              'adapter_sha256':hashlib.sha256(Path(__file__).with_name('extended_vlm_adapters.py').read_bytes()).hexdigest(),
              'server_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','accelerate','bitsandbytes']},
              'gpu':torch.cuda.get_device_name(), 'sampling':'official-chat-defaults' if profile.get('backend')=='minicpm' else 'greedy', 'message_protocol':'multiturn-public-v1', 'image_protocol':'panorama-plus-detail-v4',
              'weight_memory_bytes':torch.cuda.memory_allocated(),
              'cuda_allocator_config':os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}

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
                if not 1<=tokens<=4096:raise ValueError('max_new_tokens must be 1..4096')
                set_seed(int(data['seed']))
                items=data.get('images') or [dict(label='Image',image_png_b64=data['image_png_b64'])]
                if not isinstance(items,list) or not 1 <= len(items) <= 2:
                    raise ValueError('expected one primary image and at most one detail')
                pictures=[Image.open(io.BytesIO(base64.b64decode(x['image_png_b64'],validate=True))).convert('RGB') for x in items]
                picture=pictures[0]
                start=time.monotonic();torch.cuda.reset_peak_memory_stats()
                if self.path=='/generate' and data.get('stage')=='compaction':
                    if not backend_metadata.get('context_limit_tokens'):
                        raise ValueError('Native context capacity unavailable for memory generation')
                    tokens=backend_metadata['context_limit_tokens']
                with torch.inference_mode():
                    result=generate(pictures,text_messages(data),tokens,[x['label'] for x in items],
                                    count_only=self.path=='/count_tokens')
                if self.path=='/count_tokens':
                    self.reply(dict(model=profile['model_id'],**result));return
                self.reply({'model':profile['model_id'],**result,
                            'memory_generation_budget':'native_context_remaining' if data.get('stage')=='compaction' else None,
                            'decoding':dict(backend_metadata['decoding'],seed=int(data['seed']),max_new_tokens=tokens),
                            'generation_s':time.monotonic()-start,
                            'peak_memory_bytes':torch.cuda.max_memory_allocated(),
                            'output_budget_reached':result['output_tokens']>=tokens})
            except Exception as error:
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
