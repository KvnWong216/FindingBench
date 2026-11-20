"""Official-model adapters for the second local VLM cohort.

Keep separate from the first cohort's frozen executable sources.
"""
from vlm_messages import hf_messages, minicpm_messages, phi_messages, validate_context_budget, context_generation_exhausted

def load_backend(profile, path, quantization):
    import torch
    from transformers import (AutoModel, AutoModelForCausalLM,
        AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, BitsAndBytesConfig)
    kind=profile.get('backend','standard')
    kwargs=dict(torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},
                trust_remote_code=profile['trust_remote_code'])
    # Preserve custom vision and adapter layers; quantify the language backbone.
    skip={'minicpm':['vpm','resampler','lm_head'],
          'phi4':['embed_tokens_extend','vision_embed_tokens','audio_embed_tokens','lm_head','lora_A','lora_B']}.get(kind)
    if quantization=='nf4':
        kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=skip)
    if kind=='minicpm':
        kwargs['attn_implementation']='sdpa'
        model=AutoModel.from_pretrained(path,**kwargs).eval()
        tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True)
        processor=AutoProcessor.from_pretrained(path,trust_remote_code=True)
    elif kind=='phi4':
        kwargs['_attn_implementation']='sdpa'
        model=AutoModelForCausalLM.from_pretrained(path,**kwargs).eval()
        processor=AutoProcessor.from_pretrained(path,trust_remote_code=True)
        tokenizer=processor.tokenizer
    else:
        model=AutoModelForImageTextToText.from_pretrained(path,**kwargs).eval()
        processor=AutoProcessor.from_pretrained(path,trust_remote_code=False)
        tokenizer=processor.tokenizer
    text_config=getattr(model.config,"text_config",model.config)
    context_limit=getattr(text_config,"max_position_embeddings",None)
    def generate(picture,prompt,tokens,labels=None,count_only=False):
        pictures=picture if isinstance(picture,list) else [picture]
        messages=prompt if isinstance(prompt,list) else [{'role':'user','content':prompt}]
        labels=labels or ['Image']*len(pictures)
        if kind=='minicpm':
            # Capture exact counts at the underlying generation call; do not
            # estimate vision tokens from decoded text.
            stats={}
            class CountComplete(Exception):
                pass
            original=model.llm.generate
            def counted(*args,**kw):
                n=kw['inputs_embeds'].shape[-2]
                if count_only:
                    stats.update(input_tokens=n)
                    raise CountComplete()
                effective_tokens=validate_context_budget(n,tokens,context_limit)
                kw['max_new_tokens']=effective_tokens
                result=original(*args,**kw)
                stats.update(input_tokens=n,output_tokens=result.shape[-1],
                             effective_max_new_tokens=effective_tokens,
                             context_budget_exhausted=context_generation_exhausted(result[0],effective_tokens,tokens,kw.get('eos_token_id',model.llm.generation_config.eos_token_id)))
                return result
            model.llm.generate=counted
            try:
                text=model.chat(msgs=minicpm_messages(messages,pictures,labels),
                    tokenizer=tokenizer,processor=processor,enable_thinking=False,
                    sampling=True,stream=False,
                    max_inp_length=10**9,max_new_tokens=tokens)
            except CountComplete:
                return stats
            finally:model.llm.generate=original
            return dict(text=text,**stats)
        if kind=='phi4':
            formatted=phi_messages(messages,len(pictures),labels)
            inputs=processor(text=formatted,images=pictures,return_tensors='pt')
        else:
            inputs=processor.apply_chat_template(hf_messages(messages,pictures,labels),tokenize=True,
                add_generation_prompt=True,return_dict=True,return_tensors='pt')
        inputs={k:v.to(model.device) if hasattr(v,'to') else v for k,v in inputs.items()}
        n=inputs['input_ids'].shape[-1]
        if count_only:return dict(input_tokens=n)
        effective_tokens=validate_context_budget(n,tokens,context_limit)
        result=model.generate(**inputs,max_new_tokens=effective_tokens,do_sample=False,num_beams=1)
        generated=result[0,n:]
        return dict(text=processor.decode(generated,skip_special_tokens=True),
                    input_tokens=n,output_tokens=len(generated),effective_max_new_tokens=effective_tokens,
                    context_budget_exhausted=context_generation_exhausted(generated,effective_tokens,tokens,model.generation_config.eos_token_id))
    return model,generate,{'context_limit_tokens':context_limit,'backend':kind,'quantization_skip_modules':skip,
                          'enable_thinking':False if kind=='minicpm' else None,
                          'decoding':({'do_sample':True,'temperature':0.7,'top_p':0.8,'top_k':100,'repetition_penalty':1.03} if kind=='minicpm' else {'do_sample':False,'num_beams':1}),
                          'attention_implementation':getattr(model.config,'_attn_implementation',None)}
