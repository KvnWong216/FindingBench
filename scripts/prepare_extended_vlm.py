"""Download the fixed second-cohort weights without using the GPU."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
NAMES=['smolvlm2-2.2b','phi4-multimodal','llava-onevision-7b','minicpm-v4.5-8b','gemma3-4b']
if __name__=='__main__':
    out=ROOT/'runs/vlm_expansion_setup';out.mkdir(exist_ok=True,parents=True)
    profiles=json.loads((ROOT/'configs/models/local_vlm.json').read_text())['models']
    state={'status':'downloading','models':{n:{'status':'pending'} for n in NAMES}}
    def save():
        tmp=out/'downloads.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'downloads.json')
    save()
    for name in NAMES:
        profile=profiles[name];state['models'][name]={'status':'downloading',**profile};save()
        try:
            path=snapshot_download(profile['model_id'],revision=profile['revision'],
                ignore_patterns=['*.bin','*.pt','original/*','*.gguf'],max_workers=4)
            state['models'][name].update(status='ready',path=path)
        except Exception as e:
            state['models'][name].update(status='blocked_access' if type(e).__name__=='GatedRepoError' else 'download_error',error=f'{type(e).__name__}: {e}')
        save();print(name,state['models'][name]['status'],flush=True)
    state['status']='complete';save()
