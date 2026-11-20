"""Run in ygbench-vlm; no model download or GPU required."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from types import SimpleNamespace
from capture_qwen_attention import last_query_weights
from transformers.integrations.sdpa_attention import sdpa_attention_forward, repeat_kv

torch.manual_seed(0)
m=SimpleNamespace(num_key_value_groups=2,is_causal=True)
q=torch.randn(1,4,5,8)
k=torch.randn(1,2,5,8)
v=torch.randn(1,2,5,8)
for mask in [None, torch.zeros(1,1,5,5), torch.ones(1,1,5,5,dtype=torch.bool)]:
    if mask is not None:
        if mask.dtype==torch.bool:mask[:,:,:,-1]=False
        else:mask[:,:,:,-1]=float("-inf")
    before=sdpa_attention_forward(m,q,k,v,mask,scaling=8**-.5)[0]
    w=last_query_weights(m,q,k,mask,8**-.5)
    after=sdpa_attention_forward(m,q,k,v,mask,scaling=8**-.5)[0]
    assert torch.equal(before,after), "Read-only weights changed model output"
    kk=repeat_kv(k,2)
    scores=(q.float()@kk.float().transpose(-2,-1))*8**-.5
    if mask is not None:
        scores=scores.masked_fill(~mask,float("-inf")) if mask.dtype==torch.bool else scores+mask
    reference=scores.softmax(-1)[0,:,-1,:]
    torch.testing.assert_close(w,reference)
    torch.testing.assert_close(w.sum(-1),torch.ones(4))
print("PASS: live-QK weight extraction matches explicit attention; SDPA output bit-identical.")
