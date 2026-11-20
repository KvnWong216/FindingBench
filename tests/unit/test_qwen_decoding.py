import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('qwen_decoding',Path(__file__).resolve().parents[2]/'scripts/qwen_decoding.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_presence_penalizes_only_generated_tokens_once():
    scores=torch.tensor([[1.,2.,3.,4.,5.],[1.,2.,3.,4.,5.]])
    ids=torch.tensor([[0,1,2,2,3],[3,4,1,1,1]])
    result=module.GeneratedPresencePenalty(2)(ids,scores)
    assert torch.equal(result,torch.tensor([[1.,2.,1.5,2.5,5.],[1.,.5,3.,4.,5.]]))
    assert torch.equal(module.GeneratedPresencePenalty(5)(ids,scores),scores)


def test_official_sampling_and_no_action_filter():
    kw=module.generation_kwargs(10)
    assert kw['do_sample'] is True
    assert (kw['temperature'],kw['top_p'],kw['top_k'])==(.7,.8,20)
    assert kw['repetition_penalty']==1.0
    assert module.SETTINGS['seed']==3407
