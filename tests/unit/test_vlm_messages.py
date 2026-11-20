import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('vlm_messages',Path(__file__).resolve().parents[2]/'scripts/vlm_messages.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_all_adapters_preserve_roles_text_and_current_image_only():
    raw='Reasoning\n' * 400 + '{"action": malformed}'
    turns=[{'role':'user','content':'initial'}, {'role':'assistant','content':raw},
           {'role':'user','content':'feedback'}]
    messages=module.text_messages({'messages':turns,'prompt':'AUDIT_NOT_INPUT'})
    assert messages==turns
    hf=module.hf_messages(messages,['PIXELS'],['panorama'])
    assert hf[1]['content']==[{'type':'text','text':raw}]
    assert not any(c['type']=='image' for m in hf[:-1] for c in m['content'])
    assert hf[-1]['content'][1]=={'type':'image','image':'PIXELS'}
    mini=module.minicpm_messages(messages,['PIXELS'],['panorama'])
    assert mini[1]=={'role':'assistant','content':[raw]}
    assert mini[-1]['content']==['PIXELS','panorama','feedback']
    phi=module.phi_messages(messages,1,['panorama'])
    assert '<|assistant|>'+raw+'<|end|><|user|><|image_1|>' in phi
    assert phi.endswith('feedback<|end|><|assistant|>')
    assert turns==messages


def test_legacy_and_invalid_conversations():
    assert module.text_messages({'prompt':'legacy'})==[{'role':'user','content':'legacy'}]
    with pytest.raises(ValueError):
        module.text_messages({'messages':[{'role':'assistant','content':'wrong'}]})
    with pytest.raises(ValueError):
        module.text_messages({'messages':[]})


def test_native_context_budget_is_explicit_and_never_truncates():
    import pytest
    validate_context_budget = module.validate_context_budget
    validate_context_budget(36863, 1, 36864)
    validate_context_budget(40959, 1, 40960)
    with pytest.raises(ValueError, match="ContextWindowExceeded.*History was not truncated"):
        validate_context_budget(47764, 1, 40960)
    assert validate_context_budget(36000, 4096, 36864) == 864
    assert validate_context_budget(4236, 4096, 8192) == 3956
    assert module.context_generation_exhausted([1, 2], 2, 10, 9)
    assert not module.context_generation_exhausted([1, 9], 2, 10, 9)
    assert not module.context_generation_exhausted([1, 2], 2, 2, 9)
