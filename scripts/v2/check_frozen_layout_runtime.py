"""Check the actual frozen producer/provider ABI before launching simulation."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json
from types import SimpleNamespace


def check(runtime):
    release=json.loads((runtime/'release.json').read_text())
    for relative,expected in release['source_sha256'].items():
        if hashlib.sha256((runtime/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('FROZEN_RUNTIME_SOURCE_CHANGED: '+relative)
    path=runtime/'src/rummagebench/sim/omnigibson/inside_volume.py'
    spec=importlib.util.spec_from_file_location('frozen_inside_volume_contract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pose=[.1,.1,.02,0.,0.,0.,1.]
    bank=[pose]
    if hasattr(module,'native_drop_candidates'):
        planes=[[-1.,0,0,0],[1.,0,0,-.2],[0,-1.,0,0],[0,1.,0,-.2],[0,0,-1.,0],[0,0,1.,-.1]]
        bank=module.native_drop_candidates(pose,[0.,0.,0.],planes)
        assert len(bank)==3
    for candidate,n in [(pose,1),(bank,len(bank))]:
        result=module.native_gpu_proposals(SimpleNamespace(_findingbench_layout_candidate=candidate),None,None,None,100)
        assert tuple(result.shape)==(n,7)
    # Check the consumer as well: accepting a bank is insufficient if the
    # transformed SDK loop still executes only the first candidate.
    native=importlib.util.find_spec('omnigibson')
    if native is None:raise RuntimeError('Native SDK not installed')
    source=(Path(native.origin).parent/'object_states/inside.py').read_text()
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Inside')
    setter=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_set_value')
    adapted=ast.parse(module.gpu_volume_source(ast.get_source_segment(source,setter))).body[0]
    bank_index=next(i for i,n in enumerate(adapted.body) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='_gpu_bank' for t in n.targets))
    loop_index=next(i for i,n in enumerate(adapted.body) if isinstance(n,ast.For) and ast.unparse(n.target)=='attempt_idx')
    namespace=dict(self=SimpleNamespace(obj=SimpleNamespace(_findingbench_layout_candidate=bank)),
        _findingbench_gpu_proposals=module.native_gpu_proposals,container_link=None,aabb_low=None,aabb_high=None,
        total_attempts=1,_findingbench_audit={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=adapted.body[bank_index:loop_index],type_ignores=[])),str(path),'exec'),namespace)
    attempts=list(eval(compile(ast.Expression(adapted.body[loop_index].iter),str(path),'eval'),namespace))
    assert len(attempts)==len(bank),'Frozen SDK consumer dropped candidate poses'
    for invalid in ([pose]*4,[[float('nan')]*7],[[0.]*6]):
        try:module.native_gpu_proposals(SimpleNamespace(_findingbench_layout_candidate=invalid),None,None,None,100)
        except ValueError:pass
        else:raise AssertionError('Invalid candidate bank accepted')
    return dict(status='frozen_layout_contract_passed',runtime=str(runtime),candidate_count=len(bank))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('runtime',type=Path)
    print(json.dumps(check(parser.parse_args().runtime.resolve())))
