"""Optional real pinned SDK contract checks; no generated data or GPU imports."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from rummagebench.authoring.tasks.infinigen import CPU_ENV, cpu_sandbox_argv, source_audit


@pytest.mark.local_assets
def test_real_search_constraint_relevance_in_all_upstream_greedy_stages(tmp_path):
    root = os.environ.get("FINDINGBENCH_INFINIGEN_ROOT")
    python = os.environ.get("FINDINGBENCH_INFINIGEN_CPU_PYTHON")
    if not root or not python:
        pytest.skip("set FINDINGBENCH_INFINIGEN_ROOT and FINDINGBENCH_INFINIGEN_CPU_PYTHON for real SDK checks")
    source_audit(Path(root))
    script = '''
import json
import infinigen.terrain as terrain
from infinigen.terrain.core import Terrain
terrain.Terrain=Terrain
from infinigen_examples.constraints import home,util as cu
from infinigen_examples.generate_indoors import default_greedy_stages
from infinigen.core.constraints import constraint_language as cl,reasoning as r
from infinigen.core.constraints.evaluator.evaluate import relevant
from infinigen.core.tags import Semantics,SpecificObject
from infinigen.assets.objects import shelves
from rummagebench.authoring.tasks.infinigen import apply_search_constraints
stages=default_greedy_stages()
room=r.Domain({Semantics.Room,-Semantics.Object,Semantics.Kitchen,SpecificObject('kitchen_0/0')})
reports={}
for count in (2,4):
    graph=apply_search_constraints(home.home_furniture_constraints(),count,.25,
                                  cl=cl,cu=cu,Semantics=Semantics,shelves=shelves)
    whole=graph.constraints['findingbench_complete_storage_count']
    assert len(r.constraint_bounds(whole))==0
    bounds=r.constraint_bounds(graph.constraints['findingbench_cabinet_count'])
    assert bounds and all(b.low==count and b.high==count for b in bounds)
    reports[count]={}
    for name in ('on_floor_and_wall','on_floor_freestanding','on_wall','on_ceiling'):
        stage=r.substitute_all(stages[name],{cu.variable_room:room})
        reports[count][name]=relevant(whole,stage)
        assert reports[count][name]
print(json.dumps({'real_sdk_only':True,'relevance':reports}))
'''
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, **CPU_ENV, PYTHONPATH=os.pathsep.join([str(repo/"src"),root]))
    result = subprocess.run(cpu_sandbox_argv([python,"-c",script],tmp_path),cwd=root,env=env,
                            capture_output=True,text=True,timeout=60,check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(next(line for line in reversed(result.stdout.splitlines()) if line.startswith('{')))
    assert report['real_sdk_only'] is True
