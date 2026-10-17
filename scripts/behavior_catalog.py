"""Index all official definitions and installed pre-sampled instances."""
import argparse
import json
from pathlib import Path
from rummagebench.datasets.behavior import catalog

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bddl-root', type=Path, required=True)
    p.add_argument('--assets-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = catalog(a.bddl_root, a.assets_root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'tasks'}))
