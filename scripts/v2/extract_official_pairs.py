"""Official BEHAVIOR-1K placement sub-goals -> (object category, receptacle category) pairs (CPU).

For every activity definition in build/behavior/official_catalog.json, every
goal literal ontop(X, Y) / inside(X, Y) (also inside forall / forn / exists
quantifiers) is recorded with the synsets of X and Y. Synsets expand to
BEHAVIOR asset categories with bddl.object_taxonomy (leaf categories of the
synset subtree). These pairs are the source of vla_difficulty_v2 base tasks:
a base task re-uses one official single-object sub-goal ("put an apple
in the fridge" -> object category apple, receptacle category fridge) in a
scene where that receptacle category is reachable and certified.

    python scripts/v2/extract_official_pairs.py
Writes build/vla_difficulty_v2/official_pairs.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_official_tasks import _literals  # noqa: E402

from rummagebench.datasets.behavior import parse_bddl  # noqa: E402

OUT = ROOT / "build/vla_difficulty_v2/official_pairs.json"


def synset_of(token: str) -> str:
    t = token.lstrip("?")
    m = re.match(r"^(.*\.n\.\d+)(_\d+)?$", t)
    return m.group(1) if m else t


def main() -> int:
    from bddl.object_taxonomy import ObjectTaxonomy
    tax = ObjectTaxonomy()
    cat = json.loads((ROOT / "build/behavior/official_catalog.json").read_text())
    pairs: dict[tuple, dict] = {}
    for task in cat["tasks"]:
        tree = parse_bddl(Path(task["definition"]).read_text())
        sec = {n[0]: n[1:] for n in tree if isinstance(n, list) and n}
        goal: list = []
        for e in sec.get(":goal", []):
            _literals(e, goal)
        for lit in goal:
            if lit[0] not in ("ontop", "inside") or len(lit) != 3:
                continue
            obj, rec = synset_of(lit[1]), synset_of(lit[2])
            key = (lit[0], obj, rec)
            row = pairs.setdefault(key, {"relation": lit[0], "object_synset": obj, "receptacle_synset": rec,
                                         "activities": []})
            src = f"{task['activity_name']}#{task['activity_definition_id']}"
            if src not in row["activities"]:
                row["activities"].append(src)
    out = []
    for row in pairs.values():
        try:
            oc = sorted(tax.get_subtree_categories(row["object_synset"]))
            rc = sorted(tax.get_subtree_categories(row["receptacle_synset"]))
        except Exception:  # noqa: BLE001 - substances / unknown synsets
            continue
        if not oc or not rc:
            continue
        out.append({**row, "object_categories": oc, "receptacle_categories": rc})
    out.sort(key=lambda r: -len(r["activities"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": "build/behavior/official_catalog.json (BEHAVIOR-1K BDDL)",
                               "pairs": out}, indent=1) + "\n")
    rc = Counter()
    for r in out:
        for c in r["receptacle_categories"]:
            rc[c] += len(r["activities"])
    print(f"{len(out)} official (relation, object, receptacle) sub-goal patterns from "
          f"{len({a for r in out for a in r['activities']})} activity definitions")
    print("top receptacle categories:", rc.most_common(30))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
