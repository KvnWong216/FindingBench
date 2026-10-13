"""Promote / freeze the certified episodes of a tier's split into the
repository (TASK_TIERS_PLAN §26: build candidate != benchmark release asset).

    PYTHONPATH=src python scripts/tasks/promote_tasks.py --tier easy_v1

Writes
    assets/benchmark/<tier>/scenarios/<id>.yaml   certified scenario
    assets/benchmark/<tier>/split.json            dev / test ids + stats
    assets/benchmark/<tier>/manifest.json         per-episode metadata
    assets/robots/<robot>/                        URDF + collision meshes

Each scenario is the certified candidate byte-for-byte except that
`kinematics.urdf_path` points at the released robot (build/ is gitignored).
The manifest keeps the certified `scenario_sha256` next to the released
file's hash; the robot URDF must hash to the certified `urdf_hash`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from rummagebench.authoring.tasks.gates import structural_violations, simulator_violations
from rummagebench.authoring.tasks.search_certificate import SearchStructureCertificate
from rummagebench.authoring.tasks.tiers import load_tier

from _common import REPO, host_config

ROBOT_SRC = {"r1pro": ("build/robots/r1pro.urdf", "build/robots/r1pro_meshes",
                       "build/robots/r1pro.manifest.json")}


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def promote_robot(robot_id: str, urdf_hash: str):
    urdf, meshes, manifest = (REPO / p for p in ROBOT_SRC[robot_id])
    if sha256(urdf) != urdf_hash:
        raise SystemExit(f"{urdf} does not match the certified urdf_hash {urdf_hash}")
    dst = REPO / "assets" / "robots" / robot_id
    if dst.exists():
        expected = {urdf.name: urdf, manifest.name: manifest}
        expected.update({f"{meshes.name}/{p.relative_to(meshes).as_posix()}": p
                         for p in meshes.rglob("*") if p.is_file()})
        actual = {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()}
        if actual != set(expected) or any(not (dst / name).is_file() or sha256(dst / name) != sha256(p)
                                         for name, p in expected.items()):
            raise SystemExit(f"conflicting released robot at {dst}; cannot overwrite frozen assets")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(dir=dst.parent, prefix=f".{robot_id}."))
        try:
            shutil.copytree(meshes, stage / meshes.name)
            shutil.copy2(urdf, stage / urdf.name)
            shutil.copy2(manifest, stage / manifest.name)
            stage.rename(dst)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return urdf.relative_to(REPO).as_posix(), (dst / urdf.name).relative_to(REPO).as_posix()


def preflight_episodes(src, ids, records, tier):
    """Check every episode before touching any released asset.

    This legacy promoter certifies single-episode feasibility/visibility.
    A household family must go through a family verifier; this path cannot
    silently inherit an episode's status as family/search-cost acceptance.
    """
    spec = load_tier(tier)
    for i in sorted(ids):
        rec = records[i]
        if rec.get("status") != "certified":
            raise SystemExit(f"{i}: status {rec.get('status')}")
        cert = SearchStructureCertificate.from_dict(rec["search_certificate"])
        violations = structural_violations(cert, spec) + simulator_violations(cert, spec)
        if cert.status != "certified" or violations:
            raise SystemExit(f"{i}: certificate does not pass current gates")
        cand = src / "candidates" / f"{i}.yaml"
        if sha256(cand) != rec["provenance"]["certifier"]["scenario_sha256"]:
            raise SystemExit(f"{i}: candidate changed since certification")
        plan = json.loads((src / "plans" / f"{i}.json").read_text())
        prov = plan.get("provenance") or {}
        if ("household" in tier or any(k in plan or k in prov or k in rec
                                      for k in ("family_id", "family", "background_id", "family_binding"))):
            raise SystemExit(f"{i}: household family cannot be promoted with a single-episode certificate")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="easy_v1")
    ap.add_argument("--host-config")
    args = ap.parse_args(argv)

    src = host_config(args.host_config)["build_root"] / args.tier
    split = json.loads((src / "split.json").read_text())
    if not split.get("release_candidate") or split.get("notes"):
        raise SystemExit(f"{src/'split.json'} is not a clean release candidate")
    ids = {i: name for name in ("dev", "test") for i in split[name]}

    records = {i: json.loads((src / "certificates" / f"{i}.json").read_text()) for i in ids}
    preflight_episodes(src, ids, records, args.tier)
    robots = {(r["provenance"]["robot"]["id"], r["provenance"]["robot"]["urdf_hash"])
              for r in records.values()}
    if len(robots) != 1:
        raise SystemExit(f"episodes certified against several robots: {robots}")
    robot_id, _ = next(iter(robots))
    expected_urdf = ROBOT_SRC[robot_id][0]
    for i in ids:
        text = (src / "candidates" / f"{i}.yaml").read_text()
        if text.count(f"urdf_path: {expected_urdf}") != 1:
            raise SystemExit(f"{i}: expected one urdf_path {expected_urdf}")
    old_urdf, new_urdf = promote_robot(*robots.pop())

    out = REPO / "assets" / "benchmark" / args.tier
    if (out / "scenarios").exists():  # README.md is hand-written, keep it
        shutil.rmtree(out / "scenarios")
    (out / "scenarios").mkdir(parents=True)
    episodes = []
    for i in sorted(ids):
        rec = records[i]
        if rec["status"] != "certified":
            raise SystemExit(f"{i}: status {rec['status']}")
        cand = src / "candidates" / f"{i}.yaml"
        certified_sha = rec["provenance"]["certifier"]["scenario_sha256"]
        if sha256(cand) != certified_sha:
            raise SystemExit(f"{i}: candidate changed since certification")
        text = cand.read_text(encoding="utf-8")
        if text.count(f"urdf_path: {old_urdf}") != 1:
            raise SystemExit(f"{i}: expected one urdf_path {old_urdf}")
        text = text.replace(
            "# build candidate (scripts/tasks/06_compile.py) — not a release asset",
            f"# released by scripts/tasks/promote_tasks.py ({args.tier}, {ids[i]})"
        ).replace(f"urdf_path: {old_urdf}", f"urdf_path: {new_urdf}")
        dst = out / "scenarios" / f"{i}.yaml"
        dst.write_text(text, encoding="utf-8")

        plan = json.loads((src / "plans" / f"{i}.json").read_text())
        sc, oracle = rec["search_certificate"], rec["oracle_certificate"]
        episodes.append({
            "id": i, "split": ids[i], "scenario": f"scenarios/{i}.yaml",
            "instruction": plan["instruction"], "mode": plan["mode"],
            "scene": plan["scene"], "room": plan["room"], "room_type": plan["room_type"],
            "search_region": plan["search_region"],
            "target": {k: plan["target"][k] for k in ("category", "model", "entity", "slot_id")},
            "target_slot_type": sc["target_slot_type"],
            "candidate_slot_count": sc["candidate_slot_count"],
            "open_depth": sc["open_depth"],
            "revealed_visibility": round(sc["revealed_visibility"], 4),
            "oracle_depth": oracle["oracle_depth"],
            "witness_steps": len(rec["witness_trace"]),
            "scenario_sha256": sha256(dst),
            "certified_scenario_sha256": certified_sha,
        })

    (out / "split.json").write_text(json.dumps(split, indent=1, sort_keys=True))
    (out / "manifest.json").write_text(json.dumps({
        "tier": args.tier, "robot_urdf": new_urdf,
        "certification_scope": "single_episode_execution_and_visibility",
        "family_certified": False, "search_difficulty_status": "unmeasured",
        "max_planning_steps": 16, "episodes": episodes}, indent=1))
    print(f"{len(episodes)} episodes (dev {len(split['dev'])} / test {len(split['test'])}) "
          f"-> {out.relative_to(REPO)}; robot -> {new_urdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
