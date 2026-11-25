#!/usr/bin/env python3
"""Sync and relocate heavy RummageBench v2 binary/USD assets from Hugging Face Hub.

Downloads the 4 compressed archives from the private dataset repository
(default: ``KevinWong216/RummageBench-v2-Assets``), verifies SHA-256 checksums,
extracts them into the repository root, and relocates embedded USD/JSON paths
when the collaborator checkout path differs from the authoring machine.

Usage:
  conda activate ygbench
  python scripts/sync_v2_assets.py
  python scripts/sync_v2_assets.py --relocate-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "KevinWong216/RummageBench-v2-Assets"
OLD_PREFIXES = (
    "/home/ubuntu/ygwang/FindingBench_20261016_200444/workspace",
    "/home/ubuntu/ygwang/FindingBench",
)
ARCHIVES = (
    "rummagebench_v2_release.tar.zst",
    "rummagebench_v2_shared_rooms.tar.zst",
    "rummagebench_v2_shared_objects.tar.zst",
    "rummagebench_v2_robot_and_refs.tar.zst",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_from_hf(repo_id: str, filename: str, token: str | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            token=token,
        )
        return Path(local)
    except ImportError:
        hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
        if not hf_bin:
            raise RuntimeError(
                "Neither huggingface_hub nor `hf` CLI is installed. Run `pip install huggingface_hub`."
            )
        out_dir = ROOT / ".cache" / "hf_assets"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [hf_bin, "download", repo_id, filename, "--repo-type", "dataset", "--local-dir", str(out_dir)]
        env = os.environ.copy()
        if token:
            env["HF_TOKEN"] = token
        subprocess.run(cmd, check=True, env=env)
        return out_dir / filename


def extract_tar_zst(archive: Path, dest: Path) -> None:
    if shutil.which("zstd"):
        subprocess.run(["tar", "-I", "zstd -d", "-xf", str(archive), "-C", str(dest)], check=True)
        return
    try:
        import zstandard  # type: ignore
        import tarfile

        with archive.open("rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tf:
                    tf.extractall(path=dest)
    except ImportError as exc:
        raise RuntimeError("Install `zstd` (`sudo apt install zstd`) or `pip install zstandard`.") from exc


def rewrite_prefix(val: str, new_root: str) -> str:
    for old in OLD_PREFIXES:
        if val == old or val.startswith(old + "/"):
            return new_root + val[len(old):]
    return val


def relocate_assets(target_root: Path) -> None:
    new_root = str(target_root.resolve())
    if new_root in OLD_PREFIXES:
        print(f"[relocate] Checkout path matches authoring root ({new_root}); no relocation needed.")
        return

    print(f"[relocate] Rewriting embedded paths -> {new_root}")
    json_targets = [
        target_root / "runs/rummagebench_v2_20261124/combined_manifest.json",
        target_root / "runs/rab_lived_in_20261120/combined_manifest.json",
    ]
    cases_dir = target_root / "runs/rummagebench_v2_20261124/cases"
    if cases_dir.is_dir():
        json_targets.extend(sorted(cases_dir.glob("*/episode.json")))
        json_targets.extend(sorted(cases_dir.glob("*/native_validation.json")))

    rewritten_json = 0
    for jp in json_targets:
        if not jp.is_file():
            continue
        raw = jp.read_text(encoding="utf-8")
        updated = raw
        for old in OLD_PREFIXES:
            updated = updated.replace(old, new_root)
        if updated != raw:
            jp.write_text(updated, encoding="utf-8")
            rewritten_json += 1

    try:
        from pxr import Sdf
    except ImportError:
        print(
            f"[relocate] Rewrote {rewritten_json} JSON files, but `pxr` is unavailable in {sys.executable}.\n"
            "           Run inside `conda activate ygbench`: python scripts/sync_v2_assets.py --relocate-only",
            file=sys.stderr,
        )
        return

    usd_dirs = [
        target_root / "runs/rummagebench_v2_20261124/cases",
        target_root / "runs/rummagebench_v2_20261124/layouts",
        target_root / "runs/rummagebench_v2_20261124/asset_layers",
        target_root / "runs/rab_migration_20261118/search_build_v3",
        target_root / "runs/rab_migration_20261118/available_resources",
        target_root / "runs/rab_dense_fit_20261119/drawer",
    ]
    usd_files: list[Path] = []
    for d in usd_dirs:
        if d.is_dir():
            for ext in ("*.usd", "*.usda", "*.usdc"):
                usd_files.extend(d.rglob(ext))

    updated_usd = 0
    for up in sorted(set(usd_files)):
        layer = Sdf.Layer.FindOrOpen(str(up))
        if layer is None:
            continue
        changed = False
        for ref in list(layer.GetExternalReferences()):
            new_ref = rewrite_prefix(ref, new_root)
            if new_ref != ref:
                if layer.UpdateExternalReference(ref, new_ref):
                    changed = True
        if changed:
            layer.Save()
            updated_usd += 1

    # Refresh scene_sha256 in episode.json and native_validation.json after scene.usda rewrite.
    refreshed_sha = 0
    if cases_dir.is_dir():
        for ep_dir in sorted(cases_dir.iterdir()):
            scene_p = ep_dir / "scene.usda"
            ep_p = ep_dir / "episode.json"
            val_p = ep_dir / "native_validation.json"
            if not (scene_p.is_file() and ep_p.is_file()):
                continue
            digest = sha256_file(scene_p)
            ep_obj = json.loads(ep_p.read_text(encoding="utf-8"))
            if ep_obj.get("scene_sha256") != digest:
                ep_obj["scene_sha256"] = digest
                ep_p.write_text(json.dumps(ep_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                refreshed_sha += 1
            if val_p.is_file():
                val_obj = json.loads(val_p.read_text(encoding="utf-8"))
                if "scene_sha256" in val_obj and val_obj["scene_sha256"] != digest:
                    val_obj["scene_sha256"] = digest
                    val_p.write_text(json.dumps(val_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[relocate] Complete: json={rewritten_json}, usd_layers={updated_usd}, scene_sha256={refreshed_sha}")


def verify_layout(target_root: Path) -> bool:
    required = [
        target_root / "runs/rummagebench_v2_20261124/combined_manifest.json",
        target_root / "runs/rummagebench_v2_20261124/diversity_report/diversity_metrics.json",
        target_root / "assets/robots/unitree_g1/g1_body29_dex1/simready_usd/g1_body29_dex1.usd",
        target_root / "references/curobo/curobo/content/configs/robot/unitree_g1_29dof.yml",
        target_root / "references/RobotAgentBench/README.md",
    ]
    missing = [str(p.relative_to(target_root)) for p in required if not p.exists()]
    if missing:
        print("[verify] MISSING required assets:", missing, file=sys.stderr)
        return False
    cases_count = len(list((target_root / "runs/rummagebench_v2_20261124/cases").glob("*/scene.usda")))
    print(f"[verify] PASS: all core assets present (v2 USD scenes={cases_count}/1000).")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Download/extract and relocate RummageBench v2 heavy USD/robot assets.")
    ap.add_argument("--from-dir", type=Path, default=None, help="Extract from a local bundle directory containing the 4 .tar.zst archives + SHA256SUMS (skips Hugging Face download).")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo ID.")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Optional HF access token.")
    ap.add_argument("--relocate-only", action="store_true", help="Skip download/extract; only relocate USD/JSON paths.")
    ap.add_argument("--verify-only", action="store_true", help="Only verify required local assets exist.")
    args = ap.parse_args()

    if args.verify_only:
        return 0 if verify_layout(ROOT) else 1

    if not args.relocate_only:
        src_dir = args.from_dir.resolve() if args.from_dir else None
        sums_path = (src_dir / "SHA256SUMS") if src_dir else download_from_hf(args.repo, "SHA256SUMS", args.token)
        expected = {}
        if sums_path.is_file():
            for line in sums_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    expected[Path(parts[-1]).name] = parts[0]

        for name in ARCHIVES:
            if src_dir:
                local_arc = src_dir / name
                if not local_arc.is_file():
                    raise FileNotFoundError(f"Archive not found in --from-dir: {local_arc}")
                print(f"[sync] Using local archive {local_arc} ...")
            else:
                print(f"[sync] Fetching {args.repo}/{name} ...")
                local_arc = download_from_hf(args.repo, name, args.token)
            if name in expected:
                actual = sha256_file(local_arc)
                if actual != expected[name]:
                    raise RuntimeError(f"Checksum mismatch for {name}: expected {expected[name]}, got {actual}")
                print(f"[sync] Verified SHA-256 for {name}")
            print(f"[sync] Extracting {name} into {ROOT} ...")
            extract_tar_zst(local_arc, ROOT)

    relocate_assets(ROOT)
    return 0 if verify_layout(ROOT) else 1


if __name__ == "__main__":
    raise SystemExit(main())
