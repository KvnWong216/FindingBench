#!/usr/bin/env python3
"""Read-only LAN Collaborator Portal for RummageBench v2 (docs, figures, gallery, .orx, and 1.95GB asset closure).

Serves:
  /                       Collaborator onboarding & progress portal HTML
  /openresearch/...       OpenResearch docs (01..06), figures, interactive HTML overview
  /inspect100/...         100-scene dual-view ray-traced inspection gallery
  /diversity/...          TPT-Bench + Qwen3-VL non-collapse figures & diversity_metrics.json
  /RummageBench.orx       Full offline OpenResearch project bundle (experiment tree + chats + docs + repo.bundle)
  /assets/...             4 SHA-256-verified compressed asset archives + SHA256SUMS (supports Range & streaming)
"""
from __future__ import annotations

import argparse
import html
import mimetypes
import os
import posixpath
import socket
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORX_DIR = Path(
    os.environ.get(
        "RUMMAGEBENCH_ORX_FILES",
        "/home/ubuntu/.local/share/openresearch/files/rummagebench",
    )
)
BUNDLE_DIR = ROOT / "runs" / "rummagebench_v2_release_bundle"
INSPECT_DIR = ROOT / "runs" / "rummagebench_v2_20261124" / "inspect100"
DIVERSITY_DIR = ROOT / "runs" / "rummagebench_v2_20261124" / "diversity_report"
ORX_FILE = ROOT / "RummageBench.orx"

CLOUD_ORX_URL = "https://www.autoarxiv.org/p/279z42FCuDrb"
GITHUB_URL = "https://github.com/KvnWong216/FindingBench"


def detect_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def safe_subpath(base: Path, rel: str) -> Path | None:
    clean = posixpath.normpath("/" + urllib.parse.unquote(rel)).lstrip("/")
    target = (base / clean).resolve()
    base_res = base.resolve()
    if target == base_res or base_res in target.parents:
        return target
    return None


def portal_html(host_addr: str, port: int) -> bytes:
    base_url = f"http://{host_addr}:{port}"
    archives = []
    if BUNDLE_DIR.is_dir():
        for name in (
            "rummagebench_v2_release.tar.zst",
            "rummagebench_v2_shared_rooms.tar.zst",
            "rummagebench_v2_shared_objects.tar.zst",
            "rummagebench_v2_robot_and_refs.tar.zst",
            "SHA256SUMS",
        ):
            p = BUNDLE_DIR / name
            if p.is_file():
                mb = p.stat().st_size / (1024 * 1024)
                sz_str = f"{mb:.1f} MB" if mb >= 1 else f"{p.stat().st_size} B"
                archives.append((name, sz_str))

    arc_rows = "\n".join(
        f'<tr><td><a href="/assets/{html.escape(n)}"><code>{html.escape(n)}</code></a></td><td>{sz}</td></tr>'
        for n, sz in archives
    )
    orx_sz = f"{ORX_FILE.stat().st_size / (1024*1024):.1f} MB" if ORX_FILE.is_file() else "missing"

    doc_links = "\n".join(
        f'<li><a href="/openresearch/{fn}"><code>{fn}</code></a></li>'
        for fn in (
            "01-project-overview-and-mechanisms.md",
            "02-metrics-and-empirical-results.md",
            "03-collaborator-sync-and-roadmap.md",
            "04-literature-review-and-design-philosophy.md",
            "05-dataset-v2-10pct-audit.md",
            "06-diversity-and-non-collapse-analysis.md",
        )
        if (ORX_DIR / fn).is_file()
    )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RummageBench v2 — 局域网协作总控门户</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif; background: #f8fafc; color: #0f172a; margin: 0; line-height: 1.6; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 28px 20px 60px; }}
  .hero {{ background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%); color: #fff; padding: 26px 30px; border-radius: 16px; margin-bottom: 20px; }}
  .hero h1 {{ margin: 0 0 8px; font-size: 25px; }}
  .hero p {{ margin: 0; color: #dbeafe; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 16px; margin-bottom: 18px; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px 22px; }}
  .card h2 {{ margin: 0 0 10px; font-size: 18px; color: #1e293b; }}
  pre {{ background: #0f172a; color: #e2e8f0; padding: 14px; border-radius: 10px; overflow-x: auto; font-size: 13px; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  a {{ color: #1d4ed8; text-decoration: none; font-weight: 500; }}
  a:hover {{ text-decoration: underline; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid #e2e8f0; }}
  .figs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }}
  .figs img {{ width: 100%; border-radius: 8px; border: 1px solid #cbd5e1; }}
  .pill {{ display: inline-block; background: #eff6ff; color: #1d4ed8; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-right: 6px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <span class="pill" style="background:rgba(255,255,255,.18);color:#fff;">Zero-Gap Collaborator LAN Portal · {html.escape(base_url)}</span>
    <h1>RummageBench v2（1,000 场景 · 60 布局 · Core-120 · 不坍缩认证）</h1>
    <p>打开本页面即可零信息差同步全部代码、实验树、研究文档（01–06）、论文配图、100 场景实渲抽检与完整 4,057 文件 USD/机器人运行时资产闭环。</p>
  </div>

  <div class="card" style="margin-bottom:18px;">
    <h2>🚀 1 分钟本地完整搭建与一键开发命令</h2>
<pre># 1. 拉取 GitHub 最新主分支（含 v2 构建/MCP/评测/分析全套源码与 Core-120 配置）
git clone {GITHUB_URL}.git RummageBench
cd RummageBench &amp;&amp; conda activate ygbench

# 2. 通过本局域网高速拉取 1.95 GB 精确闭环资产（自动校验 SHA-256、解压、重定位所有 USD/JSON 路径）
python scripts/sync_v2_assets.py --from-url {html.escape(base_url)}/assets

# 3. 跑 CPU 单元测试（45 passed）与直接用付费前沿大模型开跑 Core-120
pytest tests/test_rab_*.py tests/unit/test_rab_mcp_lifecycle.py tests/unit/test_policy_adapter.py -q
export GEMINI_API_KEY="YOUR_API_KEY"
python scripts/run_rab_mcp_eval.py --config configs/benchmarks/eval_v2_core120_api.json --resume</pre>
  </div>

  <div class="grid">
    <div class="card">
      <h2>📊 在线看板与交互巡检</h2>
      <ul>
        <li><a href="/openresearch/rummagebench-interactive-overview.html"><strong>RummageBench v2 交互式架构与失败范式看板</strong></a></li>
        <li><a href="/inspect100/index.html"><strong>100 场景（200 图）双视角光追抽检画廊（neg_z = 0）</strong></a></li>
        <li><a href="{CLOUD_ORX_URL}" target="_blank"><strong>OpenResearch 云端在线发布页</strong></a>（含完整实验树与对话记录）</li>
        <li><a href="/RummageBench.orx"><strong>下载离线 <code>RummageBench.orx</code>（{orx_sz}）</strong></a> — 本地执行 <code>orx project import RummageBench.orx</code></li>
        <li><a href="{GITHUB_URL}" target="_blank"><strong>GitHub 主仓库 <code>KvnWong216/FindingBench</code></strong></a></li>
      </ul>
    </div>

    <div class="card">
      <h2>📚 研究文档库（01–06）</h2>
      <ul>{doc_links}</ul>
      <p style="font-size:13px;color:#475569;margin-top:8px;">完整指标 JSON：<a href="/diversity/diversity_metrics.json"><code>diversity_metrics.json</code></a></p>
    </div>
  </div>

  <div class="card" style="margin-bottom:18px;">
    <h2>📦 完整 USD &amp; Unitree G1 / cuRobo 运行时资产直链（{html.escape(base_url)}/assets/）</h2>
    <table>
      <thead><tr><th>压缩包 / 校验文件</th><th>大小</th></tr></thead>
      <tbody>{arc_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>📈 TPT-Bench + Qwen3-VL 双重不坍缩核心配图（点击放大）</h2>
    <div class="figs">
      <a href="/diversity/fig1_tpt_attribute_distributions.png" target="_blank"><img src="/diversity/fig1_tpt_attribute_distributions.png" alt="Fig 1" /></a>
      <a href="/diversity/fig2_visual_non_collapse_manifold.png" target="_blank"><img src="/diversity/fig2_visual_non_collapse_manifold.png" alt="Fig 2" /></a>
      <a href="/diversity/fig3_semantic_non_collapse_manifold.png" target="_blank"><img src="/diversity/fig3_semantic_non_collapse_manifold.png" alt="Fig 3" /></a>
      <a href="/diversity/fig4_spatial_layout_non_collapse.png" target="_blank"><img src="/diversity/fig4_spatial_layout_non_collapse.png" alt="Fig 4" /></a>
    </div>
  </div>
</div>
</body>
</html>
"""
    return page.encode("utf-8")


class PortalHandler(BaseHTTPRequestHandler):
    portal_host = "127.0.0.1"
    portal_port = 8095

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass

    def _resolve(self, url_path: str) -> Path | None:
        p = urllib.parse.urlparse(url_path).path
        if p == "/RummageBench.orx":
            return ORX_FILE if ORX_FILE.is_file() else None
        prefixes = (
            ("/assets/", BUNDLE_DIR),
            ("/openresearch/", ORX_DIR),
            ("/inspect100/", INSPECT_DIR),
            ("/diversity/", DIVERSITY_DIR),
        )
        for pref, root_dir in prefixes:
            if p.startswith(pref):
                sub = p[len(pref):]
                if not sub:
                    return None
                return safe_subpath(root_dir, sub)
        return None

    def _send_file(self, fp: Path, head_only: bool = False) -> None:
        if fp.is_dir():
            idx = fp / "index.html"
            if idx.is_file():
                fp = idx
            else:
                self.send_error(HTTPStatus.FORBIDDEN, "Directory listing disabled")
                return
        if not fp.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        size = fp.stat().st_size
        ctype, _ = mimetypes.guess_type(str(fp))
        if fp.suffix == ".md":
            ctype = "text/plain; charset=utf-8"
        elif fp.suffix in (".zst", ".orx"):
            ctype = "application/octet-stream"
        elif not ctype:
            ctype = "application/octet-stream"

        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_hdr = self.headers.get("Range")
        if range_hdr and range_hdr.startswith("bytes="):
            spec = range_hdr[6:].split(",")[0].strip()
            if "-" in spec:
                s_str, e_str = spec.split("-", 1)
                start = int(s_str) if s_str else max(0, size - int(e_str))
                end = int(e_str) if (e_str and s_str) else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                status = HTTPStatus.PARTIAL_CONTENT

        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if head_only:
            return
        with fp.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path).path
        if parsed in ("/", "/index.html"):
            body = portal_html(self.portal_host, self.portal_port)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        fp = self._resolve(self.path)
        if fp is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self._send_file(fp, head_only=True)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path).path
        if parsed in ("/", "/index.html"):
            body = portal_html(self.portal_host, self.portal_port)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        fp = self._resolve(self.path)
        if fp is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self._send_file(fp, head_only=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve read-only RummageBench v2 LAN collaborator portal.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--public-ip", default=None, help="LAN IP displayed in generated commands.")
    args = ap.parse_args()

    lan_ip = args.public_ip or detect_lan_ip()
    PortalHandler.portal_host = lan_ip
    PortalHandler.portal_port = args.port

    httpd = ThreadingHTTPServer((args.host, args.port), PortalHandler)
    print(f"[lan-portal] Serving read-only RummageBench v2 portal at http://{lan_ip}:{args.port}/", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
