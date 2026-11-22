#!/usr/bin/env python3
"""Fair-play public-only CLI bridge over the stdio FindingBench RAB MCP server.

Allows agents without direct native MCP tool injection (e.g. swarm workers pinned
to a specific model) to play FindingBench RAB v1 strictly through the 4 public
MCP tools (reset_episode, observe, act, episode_status). Only public MCP JSON
metadata and returned public PNGs are exposed.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import socket
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ALLOWED_TOOLS = {"reset_episode", "observe", "act", "episode_status"}
ROOT = Path(__file__).resolve().parents[1]


async def _serve(sock_path: Path, view_dir: Path, out_dir: Path) -> None:
    view_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()

    env = dict(os.environ, FINDINGBENCH_RAB_MCP_OUT=str(out_dir.resolve()))
    params = StdioServerParameters(
        command=str(ROOT / "mcp/run_findingbench_rab_mcp.sh"),
        args=[],
        cwd=str(ROOT),
        env=env,
    )
    img_counter = 0

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            lock = asyncio.Lock()
            stop_event = asyncio.Event()

            async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                nonlocal img_counter
                try:
                    raw = await reader.readline()
                    if not raw:
                        return
                    req = json.loads(raw.decode("utf-8"))
                    tool = req.get("tool")
                    if tool == "__shutdown__":
                        writer.write(b'{"ok":true,"shutdown":true}\n')
                        await writer.drain()
                        stop_event.set()
                        return
                    if tool not in ALLOWED_TOOLS:
                        resp = {"ok": False, "error": f"Tool not allowed: {tool}"}
                    else:
                        async with lock:
                            res = await session.call_tool(tool, req.get("args") or {})
                        texts = [c.text for c in res.content if c.type == "text"]
                        images = [c for c in res.content if c.type == "image"]
                        payload = json.loads(texts[0]) if texts else {}
                        saved = []
                        for idx, im in enumerate(images):
                            img_counter += 1
                            tag = "scene" if idx == 0 else "check"
                            p = view_dir / f"obs_{img_counter:04d}_{tag}.png"
                            p.write_bytes(base64.b64decode(im.data))
                            saved.append(str(p))
                        resp = {"ok": not res.isError, "payload": payload, "images": saved}
                    writer.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
                    await writer.drain()
                except Exception as e:
                    writer.write((json.dumps({"ok": False, "error": str(e)}) + "\n").encode("utf-8"))
                    await writer.drain()
                finally:
                    writer.close()

            server = await asyncio.start_unix_server(handle, path=str(sock_path))
            print(f"[bridge] ready on {sock_path}", flush=True)
            async with server:
                await stop_event.wait()


def _call(sock_path: Path, tool: str, args: dict) -> None:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(420.0)
    s.connect(str(sock_path))
    s.sendall((json.dumps({"tool": tool, "args": args}) + "\n").encode("utf-8"))
    chunks = []
    while True:
        buf = s.recv(65536)
        if not buf:
            break
        chunks.append(buf)
        if b"\n" in buf:
            break
    s.close()
    print(b"".join(chunks).decode("utf-8").strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("serve")
    sp.add_argument("--sock", type=Path, required=True)
    sp.add_argument("--view-dir", type=Path, required=True)
    sp.add_argument("--out-dir", type=Path, required=True)
    cp = sub.add_parser("call")
    cp.add_argument("--sock", type=Path, required=True)
    cp.add_argument("tool", choices=[*sorted(ALLOWED_TOOLS), "__shutdown__"])
    cp.add_argument("json_args", nargs="?", default="{}")
    args = ap.parse_args()
    if args.cmd == "serve":
        asyncio.run(_serve(args.sock, args.view_dir, args.out_dir))
    else:
        _call(args.sock, args.tool, json.loads(args.json_args))


if __name__ == "__main__":
    main()
