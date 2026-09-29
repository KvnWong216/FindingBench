#!/usr/bin/env python3
"""FindingBench interactive play server — the human entry point.

    CUDA_VISIBLE_DEVICES=5 python scripts/serve_play.py [--scenario knife_search_001] [--port 8090]

Boots the real simulator (BEHAVIOR-1K + OmniGibson), then serves a browser
UI on http://0.0.0.0:<port>:

  - egocentric head-cam view, with a small task card in the TOP-LEFT corner
    (instruction + step budget + last action verdict)
  - BOTTOM bar: one square button per available skill type — icon in the
    center, small caption underneath (door icon + "open" for OPEN, etc.).
    Button layout DENSITY tracks the size of the grounded action space A_t:
    a handful of skills -> large sparse buttons, many -> a dense grid.
  - click behavior: object-directed skills (OPEN/CLOSE/GRASP) execute
    directly when exactly one target is grounded; if a skill needs a
    parameter (NAV destination, or several candidate objects) a dialog pops
    up listing the currently grounded targets to choose from.
  - RESET button restarts the episode from the compiled initial state.

No embodiment parameters or interaction regions are shown: the player sees
exactly what the benchmark exposes — the egocentric view and A_t.
"""

import argparse
import base64
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SKILL_ORDER = ["NAV", "OPEN", "CLOSE", "GRASP", "PLACE"]

ICONS = {
    "NAV": '<svg viewBox="0 0 48 48"><path d="M24 6 38 40 24 32 10 40Z" fill="none" stroke="#0f6e4e" stroke-width="3" stroke-linejoin="round"/><circle cx="24" cy="23" r="3" fill="#0f6e4e"/></svg>',
    "OPEN": '<svg viewBox="0 0 48 48"><path d="M12 6h20a2 2 0 0 1 2 2v32a2 2 0 0 1-2 2H12" fill="none" stroke="#0f6e4e" stroke-width="3"/><path d="M34 14h8v20h-8" fill="none" stroke="#0f6e4e" stroke-width="3"/><circle cx="29" cy="24" r="2.5" fill="#0f6e4e"/></svg>',
    "CLOSE": '<svg viewBox="0 0 48 48"><rect x="12" y="6" width="24" height="36" rx="2" fill="none" stroke="#0f6e4e" stroke-width="3"/><circle cx="30" cy="24" r="2.5" fill="#0f6e4e"/><path d="M40 8l-6 6M40 40l-6-6" stroke="#b3261e" stroke-width="3"/></svg>',
    "GRASP": '<svg viewBox="0 0 48 48"><path d="M14 26V14a3 3 0 0 1 6 0v8m0-10a3 3 0 0 1 6 0v10m0-12a3 3 0 0 1 6 0v12m0-8a3 3 0 0 1 6 0v10c0 8-5 14-12 14h-2c-6 0-10-4-10-10v-6" fill="none" stroke="#0f6e4e" stroke-width="3" stroke-linecap="round"/></svg>',
    "PLACE": '<svg viewBox="0 0 48 48"><path d="M24 8v18m0 0-7-7m7 7 7-7" fill="none" stroke="#0f6e4e" stroke-width="3" stroke-linecap="round"/><path d="M8 32v6a4 4 0 0 0 4 4h24a4 4 0 0 0 4-4v-6" fill="none" stroke="#0f6e4e" stroke-width="3" stroke-linecap="round"/></svg>',
}

INTERACTIVE_TYPES = {"OPEN", "CLOSE", "GRASP"}  # direct-execute when unambiguous

SKILL_RE = re.compile(r"^([A-Z]+)\((.*)\)$")


def parse_skills(available: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for entry in available or []:
        m = SKILL_RE.match(entry)
        if m:
            grouped.setdefault(m.group(1), []).append(m.group(2))
    return grouped


class PlayState:
    """Shared state between the sim worker and the HTTP handlers.

    OmniGibson/Kit calls MUST run on the thread that created the app, so the
    HTTP handlers only ENQUEUE actions; the main thread (which booted the
    simulator) executes them. /act therefore returns {"queued": true} and the
    UI polls /state until the sequence counter advances.
    """

    def __init__(self, scenario_path: str, seed: int):
        self.scenario_path = scenario_path
        self.seed = seed
        self.lock = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.env = None
        self.snapshot: dict = {}
        self.last_result: dict | None = None
        self.seq = 0
        self._pending: tuple[str, str] | None = None

    def boot(self) -> None:
        from rummagebench.environment.env import InteractiveSearchEnv

        try:
            env = InteractiveSearchEnv(self.scenario_path, seed=self.seed)
            reset = env.reset()
            self.env = env
            self.snapshot = self._pack(reset["observation"], status="RUNNING")
            self.ready = True
            print("== simulator ready", flush=True)
        except Exception as e:  # surface boot failures to the browser
            self.error = f"{type(e).__name__}: {e}"
            print(f"== BOOT FAILED: {self.error}", flush=True)
            raise

    def _pack(self, observation: dict, status: str) -> dict:
        return {
            "instruction": observation["instruction"],
            "planning_step": observation["planning_step"],
            "max_planning_steps": observation["max_planning_steps"],
            "available_skills": observation["available_skills"],
            "previous_action_result": observation.get("previous_action_result"),
            "image_png_b64": observation.get("image_png_b64"),
            "episode_status": status,
        }

    # ------------------------------------------------------- HTTP-facing API

    def enqueue(self, skill: str, target: str) -> dict:
        with self.lock:
            if not self.ready:
                return {"error": "simulator not ready"}
            if self._pending is not None:
                return {"error": "an action is already executing", "queued": False}
            self._pending = (skill, target)
            return {"queued": True}

    def reset_request(self) -> dict:
        with self.lock:
            if not self.ready:
                return {"error": "simulator not ready"}
            if self._pending is not None:
                return {"error": "an action is already executing", "queued": False}
            self._pending = ("RESET", "")
            return {"queued": True}

    def state_payload(self) -> dict:
        payload = dict(self.snapshot)
        payload["ready"] = self.ready
        payload["error"] = self.error
        payload["last_result"] = self.last_result
        payload["seq"] = self.seq
        payload["pending"] = self._pending is not None
        return payload

    # ---------------------------------------------------- main-thread worker

    def run_worker(self) -> None:
        """Runs on the MAIN thread forever: executes queued sim calls and
        refreshes the egocentric view between actions."""
        import time

        last_view = 0.0
        while True:
            now = time.time()
            pending = self._pending
            if pending is not None:
                skill, target = pending
                self._pending = None
                try:
                    if skill == "RESET":
                        reset = self.env.reset()
                        self.snapshot = self._pack(reset["observation"], status="RUNNING")
                        self.last_result = {
                            "skill": None, "target": None, "status": "RESET",
                            "reason": None, "episode_status": "RUNNING",
                        }
                    else:
                        action = {
                            "skill": skill,
                            "target": {
                                "type": "place" if skill == "NAV" else "entity",
                                "value": target,
                            },
                        }
                        result = self.env.step(action)
                        self.snapshot = self._pack(
                            result["observation"], status=result["episode_status"]
                        )
                        self.last_result = {
                            "skill": skill,
                            "target": target,
                            "status": result["status"],
                            "reason": result["reason"],
                            "episode_status": result["episode_status"],
                            "planning_step": result["planning_step"],
                            "state_update": result["state_update"],
                        }
                    self.seq += 1
                    last_view = now  # action responses carry a fresh frame
                except Exception as e:
                    self.last_result = {
                        "skill": skill, "target": target, "status": "ERROR",
                        "reason": f"{type(e).__name__}: {e}",
                    }
                    self.seq += 1
                    print(f"== action error: {e}", flush=True)
            elif self.ready and now - last_view > 5.0:
                # idle refresh: the view keeps tracking the settling scene
                try:
                    self.snapshot = self._pack(
                        self.env._observation(), status=self.snapshot.get("episode_status", "RUNNING")
                    )
                    self.seq += 1
                except Exception:
                    pass
                last_view = now
            time.sleep(0.2)


class Handler(BaseHTTPRequestHandler):
    state: PlayState  # injected via server attribute

    # silence per-request logging
    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/state":
            self._json(self.state.state_payload())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        if self.path == "/act":
            skill, target = body.get("skill"), body.get("target")
            if not skill or target is None:
                self._json({"error": "skill and target required"}, 400)
                return
            self._json(self.state.enqueue(skill, target))
        elif self.path == "/reset":
            self._json(self.state.reset_request())
        else:
            self._json({"error": "not found"}, 404)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FindingBench — Play</title>
<style>
  :root { --ink:#182028; --accent:#0f6e4e; --bad:#b3261e; --warn:#b26a00; }
  html,body { background:#f2f4f3; color:var(--ink); margin:0;
              font-family:'Segoe UI',system-ui,sans-serif; height:100%; }
  .wrap { max-width:900px; margin:0 auto; padding:10px 12px 30px; }
  header { display:flex; align-items:baseline; gap:14px; padding:6px 2px 10px; }
  header h1 { font-size:1.15em; margin:0; color:var(--accent); }
  header .st { font-size:0.9em; color:#555; }
  header button { margin-left:auto; }
  #view { position:relative; background:#000; border-radius:8px; overflow:hidden;
          min-height:340px; }
  #view img { display:block; width:100%; }
  #task { position:absolute; top:10px; left:10px; background:rgba(10,14,12,.82);
          color:#e8ffe9; border-radius:8px; padding:8px 12px; max-width:62%;
          font-size:0.82em; line-height:1.45; }
  #task .t { color:#7dffb0; font-weight:600; }
  #verdict { position:absolute; top:10px; right:10px; font-size:0.75em;
             background:rgba(10,14,12,.82); color:#ffd; border-radius:8px;
             padding:6px 10px; max-width:34%; text-align:right; }
  .verdict-ok { color:#7dffb0; } .verdict-bad { color:#ff9c93; }
  #skills { margin-top:12px; min-height:110px; }
  #skills .lbl { font-size:0.78em; color:#667; margin-bottom:6px; }
  #btns { display:grid; gap:10px; justify-content:start; }
  #btns.d1 { grid-template-columns:repeat(auto-fill, 96px); }
  #btns.d2 { grid-template-columns:repeat(auto-fill, 76px); }
  #btns.d3 { grid-template-columns:repeat(auto-fill, 60px); gap:6px; }
  .sk { width:100%; aspect-ratio:1; border:1.5px solid #cdd6d1; border-radius:12px;
        background:#fff; cursor:pointer; display:flex; flex-direction:column;
        align-items:center; justify-content:center; gap:2px; padding:4px;
        transition:transform .06s, border-color .06s; position:relative; }
  .sk:hover { border-color:var(--accent); transform:translateY(-2px); }
  .sk:active { transform:translateY(0); }
  .sk svg { width:62%; height:62%; }
  .sk .cap { font-size:0.62em; color:#445; letter-spacing:.04em; }
  .sk .n { position:absolute; top:3px; right:5px; font-size:0.58em; color:#899; }
  .sk:disabled { opacity:.45; cursor:wait; }
  #modal { position:fixed; inset:0; background:rgba(20,26,24,.55); display:none;
           align-items:center; justify-content:center; }
  #modal .box { background:#fff; border-radius:12px; padding:16px 18px;
                max-width:520px; width:92%; max-height:70vh; overflow:auto; }
  #modal h3 { margin:2px 0 10px; font-size:1em; }
  #modal .opt { display:block; width:100%; text-align:left; margin:6px 0;
                padding:9px 12px; border:1.2px solid #cdd6d1; border-radius:8px;
                background:#fafcfb; cursor:pointer; font-size:0.9em;
                font-family:Menlo,Consolas,monospace; }
  #modal .opt:hover { border-color:var(--accent); background:#f0f7f4; }
  #modal .cancel { background:none; border:none; color:#777; cursor:pointer;
                   margin-top:8px; }
  #boot { padding:40px; text-align:center; color:#567; font-size:0.95em; }
  .spin { display:inline-block; width:18px; height:18px; border:3px solid #cde;
          border-top-color:var(--accent); border-radius:50%;
          animation:sp 1s linear infinite; vertical-align:-4px; margin-right:8px; }
  @keyframes sp { to { transform:rotate(360deg); } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>FindingBench · Play</h1>
    <span class="st" id="status">booting…</span>
    <button id="reset">RESET</button>
  </header>
  <div id="view">
    <div id="boot"><span class="spin"></span>launching the simulator (scene build takes a few minutes)…</div>
    <img id="cam" alt="egocentric view" style="display:none">
    <div id="task" style="display:none">
      <span class="t" id="instruction"></span><br>
      <span id="steps"></span>
    </div>
    <div id="verdict" style="display:none"></div>
  </div>
  <div id="skills">
    <div class="lbl">available skills (A<sub>t</sub>) — the grounded action space</div>
    <div id="btns"></div>
  </div>
</div>
<div id="modal"><div class="box">
  <h3 id="m-title"></h3>
  <div id="m-opts"></div>
  <button class="cancel" id="m-cancel">cancel</button>
</div></div>
<script>
const INTERACTIVE = ["OPEN", "CLOSE", "GRASP"];
const ICONS = __ICONS__;
let busy = false, cur = null, busyKey = null;

function $g(id) { return document.getElementById(id); }

function setBusy(b) {
  busy = b;
  document.querySelectorAll(".sk").forEach(el => el.disabled = b);
  $g("reset").disabled = b;
}

function verdictHtml(lr) {
  if (!lr) return "";
  if (lr.status === "RESET") return "episode reset";
  const ok = lr.status === "SUCCESS";
  const cls = ok ? "verdict-ok" : "verdict-bad";
  let s = `<span class="${cls}">${lr.skill}(${lr.target ?? ""})<br>${ok ? "SUCCESS" : "FAILURE · " + lr.reason}</span>`;
  if (lr.episode_status && lr.episode_status !== "RUNNING")
    s += `<br>EPISODE: <b>${lr.episode_status}</b>`;
  return s;
}

function render(st) {
  cur = st;
  if (st.error) { $g("boot").innerHTML = "BOOT FAILED: " + st.error; return; }
  if (!st.ready) return;
  // actions are queued server-side: busy clears when last_result changes
  if (busy && busyKey !== null) {
    const k = JSON.stringify(st.last_result);
    if (k !== busyKey) { busyKey = null; setBusy(false); }
    else if (st.pending === false) { busyKey = null; setBusy(false); }
  }
  $g("boot").style.display = "none";
  $g("cam").style.display = "block";
  $g("task").style.display = "block";
  if (st.image_png_b64) $g("cam").src = "data:image/png;base64," + st.image_png_b64;
  $g("instruction").textContent = st.instruction;
  $g("steps").textContent = "step " + st.planning_step + " / " + st.max_planning_steps;
  const es = st.episode_status;
  $g("status").textContent = "episode: " + es;
  const v = $g("verdict");
  const vh = verdictHtml(st.last_result);
  v.style.display = vh ? "block" : "none";
  v.innerHTML = vh;
  renderSkills(st.available_skills);
}

function renderSkills(available) {
  const grouped = {};
  (available || []).forEach(e => {
    const m = e.match(/^([A-Z]+)\\((.*)\\)$/);
    if (m) (grouped[m[1]] = grouped[m[1]] || []).push(m[2]);
  });
  const types = __SKILL_ORDER__.filter(t => grouped[t]);
  const nInstances = (available || []).length;
  const dens = nInstances <= 4 ? "d1" : (nInstances <= 8 ? "d2" : "d3");
  const btns = $g("btns");
  btns.className = dens;
  btns.innerHTML = "";
  types.forEach(t => {
    const b = document.createElement("button");
    b.className = "sk";
    b.disabled = busy;
    b.innerHTML = ICONS[t] + `<span class="cap">${t.toLowerCase()}</span>` +
      (grouped[t].length > 1 ? `<span class="n">×${grouped[t].length}</span>` : "");
    b.onclick = () => onSkill(t, grouped[t]);
    btns.appendChild(b);
  });
  if (!types.length)
    btns.innerHTML = "<span style='color:#899;font-size:.85em'>no grounded skills — episode over? press RESET</span>";
}

function onSkill(type, targets) {
  if (busy) return;
  // interactive object skill with exactly one grounded target: direct execute
  if (INTERACTIVE.includes(type) && targets.length === 1) {
    act(type, targets[0]);
    return;
  }
  // otherwise (NAV/PLACE destinations, or several candidate objects):
  // pop a dialog to provide the parameter
  const m = $g("modal");
  $g("m-title").textContent = type + " — choose target";
  const opts = $g("m-opts");
  opts.innerHTML = "";
  targets.forEach(t => {
    const b = document.createElement("button");
    b.className = "opt";
    b.textContent = t;
    b.onclick = () => { m.style.display = "none"; act(type, t); };
    opts.appendChild(b);
  });
  m.style.display = "flex";
}

async function act(skill, target) {
  if (busy) return;
  setBusy(true);
  busyKey = JSON.stringify(cur ? cur.last_result : null);
  try {
    const r = await fetch("/act", { method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ skill, target }) });
    const j = await r.json();
    if (j.error) { alert(j.error); busyKey = null; setBusy(false); }
  } catch (e) { busyKey = null; setBusy(false); }
}

$g("reset").onclick = async () => {
  if (busy) return;
  setBusy(true);
  busyKey = JSON.stringify(cur ? cur.last_result : null);
  try {
    const r = await fetch("/reset", { method: "POST" });
    const j = await r.json();
    if (j.error) { alert(j.error); busyKey = null; setBusy(false); }
  } catch (e) { busyKey = null; setBusy(false); }
};
$g("m-cancel").onclick = () => $g("modal").style.display = "none";

async function refresh() {
  try {
    const r = await fetch("/state");
    render(await r.json());
  } catch (e) { /* transient */ }
}
refresh();
setInterval(refresh, 2500);
</script>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenarios/knife_search_001/scenario.yaml")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scenario_path = args.scenario
    if not Path(scenario_path).exists():
        scenario_path = str(REPO / args.scenario)

    state = PlayState(scenario_path, seed=args.seed)
    page = PAGE.replace("__ICONS__", json.dumps(ICONS)).replace(
        "__SKILL_ORDER__", json.dumps(SKILL_ORDER)
    )
    globals()["PAGE"] = page
    Handler.state = state

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"== UI: http://0.0.0.0:{args.port}  (booting the simulator…)", flush=True)

    try:
        state.boot()
    except Exception:
        return 1

    print("== ready — open the UI in a browser and play. Ctrl+C to quit.", flush=True)
    try:
        state.run_worker()
    except KeyboardInterrupt:
        pass
    finally:
        state.env.close()
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
