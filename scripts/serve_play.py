#!/usr/bin/env python3
"""FindingBench human play UI — the final visual interaction protocol.

    CUDA_VISIBLE_DEVICES=5 python scripts/serve_play.py [--scenario ...] [--port 8090]

Same information boundary as the evaluated agent (§21): the player sees the
current RGB, the task, eight skill buttons and four-class feedback — never
entity names, target lists, feasibility state or simulator labels.

    MOVE  -> dialog asking signed distance in cm
    TURN  -> dialog asking signed angle in degrees
    OPEN / CLOSE / GRASP / PLACE / OBSERVE
          -> crosshair mode: click the current RGB; the normalized point plus
             the current frame_id are submitted automatically
    REPORT_DONE -> no parameter

OBSERVE results render as a non-interactive gallery strip; auxiliary views
are never valid point-reference frames.

--assist (human debugging only) breaks that boundary on purpose: while a point
skill is selected, objects the skills can act on are highlighted under the
cursor, and a click anywhere on a highlight is snapped to the most clickable
pixel of that object (thin objects such as a knife are hard to hit otherwise).
Never use it for evaluation.
"""

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ICONS = {
    "MOVE": '<svg viewBox="0 0 48 48"><path d="M8 24h26m0 0-8-8m8 8-8 8M38 14v20" fill="none" stroke="#0f6e4e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "TURN": '<svg viewBox="0 0 48 48"><path d="M38 24a14 14 0 1 1-6-11.5M32 6v7h7" fill="none" stroke="#0f6e4e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "OPEN": '<svg viewBox="0 0 48 48"><path d="M12 6h20a2 2 0 0 1 2 2v32a2 2 0 0 1-2 2H12" fill="none" stroke="#0f6e4e" stroke-width="3"/><path d="M34 14h8v20h-8" fill="none" stroke="#0f6e4e" stroke-width="3"/><circle cx="29" cy="24" r="2.5" fill="#0f6e4e"/></svg>',
    "CLOSE": '<svg viewBox="0 0 48 48"><rect x="12" y="6" width="24" height="36" rx="2" fill="none" stroke="#0f6e4e" stroke-width="3"/><circle cx="30" cy="24" r="2.5" fill="#0f6e4e"/><path d="M40 8l-6 6M40 40l-6-6" stroke="#b3261e" stroke-width="3"/></svg>',
    "GRASP": '<svg viewBox="0 0 48 48"><path d="M14 26V14a3 3 0 0 1 6 0v8m0-10a3 3 0 0 1 6 0v10m0-12a3 3 0 0 1 6 0v12m0-8a3 3 0 0 1 6 0v10c0 8-5 14-12 14h-2c-6 0-10-4-10-10v-6" fill="none" stroke="#0f6e4e" stroke-width="3" stroke-linecap="round"/></svg>',
    "PLACE": '<svg viewBox="0 0 48 48"><path d="M24 8v18m0 0-7-7m7 7 7-7" fill="none" stroke="#0f6e4e" stroke-width="3" stroke-linecap="round"/><path d="M8 32v6a4 4 0 0 0 4 4h24a4 4 0 0 0 4-4v-6" fill="none" stroke="#0f6e4e" stroke-width="3" stroke-linecap="round"/></svg>',
    "OBSERVE": '<svg viewBox="0 0 48 48"><path d="M4 24s8-12 20-12 20 12 20 12-8 12-20 12S4 24 4 24Z" fill="none" stroke="#0f6e4e" stroke-width="3"/><circle cx="24" cy="24" r="6" fill="none" stroke="#0f6e4e" stroke-width="3"/></svg>',
    "REPORT_DONE": '<svg viewBox="0 0 48 48"><circle cx="24" cy="24" r="17" fill="none" stroke="#0f6e4e" stroke-width="3"/><path d="M15 24l6 6 12-13" fill="none" stroke="#0f6e4e" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
}

POINT_SKILLS = ["OPEN", "CLOSE", "GRASP", "PLACE", "OBSERVE"]

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FindingBench — Play</title>
<style>
  :root { --ink:#182028; --accent:#0f6e4e; --bad:#b3261e; --warn:#b26a00; }
  html,body { background:#f2f4f3; color:var(--ink); margin:0;
              font-family:'Segoe UI',system-ui,sans-serif; }
  .wrap { max-width:1280px; margin:0 auto; padding:10px 12px 30px; }
  header { display:flex; align-items:baseline; gap:14px; padding:6px 2px 10px; }
  header h1 { font-size:1.15em; margin:0; color:var(--accent); }
  header .st { font-size:0.9em; color:#555; }
  #view { position:relative; background:#000; border-radius:8px; overflow:hidden; }
  #view img { display:block; width:100%; }
  #view.selecting { cursor:crosshair; outline:3px solid var(--warn); }
  #hl { position:absolute; left:0; top:0; width:100%; height:100%;
        pointer-events:none; display:none; }
  #task { position:absolute; top:10px; left:10px; background:rgba(10,14,12,.82);
          color:#e8ffe9; border-radius:8px; padding:8px 12px; max-width:62%;
          font-size:0.82em; line-height:1.45; }
  #task .t { color:#7dffb0; font-weight:600; }
  #verdict { position:absolute; top:10px; right:10px; font-size:0.75em;
             background:rgba(10,14,12,.82); border-radius:8px;
             padding:6px 10px; max-width:40%; text-align:right; color:#eee; }
  .v-executed { color:#7dffb0; } .v-invalid { color:#ffd479; }
  .v-capability { color:#9ecbff; } .v-unsafe { color:#ff9c93; }
  #skills { margin-top:12px; }
  #skills .lbl { font-size:0.78em; color:#667; margin-bottom:6px; }
  #btns { display:flex; flex-wrap:wrap; gap:10px; }
  .sk { width:84px; aspect-ratio:1; border:1.5px solid #cdd6d1; border-radius:12px;
        background:#fff; cursor:pointer; display:flex; flex-direction:column;
        align-items:center; justify-content:center; gap:2px; padding:4px;
        transition:transform .06s, border-color .06s; }
  .sk:hover { border-color:var(--accent); transform:translateY(-2px); }
  .sk.active { border-color:var(--warn); background:#fff8ef; }
  .sk svg { width:58%; height:58%; }
  .sk .cap { font-size:0.58em; color:#445; letter-spacing:.03em; text-align:center; }
  .sk:disabled { opacity:.45; cursor:wait; }
  #gallery { display:none; margin-top:10px; gap:8px; overflow-x:auto; padding-bottom:4px; }
  #gallery img { height:130px; border:1px solid #ccc; border-radius:6px; pointer-events:none; }
  #modal { position:fixed; inset:0; background:rgba(20,26,24,.55); display:none;
           align-items:center; justify-content:center; }
  #modal .box { background:#fff; border-radius:12px; padding:18px;
                width:min(420px, 92%); }
  #modal h3 { margin:2px 0 12px; font-size:1em; }
  #modal input { width:100%; box-sizing:border-box; padding:10px; font-size:1em;
                 border:1.2px solid #cdd6d1; border-radius:8px; margin-bottom:4px; }
  #modal .hint { font-size:0.78em; color:#667; margin:2px 0 10px; }
  #modal .row { display:flex; gap:8px; }
  #modal button { flex:1; padding:10px; border-radius:8px; border:1.2px solid #cdd6d1;
                  background:#fafcfb; cursor:pointer; font-size:0.9em; }
  #modal .go { background:var(--accent); color:#fff; border-color:var(--accent); }
  #reset { margin-left:auto; padding:6px 14px; border-radius:8px; cursor:pointer;
           border:1.2px solid #cdd6d1; background:#fff; color:var(--ink); font-size:0.85em; }
  #reset:hover { border-color:var(--accent); }
  #reset:disabled { opacity:.45; cursor:wait; }
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
    <button id="reset" disabled title="restart the episode from the initial state">↺ reset</button>
  </header>
  <div id="view">
    <div id="boot"><span class="spin"></span>launching the simulator (a few minutes)…</div>
    <img id="cam" alt="egocentric view" style="display:none"
         data-tip="click a point to target the selected skill">
    <canvas id="hl"></canvas>
    <div id="task" style="display:none">
      <span class="t" id="instruction"></span><br>
      <span id="steps"></span><br>
      <span id="mode" style="color:#ffd479"></span>
    </div>
    <div id="verdict" style="display:none"></div>
  </div>
  <div id="gallery"></div>
  <div id="skills">
    <div class="lbl">skill library — object skills target the current frame by click</div>
    <div id="btns"></div>
  </div>
</div>
<div id="modal"><div class="box">
  <h3 id="m-title"></h3>
  <div id="m-body"></div>
</div></div>
<script>
const ICONS = __ICONS__;
const POINT_SKILLS = ["OPEN", "CLOSE", "GRASP", "PLACE", "OBSERVE"];
const SKILLS = ["MOVE", "TURN", "OPEN", "CLOSE", "GRASP", "PLACE", "OBSERVE", "REPORT_DONE"];
let cur = null, pendingSkill = null, busy = false, lastVerdictKey = null;
// --assist: label image of actionable objects for the current frame
let hints = null, hintPixels = null, hintKey = null, hoverIdx = 0;
const hintCanvas = document.createElement("canvas");

function loadHints(h) {
  const key = h ? h.frame_id + ":" + h.png_b64.length : null;
  if (key === hintKey) return;
  hintKey = key; hints = h; hintPixels = null; hoverIdx = 0;
  drawHighlight();
  if (!h) return;
  const im = new Image();
  im.onload = () => {
    if (hintKey !== key) return;
    hintCanvas.width = h.width; hintCanvas.height = h.height;
    const ctx = hintCanvas.getContext("2d");
    ctx.drawImage(im, 0, 0);
    const d = ctx.getImageData(0, 0, h.width, h.height).data;
    hintPixels = new Uint8Array(h.width * h.height);
    for (let i = 0; i < hintPixels.length; i++) hintPixels[i] = d[4 * i];
    drawHighlight();
  };
  im.src = "data:image/png;base64," + h.png_b64;
}

function labelAt(x, y) {
  if (!hintPixels) return 0;
  const u = Math.min(hints.width - 1, Math.floor(x * hints.width));
  const v = Math.min(hints.height - 1, Math.floor(y * hints.height));
  return hintPixels[v * hints.width + u];
}

// selecting: every actionable object gets a faint tint, the hovered one a
// strong one; nothing is drawn outside point-skill mode
function drawHighlight() {
  const c = $g("hl");
  if (!hints || !hintPixels || !pendingSkill) { c.style.display = "none"; return; }
  c.width = hints.width; c.height = hints.height;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(hints.width, hints.height);
  for (let i = 0; i < hintPixels.length; i++) {
    const l = hintPixels[i];
    if (!l) continue;
    const hot = l === hoverIdx;
    img.data[4 * i] = hot ? 255 : 120;
    img.data[4 * i + 1] = hot ? 214 : 220;
    img.data[4 * i + 2] = hot ? 0 : 255;
    img.data[4 * i + 3] = hot ? 150 : 55;
  }
  ctx.putImageData(img, 0, 0);
  c.style.display = "block";
}

function camPoint(ev) {
  const rect = $g("cam").getBoundingClientRect();
  return [(ev.clientX - rect.left) / rect.width, (ev.clientY - rect.top) / rect.height];
}
$g("cam").addEventListener("mousemove", (ev) => {
  if (!pendingSkill || !hintPixels) return;
  const [x, y] = camPoint(ev);
  const idx = labelAt(x, y);
  if (idx !== hoverIdx) { hoverIdx = idx; drawHighlight(); }
});
$g("cam").addEventListener("mouseleave", () => {
  if (hoverIdx) { hoverIdx = 0; drawHighlight(); }
});

function $g(id) { return document.getElementById(id); }
function setBusy(b) {
  busy = b;
  document.querySelectorAll(".sk").forEach(el => el.disabled = b || (cur && cur.episode_status !== "RUNNING"));
}
const VCLASS = { EXECUTED: "v-executed", INVALID_ACTION: "v-invalid",
                 OUT_OF_CAPABILITY: "v-capability", UNSAFE: "v-unsafe" };

function renderSkills() {
  const btns = $g("btns");
  btns.innerHTML = "";
  SKILLS.forEach(t => {
    const b = document.createElement("button");
    b.className = "sk" + (pendingSkill === t ? " active" : "");
    b.disabled = busy || (cur && cur.episode_status !== "RUNNING");
    b.innerHTML = ICONS[t] + `<span class="cap">${t.replace("_", " ").toLowerCase()}</span>`;
    b.onclick = () => onSkill(t);
    btns.appendChild(b);
  });
}

function onSkill(t) {
  if (busy || (cur && cur.episode_status !== "RUNNING")) return;
  if (t === "REPORT_DONE") { submit({ skill: "REPORT_DONE" }); return; }
  if (t === "MOVE") return askNumber(t, "signed distance in cm", "positive = forward, negative = backward", 40);
  if (t === "TURN") return askNumber(t, "signed angle in degrees", "positive = left (CCW), negative = right (CW)", 45);
  // point skills: enter crosshair mode
  pendingSkill = (pendingSkill === t) ? null : t;
  $g("view").classList.toggle("selecting", pendingSkill !== null);
  $g("mode").textContent = !pendingSkill ? "" : hints
    ? `${pendingSkill}: click a highlighted object`
    : `${pendingSkill}: click a point on the image`;
  hoverIdx = 0;
  drawHighlight();
  renderSkills();
}

function askNumber(skill, label, hint, def) {
  const m = $g("modal");
  $g("m-title").textContent = skill;
  $g("m-body").innerHTML =
    `<input id="m-val" type="number" step="5" value="${def}">` +
    `<div class="hint">${hint} — legal magnitude ${skill === "MOVE" ? "5..100 cm" : "5..180 deg"}</div>` +
    `<div class="row"><button id="m-cancel">cancel</button><button id="m-go" class="go">execute</button></div>`;
  m.style.display = "flex";
  $g("m-cancel").onclick = () => m.style.display = "none";
  $g("m-go").onclick = () => {
    const v = parseFloat($g("m-val").value);
    m.style.display = "none";
    if (Number.isNaN(v)) return;
    submit(skill === "MOVE" ? { skill, distance_cm: v } : { skill, angle_deg: v });
  };
}

$g("cam").addEventListener("click", (ev) => {
  if (!pendingSkill || busy || !cur) return;
  let [x, y] = camPoint(ev);
  // --assist: a click on a highlight snaps to that object's clickable pixel
  const idx = hints && hints.frame_id === cur.frame_id ? labelAt(x, y) : 0;
  if (idx) { x = hints.targets[idx - 1].x; y = hints.targets[idx - 1].y; }
  const skill = pendingSkill;
  pendingSkill = null;
  hoverIdx = 0;
  drawHighlight();
  $g("view").classList.remove("selecting");
  $g("mode").textContent = "";
  renderSkills();
  submit({ skill, point: { frame_id: cur.frame_id, x: +x.toFixed(4), y: +y.toFixed(4) } });
});

async function submit(action) {
  if (cur && cur.episode_status !== "RUNNING") return;
  setBusy(true);
  lastVerdictKey = cur ? JSON.stringify(cur.feedback) : null;
  try {
    const r = await fetch("/act", { method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(action) });
    const j = await r.json();
    if (j.error) alert(j.error);
  } catch (e) { /* transient */ }
}

let resetBase = null;
$g("reset").onclick = async () => {
  if (!cur || !cur.ready || resetBase !== null) return;
  if (!confirm("Restart the episode from the beginning?")) return;
  resetBase = cur.resets;
  pendingSkill = null;
  $g("view").classList.remove("selecting");
  $g("mode").textContent = "";
  $g("reset").disabled = true;
  lastVerdictKey = null;
  setBusy(true);
  try {
    const j = await (await fetch("/reset", { method: "POST" })).json();
    if (j.error) { alert(j.error); resetBase = null; setBusy(false); }
  } catch (e) { resetBase = null; setBusy(false); }
};

function render(st) {
  cur = st;
  if (st.error) { $g("boot").innerHTML = "BOOT FAILED: " + st.error; return; }
  if (!st.ready) return;
  if (resetBase !== null && st.resets !== resetBase) {
    resetBase = null; lastVerdictKey = null; setBusy(false);
  }
  $g("reset").disabled = resetBase !== null;
  if (busy && lastVerdictKey !== null &&
      JSON.stringify(st.feedback) !== lastVerdictKey) {
    lastVerdictKey = null; setBusy(false);
  }
  if (st.pending === false && busy && lastVerdictKey !== null &&
      JSON.stringify(st.feedback) === lastVerdictKey) {
    // queued request vanished without feedback (engine error): unblock
    lastVerdictKey = null; setBusy(false);
  }
  $g("boot").style.display = "none";
  $g("cam").style.display = "block";
  $g("task").style.display = "block";
  if (st.image_png_b64) $g("cam").src = "data:image/png;base64," + st.image_png_b64;
  loadHints(st.hints || null);
  $g("instruction").textContent = st.instruction;
  $g("steps").textContent = "step " + st.planning_step + " / " + st.max_planning_steps;
  $g("status").textContent = "episode: " + st.episode_status;
  const v = $g("verdict");
  if (st.feedback) {
    const f = st.feedback;
    v.style.display = "block";
    v.innerHTML = `<span class="${VCLASS[f.code] || ""}">${f.code}</span>`;
  } else { v.style.display = "none"; }
  const gal = $g("gallery");
  if (st.observe_views && st.observe_views.length) {
    gal.style.display = "flex";
    gal.innerHTML = st.observe_views
      .map(vw => `<img src="data:image/png;base64,${vw.image_png_b64}">`).join("");
  } else { gal.style.display = "none"; gal.innerHTML = ""; }
  renderSkills();
}

async function refresh() {
  try { render(await (await fetch("/state")).json()); } catch (e) { /* transient */ }
}
renderSkills();
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


def click_hints(env) -> dict:
    """--assist overlay for the current frame: a label image (0 = nothing,
    i = i-th actionable object) plus one snap point per label. Entity names
    are not sent. Actionable = movable objects and openable furniture."""
    import io

    import numpy as np
    from PIL import Image

    session, backend = env._session, env._backend
    frame = session._store.get(session._store.current_id)
    seg = np.asarray(frame.instance_segmentation)
    robot = backend.robot_entity_names()
    by_entity: dict[str, list] = {}
    for inst in np.unique(seg):
        if inst == 0:
            continue
        entity = (backend.instance_to_entity(inst, frame)
                  if "instance_labels" in frame.meta else backend.instance_to_entity(inst))
        if entity is None or entity in robot:
            continue
        info = backend.describe_entity(entity)
        if info is None or (info.fixed_base and not info.openable):
            continue
        by_entity.setdefault(entity, []).append(inst)

    H, W = seg.shape
    labels = np.zeros((H, W), np.uint8)
    targets = []
    for entity, insts in list(by_entity.items())[:255]:
        labels[np.isin(seg, insts)] = len(targets) + 1
        # the click bridge needs one instance dominating an 11x11 patch: snap
        # to the pixel of the largest instance with the most own pixels around
        inst = max(insts, key=lambda i: int((seg == i).sum()))
        mask = seg == inst
        c = np.pad(np.pad(mask.astype(np.int32), 5).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        win = c[11:11 + H, 11:11 + W] - c[:H, 11:11 + W] - c[11:11 + H, :W] + c[:H, :W]
        win[~mask] = -1
        v, u = np.unravel_index(int(np.argmax(win)), win.shape)
        targets.append({"x": round((u + 0.5) / W, 4), "y": round((v + 0.5) / H, 4)})
    buf = io.BytesIO()
    Image.fromarray(labels).save(buf, format="PNG")
    return {"frame_id": frame.frame_id, "width": W, "height": H,
            "png_b64": base64.b64encode(buf.getvalue()).decode(), "targets": targets}


class PlayState:
    """Shared state. HTTP threads ENQUEUE; the main thread (which booted the
    simulator) executes — Kit calls deadlock off their creating thread."""

    def __init__(self, scenario_path: str, seed: int, start_anchor: str | None = None,
                 assist: bool = False):
        self.scenario_path = scenario_path
        self.seed = seed
        self.start_anchor = start_anchor
        self.assist = assist
        self.lock = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.env = None
        self.snapshot: dict = {"ready": False}
        self.seq = 0
        self.resets = 0
        self._pending: dict | None = None

    def boot(self) -> None:
        from rummagebench.environment.env import InteractiveSearchEnv

        try:
            env = InteractiveSearchEnv(self.scenario_path, seed=self.seed,
                                       mode="agent",
                                       trace_path="runs/play_trace.jsonl")
            if self.start_anchor:
                # debug: the scene is built (placements verified) from the
                # scenario spawn; episodes then start at another anchor
                scenario = env._scenario
                if self.start_anchor not in scenario.anchors:
                    raise ValueError(f"unknown anchor {self.start_anchor!r}; "
                                     f"choose from {sorted(scenario.anchors)}")
                # the AGENT session resets from its own deep copy
                for sc in (scenario, env._session._scenario):
                    sc.robot.init_anchor = self.start_anchor
            obs = env.reset()["observation"]
            self.env = env
            self.snapshot = obs
            self.snapshot.update({"ready": True, "pending": False,
                                  "episode_status": "RUNNING"})
            self._add_hints()
            self.ready = True
            print("== simulator ready", flush=True)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"== BOOT FAILED: {self.error}", flush=True)
            raise

    def _add_hints(self) -> None:
        if not self.assist:
            return
        try:
            self.snapshot["hints"] = click_hints(self.env)
        except Exception as e:
            self.snapshot.pop("hints", None)
            print(f"== hint error: {e}", flush=True)

    def enqueue(self, action: dict) -> dict:
        with self.lock:
            if not self.ready:
                return {"error": "simulator not ready"}
            if self.snapshot.get("episode_status", "RUNNING") != "RUNNING":
                return {"error": "episode is terminal; restart to play again"}
            if self._pending is not None:
                return {"error": "an action is already executing"}
            self._pending = action
            return {"queued": True}

    def request_reset(self) -> dict:
        # allowed after a terminal status: that is when players want it most
        with self.lock:
            if not self.ready:
                return {"error": "simulator not ready"}
            if self._pending is not None:
                return {"error": "an action is already executing"}
            self._pending = {"__reset__": True}
            return {"queued": True}

    def state_payload(self) -> dict:
        payload = dict(self.snapshot)
        payload["ready"] = self.ready
        payload["error"] = self.error
        payload["pending"] = self._pending is not None
        payload["seq"] = self.seq
        payload["resets"] = self.resets
        return payload

    def run_worker(self) -> None:
        import time

        last_view = time.time()
        while True:
            action, self._pending = self._pending, None
            if action is not None and action.get("__reset__"):
                try:
                    self.snapshot = self.env.reset()["observation"]
                    self.snapshot.update({"ready": True, "pending": False,
                                          "episode_status": "RUNNING"})
                    self._add_hints()
                    print("== episode reset", flush=True)
                except Exception as e:
                    print(f"== reset error: {e}", flush=True)
                self.resets += 1
                self.seq += 1
                last_view = time.time()
            elif action is not None:
                try:
                    result = self.env.step(action)
                    self.snapshot = result["observation"]
                    self.snapshot.update({"ready": True, "pending": False,
                                          "episode_status": result["episode_status"]})
                    self._add_hints()
                except Exception as e:
                    print(f"== action error: {e}", flush=True)
                self.seq += 1
                last_view = time.time()
            elif self.ready and time.time() - last_view > 5.0:
                self.seq += 1  # nudge the UI; snapshot unchanged between steps
                last_view = time.time()
            time.sleep(0.2)


class Handler(BaseHTTPRequestHandler):
    state: PlayState

    def log_message(self, *args):
        pass

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
            if not body.get("skill"):
                self._json({"error": "skill required"}, 400)
                return
            self._json(self.state.enqueue(body))
        elif self.path == "/reset":
            self._json(self.state.request_reset())
        else:
            self._json({"error": "not found"}, 404)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenarios/knife_search_001/scenario.yaml")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; loopback by default so acceptance "
                             "runs never expose an open port (use an explicit, "
                             "authorized interface only when needed)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-anchor", default=None,
                        help="debug: start each episode at this scenario anchor "
                             "instead of robot.init_anchor (scene setup still "
                             "uses init_anchor)")
    parser.add_argument("--assist", action="store_true",
                        help="human debugging only: highlight actionable objects "
                             "under the cursor and snap clicks onto them "
                             "(leaks segmentation; never for evaluation)")
    args = parser.parse_args()

    scenario_path = args.scenario
    if not Path(scenario_path).exists():
        scenario_path = str(REPO / args.scenario)

    state = PlayState(scenario_path, seed=args.seed, start_anchor=args.start_anchor,
                      assist=args.assist)
    Handler.state = state
    page = PAGE.replace("__ICONS__", json.dumps(ICONS))
    globals()["PAGE"] = page

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"== UI: http://{args.host}:{args.port}  (booting the simulator…)", flush=True)

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
