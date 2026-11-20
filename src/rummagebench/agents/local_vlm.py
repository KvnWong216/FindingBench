"""Public-only VLM client. The model lives in a separate local Python process."""
from __future__ import annotations
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from rummagebench.agents.policy_adapter import PolicyInfrastructureError

PANORAMA_CALIBRATION = """CAMERA CALIBRATION
Use English for all natural-language output; preserve protocol identifiers exactly.
Image 1 is a full 360-degree horizontal by 180-degree vertical equirectangular panorama.
Its left and right edges meet in the same rear direction; they are not separate walls.
Normalized image coordinates x,y both lie in [0,1], with origin at the top-left.
x=0.50 is reference-forward, x=0.25 is left, x=0.75 is right,
and x=0 or x=1 is behind the reference camera heading.
y=0 looks straight up, y=0.50 is the horizon, and y=1 looks straight down.
For this panorama only: yaw_deg = 180 - 360*x; pitch_deg = 90 - 180*y.
A negative yaw means right; it does NOT mean a negative image x coordinate.
These formulas do not apply to a VIEW detail, which has its own frame_id and normalized coordinates. CHECK inspection images have no scene coordinates and cannot be clicked. Use the frame_id of the scene image you click.
The full panorama is your primary visual input and source of scene information
on EVERY step. Ground your scene description in Image 1 even when Image 2 exists.
A successful VIEW adds a supplementary pinhole image for local inspection. After any non-VIEW action, including a failed action, that pinhole image is absent. CHECK instead adds a separate isolated-object inspection sheet with no actionable frame_id. Call VIEW again if a scene detail is needed.
VIEW never replaces Image 1 or moves the robot. Do not infer that an object is
absent from the scene merely because it is absent from the supplementary image.
Camera reference-forward need not equal body-forward; use the supplied body axes.
This calibration defines the projection only, not object positions or action feasibility.
"""

PROMPT_VERSION = 'public-skills-panorama-detail-on-demand-v4.2'
PROMPT = '''You control a robot using its current RGB image and public feedback.
Complete the instruction, then REPORT_DONE. Return exactly ONE JSON action.
Never invent object IDs, world coordinates, or hidden state.
TURN: {"skill":"TURN","direction":"left"}; direction must be left (90°), right (90°), or back (180°). Rotates the body in place; clearance is checked before execution. Arbitrary angles are unsupported.
MOVE_TO uses the same point format as object skills, with skill="MOVE_TO". Select either a visible fixed floor position, or a visible piece of furniture or object, in the current panorama or VIEW image. Floor: a 2D A* planner checks base clearance; if a route is found, the robot arrives directly at that floor point with its heading unchanged. Furniture/object: the robot is placed at a reachable position in front of it, on the side nearest the clicked point, facing it. Walls and ceilings are not valid targets. One request costs one step.
CHECK uses the same point format as GRASP. Select a visible movable, non-articulated object in the current panorama or VIEW image. It costs one step and returns inspection_view: a high-resolution sheet of four exterior views, including the back, without moving the object or robot. This explicitly reveals external appearance beyond the current line of sight, but no object identities, hidden-state metadata or goal labels are supplied; visible packaging text is preserved. Other objects are omitted, so do not infer contents or emptiness from this image. Inspection is not a scene camera and has no actionable frame_id; use the panorama or a new VIEW for subsequent point actions. The inspection is returned only for the CHECK response.
VIEW returns one pinhole supplement (default 1280 x 960; actual dimensions and HFOV are in detail_view), freshly rendered at the same camera origin. Optional zoom is a focal-length multiplier from 1 to 4, default 1: 1x shows 90 degrees horizontally, 2x about 53 degrees, 4x about 28 degrees. Larger zoom shows a narrower region with more detail, without moving closer. It cannot reveal occluded surfaces or text missing from the asset. Example: {"skill":"VIEW","yaw":0,"pitch":-20,"zoom":4}. Negative pitch looks down toward the robot and floor; positive pitch looks up. VIEW never rotates the body.
VIEW: {"skill":"VIEW","yaw":90,"pitch":0}; absolute head-relative angles in degrees.
Yaw range [-180,180], positive left; pitch [-90,90], positive up (default 0).
VIEW changes only the selected view of the head panorama, never body orientation.
For OPEN/CLOSE, click the visible interaction part (affordance), such as the handle or pull, not an arbitrary point on the cabinet. Native annotated handles take priority; otherwise reachability and safety are checked at your clicked surface point, without searching for another contact point.
OPEN/CLOSE/GRASP/PLACE: {"skill":"GRASP","point":{"frame_id":"CURRENT_FRAME","x":0.5,"y":0.5}}.
Use the CURRENT frame_id below. Coordinates are normalized [0,1], origin top-left.
For PLACE click the exact visible support position. Open a container first to expose its inner support surface; PLACE does not automatically select an interior.
REPORT_DONE: {"skill":"REPORT_DONE"}; reporting before completion ends in failure.
Every action, including VIEW and invalid actions, costs one planning step.
Feedback: EXECUTED, INVALID_ACTION, OUT_OF_CAPABILITY, UNSAFE. Infer your next action from RGB and history.
Image 1 is always the full equirectangular panorama. VIEW adds a scene pinhole image; CHECK adds a non-actionable inspection sheet. For a point action, choose the current panorama or VIEW detail and use its matching frame_id and coordinates.
Do not output a list, explanatory prose, or multiple candidate actions.
Your response is the action object itself. Its top-level key is "skill".
Never wrap the response in an "action" object. Never generate "feedback";
feedback comes only from the environment. Never repeat a history entry.
'''
PROMPT += '\n' + PANORAMA_CALIBRATION
RESPONSE_REQUEST = '\nChoose your next action now. Return only one JSON object with top-level "skill":\n'
PUBLIC_FIELDS = ('instruction', 'frame_id', 'planning_step', 'max_planning_steps', 'skill_library', 'feedback')


def image_payload(observation):
    images = [dict(label='Image 1: PRIMARY full panorama, authoritative scene overview; frame_id=' + observation['frame_id'],
                   image_png_b64=observation['image_png_b64'])]
    detail = observation.get('detail_view')
    if detail:
        images.append(dict(label=f"Image 2: SUPPLEMENTARY on-demand pinhole view; frame_id={detail['frame_id']}; "
                           f"yaw={detail['yaw_deg']}, pitch={detail['pitch_deg']}, "
                           f"zoom={detail.get('zoom', 1)}, HFOV={detail.get('hfov_deg', 90)} degrees",
                           image_png_b64=detail['image_png_b64']))
    inspection = observation.get('inspection_view')
    if inspection:
        images.append(dict(label="Object inspection: isolated exterior views, not a scene camera; not actionable.",
                           image_png_b64=inspection['image_png_b64']))
    return images


def detail_metadata(observation):
    detail = observation.get('detail_view')
    return {k: v for k, v in detail.items() if k != 'image_png_b64'} if detail else None


def build_prompt(observation, history, history_steps=60):
    public = {k: observation[k] for k in PUBLIC_FIELDS}
    public['detail_view'] = detail_metadata(observation)
    public['inspection_view'] = {k:v for k,v in (observation.get('inspection_view') or {}).items() if k != 'image_png_b64'} or None
    previous = [{k: item.get(k) for k in ('action', 'feedback')}
                for item in (history[-history_steps:] if history_steps else [])]
    return PROMPT + '\nCurrent observation:\n' + json.dumps(public, ensure_ascii=False) + \
        '\nPrevious actions and their feedback (oldest first):\n' + json.dumps(previous, ensure_ascii=False) + RESPONSE_REQUEST


def prompt_fingerprint():
    return hashlib.sha256((PROMPT_VERSION + PROMPT + RESPONSE_REQUEST).encode()).hexdigest()


def source_fingerprint():
    """Hash executable benchmark sources, excluding generated data and logs."""
    root = Path(__file__).resolve().parents[3]
    files = sorted((root / 'src/rummagebench').rglob('*.py'))
    files += [root / 'scripts' / name for name in
              ('evaluate_local_vlm.py', 'serve_local_vlm.py', 'run_local_vlm_suite.py',
               'serve_extended_vlm.py', 'extended_vlm_adapters.py',
               'serve_vlm.sh', 'serve_extended_vlm.sh', 'capture_qwen_attention.py',
               'vlm_messages.py', 'qwen_decoding.py')]
    files += [root / 'configs/models/local_vlm.json']
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode() + b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def request_json(url, body=None, timeout=300):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(8192).decode('utf-8', errors='replace')
        raise PolicyInfrastructureError(f'HTTP {error.code}: {detail}') from error
    except (OSError, ValueError) as error:
        raise PolicyInfrastructureError(f'{type(error).__name__}: {error}') from error
    if not isinstance(result, dict) or result.get('error'):
        raise PolicyInfrastructureError(str(result))
    return result


class LocalVLMPolicy:
    def __init__(self, endpoint, model_id, output, max_new_tokens=512, seed=0, history_steps=60, timeout=300):
        self.endpoint = endpoint.rstrip('/')
        self.model_id, self.output = model_id, Path(output)
        self.max_new_tokens, self.seed = max_new_tokens, seed
        self.history_steps, self.timeout = history_steps, timeout

    def __call__(self, observation, history):
        prompt = build_prompt(observation, history, self.history_steps)
        body = {'model':self.model_id, 'prompt':prompt, 'image_png_b64':observation['image_png_b64'],
                'max_new_tokens':self.max_new_tokens, 'seed':self.seed, 'images':image_payload(observation)}
        start = time.monotonic()
        result = request_json(self.endpoint + '/generate', body, self.timeout)
        if not isinstance(result.get('text'), str) or result.get('model') != self.model_id:
            raise PolicyInfrastructureError('inference server returned mismatched model or missing text')
        record = {'planning_step':observation['planning_step'], 'frame_id':observation['frame_id'],
                  'prompt':prompt, 'image_sha256':hashlib.sha256(observation['image_png_b64'].encode()).hexdigest(),
                  'duration_s':time.monotonic()-start,
                  'images':[dict(label=x['label'],sha256=hashlib.sha256(x['image_png_b64'].encode()).hexdigest()) for x in image_payload(observation)], **result}
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return result['text']
