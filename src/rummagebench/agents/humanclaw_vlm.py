"""HumanCLAW-inspired prompt and memory only; no action intervention.

Reference: Human-CLAW/HumanCLAW@6323d938795afd27d66ab926216bf4ba7b003ee6.
Not a reproduction of PSV or its reported scores.
"""
from __future__ import annotations
import hashlib
import json
import re
import time
from pathlib import Path
from .local_vlm import PUBLIC_FIELDS, request_json, image_payload, detail_metadata, PANORAMA_CALIBRATION
from .policy_adapter import PolicyInfrastructureError

VERSION = 'humanclaw-panorama-reason-feedback-v4.7-explicit-footprint'
PLANNER = '''You control a robot using its current RGB image and public feedback.
Write all descriptions, plans, reasoning and other natural-language output in English.
Keep the prescribed JSON keys, skill names and actual frame identifiers unchanged.
Complete the task instruction. Describe visible evidence briefly, state a short-term
 goal for the next six actions, and select exactly one immediate action yourself.
Use your previous descriptions as memory, not as verified facts.
Full original outputs are archived. Older conversation may be replaced by your own
compact memory; treat that memory as fallible historical claims. Each new
user observation reports the previous submitted action and its public feedback.
Only the latest user turn carries images; earlier image pixels are not replayed.
Historical descriptions and
plans are your own claims, not current observations or confirmed execution.
Only execution feedback establishes whether a submitted action succeeded.

BODY AND CAMERA
You inhabit the robot described in observation.embodiment. Its local base axes are
+x forward, +y left, +z up. Camera optical axes are +x image-right, +y down,
+z viewing forward. camera_origin_in_base_m is the camera location relative to
your base; body_forward_in_camera and body_up_in_camera express your body axes
in the current optical frame. These are self calibration, not object positions.
The footprint is the square used by the movement executor, not the exact body mesh.
observation.base_size_description states its length, width and occupied area.
This footprint is centered on your base origin. MOVE_TO checks clearance for
the entire square along its planned route at the current body heading.
The executor also expands obstacle horizontal bounds by 3 cm on each side.
This clearance margin is separate from the stated base dimensions.
executor_holding reports whether the skill executor holds something, not its identity.
The end_effector_link identifies the arm used for manipulation. No reach guarantee
is provided: a visible point may be outside your arm workspace or collision-blocked.
In an equirectangular panorama, center is reference-forward, left quarter is left,
right quarter is right, and the horizontal seam is behind the camera. The image
is not a perspective view: its entire bottom edge looks down. Do not treat its
bottom center as the robot's feet or estimate a straight floor lane from it.
In perspective mode, the image center follows VIEW, not necessarily body-forward.
MOVE_TO on the floor preserves body heading; MOVE_TO on furniture/objects faces them.
VIEW changes gaze only. Never assume your feet are visible.

MID-LEVEL PLANNING (six-action horizon)
- Describe the target's visibility and near-body clearance using the current image.
- Choose a specific visible landmark or operation as your next goal. Keep it across
  steps when appropriate; revise it when the image or actual outcome contradicts it.
- When the target is absent, scan in a consistent direction. After a scan, choose
  a visible place to explore and move toward it rather than scan forever in place.
- When the target is visible, plan an approach through clear floor. Prefer open
  passages to narrow gaps among furniture. A closed door is not a passage.
- For manipulation, distinguish seeing, approaching, grasping, and placing.
  A closed receptacle may require opening before its contents can be accessed.

IMMEDIATE ACTION GUIDANCE
- Choose exactly one action that advances your current goal.
- Use VIEW to inspect a direction and MOVE_TO to change position.
  Identical absolute VIEW angles show the same direction; they do not keep turning.
- After a failed action, reconsider its parameters, target or approach using the
  image and public feedback. Do not assume a rejected action changed your state.
- Select the visible object itself for GRASP, the handle or pull for OPEN/CLOSE, and
  the visible support surface for PLACE. Coordinates are not object names.
- Only REPORT_DONE when the complete instruction is satisfied. Looking at a target
  is not holding it; holding it is not placing it. Do not report done to pause.
These are general instructions, not a runtime checker; you choose every action.

Available skills and their exact action objects:
TURN: {"skill":"TURN","direction":"left"}; direction must be left (90°), right (90°), or back (180°). Rotates the body in place; clearance is checked before execution. Arbitrary angles are unsupported.
MOVE_TO uses the same point format as object skills, with skill="MOVE_TO". Select either a visible fixed floor position, or a visible piece of furniture or object, in the current panorama or VIEW image. Floor: a 2D A* planner checks base clearance; if a route is found, the robot arrives directly at that floor point with its heading unchanged. Furniture/object: the robot is placed at a reachable position in front of it, on the side nearest the clicked point, facing it. Walls and ceilings are not valid targets. One request costs one step.
CHECK uses the same point format as GRASP. Select a visible movable, non-articulated object in the current panorama or VIEW image. It costs one step and returns inspection_view: a high-resolution sheet of four exterior views, including the back, without moving the object or robot. This explicitly reveals external appearance beyond the current line of sight, but no object identities, hidden-state metadata or goal labels are supplied; visible packaging text is preserved. Other objects are omitted, so do not infer contents or emptiness from this image. Inspection is not a scene camera and has no actionable frame_id; use the panorama or a new VIEW for subsequent point actions. The inspection is returned only for the CHECK response.
VIEW returns one pinhole supplement (default 1280 x 960; actual dimensions and HFOV are in detail_view), freshly rendered at the same camera origin. Optional zoom is a focal-length multiplier from 1 to 4, default 1: 1x shows 90 degrees horizontally, 2x about 53 degrees, 4x about 28 degrees. Larger zoom shows a narrower region with more detail, without moving closer. It cannot reveal occluded surfaces or text missing from the asset. Example: {"skill":"VIEW","yaw":0,"pitch":-20,"zoom":4}. Negative pitch looks down toward the robot and floor; positive pitch looks up. VIEW never rotates the body.
VIEW: {"skill":"VIEW","yaw":90,"pitch":0}; absolute head-relative angles,
yaw [-180,180] positive left, pitch [-90,90] positive up (optional, default 0).
VIEW selects a viewport without moving the robot or changing body heading.
Think of VIEW as looking over your shoulder while your torso stays facing forward.
This is a virtual gaze change, not a physical neck or base rotation.
OPEN/CLOSE/GRASP/PLACE use an object with "skill" and "point".
For OPEN/CLOSE, click the visible interaction part (affordance), such as the handle or pull, not an arbitrary point on the cabinet. Native annotated handles take priority; otherwise reachability and safety are checked at your clicked surface point, without searching for another contact point.
"point" contains "frame_id", "x", and "y". Copy point.frame_id verbatim from
the latest observation.frame_id for Image 1, or latest observation.detail_view.frame_id
for Image 2 when present. It is an actual image identifier, not a keyword.
Choose x and y yourself from that image: normalized coordinates [0,1], top-left
origin. The image center is not automatically the target. PLACE uses the exact
clicked support position; it does not choose another point or an interior for you.
To place inside a drawer or cabinet, first open it and click its exposed inner
support surface. Clicking a closed front or a vertical face does not select the interior.
REPORT_DONE: {"skill":"REPORT_DONE"}; ends the episode; scored against the task goal.

PARAMETER MEANING AND EXAMPLES (not a task solution)
- VIEW is an absolute direction in the current head-mounted panorama reference:
  yaw=0,pitch=0 selects reference-forward; yaw=90 selects left; yaw=-90 selects right.
  Two VIEW(yaw=90) actions select the same direction if the body has not moved.
  VIEW(pitch=-45) looks downward; it does not lower or translate the camera.
- MOVE_TO on the floor preserves body heading; on furniture/objects it faces them. VIEW only changes the supplementary view.
- A point is a location IN ONE CURRENT IMAGE: x grows right, y grows down.
  x=0.25,y=0.5 means one quarter across and halfway down, not 25 cm or 25 degrees.
  Point on the visible object you intend to manipulate. Use that image's actual
  frame_id from the latest observation; do not invent an identifier or reuse
  an identifier from an earlier turn.
- A rejected MOVE_TO leaves the robot at its previous position.
Image 1 is ALWAYS the full equirectangular panorama. VIEW adds Image 2 as a
supplementary pinhole view, never replaces Image 1. This supplement appears only
in the observation returned by a successful VIEW. Any subsequent non-VIEW action
removes it, even if that action fails. Use the full panorama as your primary source
of scene information, not the pinhole view alone. While the supplement is present,
you may point in either image using its own frame_id and normalized coordinates.
The body axes in observation.embodiment refer to Image 1's reference optical frame.
The view angles describe Image 2 only. VIEW does not rotate or steer the body.
Feedback is EXECUTED, INVALID_ACTION, OUT_OF_CAPABILITY, or UNSAFE.
Feedback messages briefly explain the result of your submitted action in plain
language. A failed base motion may name an obstacle category identified by the
executor's clearance test. This is a conservative geometry test, not necessarily
physical contact. An arm reach failure means no valid interaction pose was found
in the tested candidate set, not proof that every possible pose is unreachable.
These messages report outcomes only; they do not recommend or choose your next action.
Every submitted action, including an invalid action, costs one planning step.
No verifier or controller will correct your choice or complete missing parameters
beyond defaults explicitly defined in the public skill protocol.
Use only the image, public observation and your history; do not invent hidden state.

Return exactly one JSON object with these fields:
visible_state: short description of current visual evidence;
mid_level_progress_analysis: brief progress assessment;
mid_level_goal: your short-term goal;
low_level_action_reasoning: brief reason for your choice;
action: exactly one action object using the skill schema above.
Do not output multiple alternatives or an array. All decisions are yours.
'''


PLANNER += '\n' + PANORAMA_CALIBRATION

# A common output-format instruction, not a generated action or a runtime verifier.
OUTPUT_CONTRACT = (
    'Return one JSON object only. Put visual evidence in visible_state, progress assessment '
    'in mid_level_progress_analysis, goal in mid_level_goal, and action rationale in '
    'low_level_action_reasoning (all strings). Put exactly one chosen skill object under '
    'action. Use the exact skill names and parameter keys from the protocol. Do not copy '
    'the input observation or these format instructions into your response. No Markdown '
    'or prose outside the JSON.'
)


def fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes() + PANORAMA_CALIBRATION.encode()).hexdigest()


def parse_plan(text):
    """Unwrap completed reasoning / one code fence, never repair action content."""
    if not isinstance(text, str):
        return {}, {}
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    if '<think>' in text:
        return {}, {}
    fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, flags=re.S)
    if fence:
        text = fence.group(1)
    try:
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('ambiguous duplicate JSON key')
                result[key] = value
            return result
        plan = json.loads(text, object_pairs_hook=unique_keys)
    except (ValueError, TypeError):
        return {}, {}
    if not isinstance(plan, dict):
        return {}, {}
    action = plan.get('action')
    return plan, action if isinstance(action, dict) else {}


class HumanCLAWPolicy:
    def __init__(self, endpoint, model_id, output, max_new_tokens=4096,
                 seed=0, timeout=300, request=request_json, compact_threshold_tokens=None):
        self.endpoint = endpoint.rstrip('/')
        self.model_id, self.output = model_id, Path(output)
        self.max_new_tokens, self.seed, self.timeout = max_new_tokens, seed, timeout
        self.compact_threshold_tokens = compact_threshold_tokens
        self.request = request
        self.reset()

    def reset(self):
        self.memory = []
        self.messages = []
        self.compaction_count = 0
        self.infrastructure_degraded = False

    def log(self, name, row):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.with_name(name).open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def prepare_messages(self, messages, obs):
        """Same-model memory maintenance; never executes or repairs an action."""
        if self.compact_threshold_tokens is None:
            return messages
        base = dict(model=self.model_id, max_new_tokens=self.max_new_tokens, seed=self.seed,
                    image_png_b64=obs['image_png_b64'], images=image_payload(obs))
        def count(items):
            started = time.monotonic()
            audit = dict(stage='token_count', planning_step=obs['planning_step'],
                         message_count=len(items), model=self.model_id)
            try:
                result = self.request(self.endpoint + '/count_tokens',
                                      dict(base, messages=items), self.timeout)
                audit.update(result)
                if result.get('model') != self.model_id:
                    raise PolicyInfrastructureError('Token-count model mismatch')
                return int(result['input_tokens'])
            except Exception as error:
                audit['error'] = str(error)
                self.infrastructure_degraded = True
                raise PolicyInfrastructureError(f'Context token counting failed: {error}') from error
            finally:
                audit['duration_s'] = time.monotonic() - started
                self.log('context_budget.jsonl', audit)
        before = count(messages)
        if before <= self.compact_threshold_tokens or not self.messages:
            return messages
        # Let the model see all active history and decide what its next memory
        # contains. No fixed recent-turn window or harness-authored memory.
        transcript = [dict(m) for m in self.messages]
        if transcript[0]['content'].startswith(PLANNER):
            transcript[0]['content'] = transcript[0]['content'][len(PLANNER):]
        instruction = (
            'This is a memory-maintenance turn, not a robot action turn. '
            'The following transcript contains only your previous public conversation.\n'
            'BEGIN PAST TRANSCRIPT\n' + json.dumps(transcript, ensure_ascii=False) +
            '\nEND PAST TRANSCRIPT\n'
            'Prepare the memory you want to carry into subsequent turns of this task. '
            'You decide what to retain, its organization, and its level of detail. '
            'Write in English. The historical action-output format does not apply to '
            'this turn. Your returned text will be used verbatim as your memory, not '
            'executed as a robot action. The skill protocol and next public observation '
            'will be provided separately. Use only the supplied public information.')
        compact_messages = [dict(role='user', content=instruction)]
        row = dict(stage='compaction', planning_step=obs['planning_step'],
                   messages=compact_messages, model=self.model_id,
                   threshold_tokens=self.compact_threshold_tokens,
                   input_tokens_before=before, retained_recent_pairs=None,
                   memory_generation_budget='native_context_remaining',
                   source_message_count=len(self.messages), images=[dict(label=x['label'],
                   sha256=hashlib.sha256(x['image_png_b64'].encode()).hexdigest())
                   for x in image_payload(obs)])
        started = time.monotonic()
        try:
            result = self.request(self.endpoint + '/generate',
                dict(base, messages=compact_messages, stage='compaction'), self.timeout)
            row.update(result)
            if (result.get('model') != self.model_id or
                not isinstance(result.get('text'), str)):
                raise PolicyInfrastructureError('Memory response model mismatch or missing text')
            compacted = [dict(role='user', content=PLANNER +
                '\nThe following assistant message is your own compacted historical memory, '
                'not a current observation or verified state.'),
                dict(role='assistant', content=result['text'])] + [messages[-1]]
            after = count(compacted)
            row['input_tokens_after'] = after
            row['reduced_input'] = after < before
            row['memory_accepted_without_content_check'] = True
            self.compaction_count += 1
            return compacted
        except Exception as error:
            row['error'] = str(error)
            self.infrastructure_degraded = True
            raise PolicyInfrastructureError(f'Context compaction failed: {error}') from error
        finally:
            row['duration_s'] = time.monotonic() - started
            self.log('context_compactions.jsonl', row)

    def __call__(self, obs, history):
        if obs['planning_step'] == 0:
            self.reset()
        # Bind each prior model description to its actual submitted action and
        # execution outcome, rather than keeping two detached histories.
        memory = []
        for index, item in enumerate(self.memory):
            row = dict(item)
            history_index = len(history) - len(self.memory) + index
            if 0 <= history_index < len(history):
                row['action'] = history[history_index].get('action')
                row['feedback'] = history[history_index].get('feedback')
            memory.append(row)
        context = dict(observation={k: obs[k] for k in (*PUBLIC_FIELDS, 'embodiment') if k in obs},
                       history=memory)
        context['observation']['detail_view'] = detail_metadata(obs)
        context['observation']['inspection_view'] = {k:v for k,v in (obs.get('inspection_view') or {}).items() if k != 'image_png_b64'} or None
        embodiment = obs.get('embodiment') or {}
        half = embodiment.get('executor_footprint_half_extent_m')
        if half is not None:
            side = 2 * half
            context['observation']['base_size_description'] = (
                f"Your executor base footprint is a square: length {side * 100:g} cm "
                f"(front-to-back), width {side * 100:g} cm (left-to-right), "
                f"occupied area {side * side:.4g} square meters. "
                f"It extends {half * 100:g} cm forward, backward, left and right "
                "from the base origin. These are collision-check dimensions, "
                "not measured dimensions of the robot shell."
            )
        turn = {'observation': context['observation'], 'response_format': OUTPUT_CONTRACT}
        if history:
            turn['previous_action'] = history[-1].get('action')
            turn['previous_feedback'] = history[-1].get('feedback')
        if self.memory:
            prior_plan, prior_action = parse_plan(self.memory[-1]['raw_output'])
            if not prior_plan:
                turn['previous_response_format_feedback'] = (
                    'Your previous response was not a parseable planner JSON object; no action was extracted.'
                )
            elif not isinstance(prior_plan.get('action'), dict):
                turn['previous_response_format_feedback'] = (
                    'Your previous response was parsed, but contained no action object; no action was extracted.'
                )
            if self.memory[-1].get('output_budget_reached'):
                turn['previous_output_limit_feedback'] = (
                    'Your previous generation reached the requested output token limit.'
                )
        user_text = ('Public observation:\n' + json.dumps(turn, ensure_ascii=False))
        if not self.messages:
            user_text = PLANNER + '\n' + user_text
        messages = self.messages + [{'role': 'user', 'content': user_text}]
        messages = self.prepare_messages(messages, obs)
        # Compatibility audit view for existing report readers, not model input.
        prompt = PLANNER + '\nInput:\n' + json.dumps(context, ensure_ascii=False)
        row = dict(stage='planner', attempt=1, planning_step=obs['planning_step'],
                   frame_id=obs['frame_id'], prompt=prompt, messages=messages,
                   prompt_is_audit_only=True, message_protocol='multiturn-public-v1',
                   image_sha256=hashlib.sha256(obs['image_png_b64'].encode()).hexdigest(),
                   images=[dict(label=x['label'], sha256=hashlib.sha256(x['image_png_b64'].encode()).hexdigest()) for x in image_payload(obs)])
        start = time.monotonic()
        try:
            result = self.request(self.endpoint + '/generate', dict(model=self.model_id,
                messages=messages, prompt=prompt, image_png_b64=obs['image_png_b64'],
                max_new_tokens=self.max_new_tokens, seed=self.seed, images=image_payload(obs)), self.timeout)
            row.update(result)
            if result.get('context_budget_exhausted'):
                raise PolicyInfrastructureError('ContextWindowExceeded: generation reached the native context limit; raw output retained, no action submitted.')
            if result.get('model') != self.model_id or not isinstance(result.get('text'), str):
                raise PolicyInfrastructureError('mismatched model or missing text')
        except PolicyInfrastructureError as error:
            self.infrastructure_degraded = True
            row.update(error=str(error), error_kind='transport')
            raise
        finally:
            row['duration_s'] = time.monotonic() - start
            self.log(self.output.name, row)
        plan, action = parse_plan(result['text'])
        self.messages = messages + [{'role': 'assistant', 'content': result['text']}]
        # Preserve exactly what the model returned, including malformed outputs
        # and any explicitly returned reasoning. Never pull from private traces.
        memory = {'step': obs['planning_step'], 'action': action,
                  'raw_output': result['text'],
                  'output_budget_reached':result.get('output_budget_reached',False)}
        for key in ('visible_state', 'mid_level_progress_analysis',
                    'mid_level_goal', 'low_level_action_reasoning'):
            memory[key] = plan.get(key, '')
        self.memory.append(memory)
        self.log('agent_decisions.jsonl', dict(planning_step=obs['planning_step'],
            version=VERSION, planner=plan, final_action=action, intervention=False,
            parse_valid=bool(plan) and isinstance(plan.get('action'), dict),
            output_budget_reached=result.get('output_budget_reached', False)))
        return action
