"""Candidate prompt for diagnostic A/B tests only; never changes frozen runs."""
import json
from rummagebench.agents.local_vlm import PUBLIC_FIELDS

def build_debug_prompt(observation,history):
    frame=observation['frame_id']
    examples=[{'skill':'MOVE','distance_cm':25},{'skill':'TURN','angle_deg':30},
              {'skill':'VIEW','yaw':0,'pitch':0}]
    examples += [{'skill':s,'point':{'frame_id':frame,'x':0.5,'y':0.5}}
                 for s in ('OPEN','CLOSE','GRASP','PLACE')]
    examples += [{'skill':'REPORT_DONE'}]
    return '''Control the robot to complete the instruction using its current image.
Return exactly one JSON action object. The examples below specify syntax, not a plan.
Choose parameters from the current observation; do not copy example coordinates blindly.
Only the listed keys are allowed. Do not add direction, action, feedback, or commentary.
MOVE: distance_cm is signed, + forward / - backward, magnitude 5..100 cm.
MOVE is along the ROBOT BODY heading, not the direction the camera is looking.
TURN: angle_deg is signed, + left / - right, magnitude 5..180 degrees.
VIEW: yaw is REQUIRED, absolute relative to the head/body reference, [-180,180] degrees;
positive yaw looks left, negative right. pitch is optional (default 0), [-90,90], positive up.
VIEW never moves or turns the robot body. Repeating the same angles without other changes
selects the same view. Initial RGB is an equirectangular panorama; VIEW returns a perspective image.
OPEN/CLOSE/GRASP/PLACE: select a visible target using point.x and point.y normalized to [0,1],
origin at the top left. Copy the exact current frame_id shown below, not a placeholder.
PLACE selects the receiving surface/container. REPORT_DONE ends the episode and fails if the goal is unmet.
All actions, including invalid ones, cost a step. Four feedback classes are EXECUTED,
INVALID_ACTION, OUT_OF_CAPABILITY, UNSAFE. Infer the task state from RGB and action history.
No object IDs or hidden world state are available.
Action syntax examples for THIS frame:\n'''+json.dumps(examples)+ '\nCurrent observation:\n'+json.dumps({k:observation[k] for k in PUBLIC_FIELDS})+'\nPrevious actions and feedback:\n'+json.dumps(history)+'\nReturn one action JSON object now:\n'
