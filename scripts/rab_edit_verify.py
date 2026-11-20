"""Validate an editor revision in a separate native Kit process."""
import argparse
import os
import traceback
from pathlib import Path

from rab_spectator_data import load,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,width=64,height=64,multi_gpu=False,limit_cpu_threads=4,
             extra_args=['--/log/level=error']))
        import carb.settings
        import omni.physx
        from rab_validate_search import validate
        settings=carb.settings.get_settings()
        settings.set_bool('/physics/suppressReadback',False)
        settings.set_bool('/physics/updateToUsd',True)
        from pxr import UsdPhysics
        from rab_restore import rotation_difference
        import math
        episode=load(a.input)
        interface=omni.physx.get_physx_interface()
        def after(stage,poses,initial):
            comparisons={}
            for key,ref in episode.get('editor_joint_pose',{}).items():
                p=episode['arrangement']['articulations'][key]
                body=UsdPhysics.Joint(stage.GetPrimAtPath(p)).GetBody1Rel().GetTargets()[0]
                got=interface.get_rigidbody_transformation(str(body))
                if not got.get('ret_val'):
                    raise ValueError('Cannot read saved articulated body '+str(body))
                drift=math.dist(ref[:3],got['position'])
                angle=math.degrees(rotation_difference(ref[3:],got['rotation']))
                comparisons[key]=dict(position_drift_m=drift,angle_drift_deg=angle,
                                      passed=drift<=.018 and angle<=4.)
            return dict(passed=all(r['passed'] for r in comparisons.values()),
                        joints=comparisons,scope='Editor revision independent joint pose check')
        report=validate(episode,omni.physx.get_physx_simulation_interface(),interface,after_settle=after)
        save(a.output,report)
        os._exit(0)
    except Exception as error:
        save(a.output,dict(status='native_error',error=str(error),traceback=traceback.format_exc()))
        os._exit(1)


if __name__=='__main__':
    main()
