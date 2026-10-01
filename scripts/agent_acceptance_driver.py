"""Interactive public-action acceptance driver; private snapshots report only error magnitudes."""
import argparse, base64, json, time, traceback
from pathlib import Path
import numpy as np
from rummagebench.adapters.python_api import create_session

def array(x):
    return x.detach().cpu().numpy().copy() if hasattr(x,"detach") else np.asarray(x).copy()

def snapshot(backend):
    result={}
    for obj in backend._env.scene.objects:
        pos, quat=obj.get_position_orientation()
        joints=obj.get_joint_positions() if obj.n_joints else []
        result["object/"+obj.name]=(array(pos),array(quat),array(joints))
        for name,link in obj.links.items():
            lp,lq=link.get_position_orientation()
            result["link/"+obj.name+"/"+name]=(array(lp),array(lq),np.array([]))
    for name,sensor in backend._robot.sensors.items():
        if hasattr(sensor,"get_position_orientation"):
            sp,sq=sensor.get_position_orientation()
            result["sensor/"+name]=(array(sp),array(sq),np.array([]))
    return result

def compare(a,b):
    errors={"position_m":0.,"orientation_deg":0.,"joint_rad":0.}
    assert a.keys()==b.keys(), "world entity set changed"
    for key in a:
        p,q,j=a[key]; p2,q2,j2=b[key]
        if any(not np.isfinite(v).all() for v in (p,q,j,p2,q2,j2)):
            raise ValueError("Nonfinite world state in restoration audit")
        if not np.isclose(np.linalg.norm(q),1.,atol=1e-3) or not np.isclose(np.linalg.norm(q2),1.,atol=1e-3):
            raise ValueError("Invalid quaternion in restoration audit")
        if j.shape != j2.shape:
            raise ValueError("Joint count changed in restoration audit")
        errors["position_m"]=max(errors["position_m"],float(np.linalg.norm(p-p2)))
        dot=abs(float(np.dot(q/np.linalg.norm(q),q2/np.linalg.norm(q2))))
        errors["orientation_deg"]=max(errors["orientation_deg"],float(np.degrees(2*np.arccos(np.clip(dot,0,1)))))
        if len(j): errors["joint_rad"]=max(errors["joint_rad"],float(np.max(np.abs(j-j2))))
    errors["ok"]=errors["position_m"]<=.001 and errors["orientation_deg"]<=.1 and errors["joint_rad"]<=1e-4
    return errors

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    parser.add_argument("--scenario",default="scenarios/knife_search_001/scenario.yaml")
    parser.add_argument("--physics-only-settle", action="store_true")
    parser.add_argument("--trace-render", action="store_true")
    parser.add_argument("--defer-camera-modalities", action="store_true")
    args=parser.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if args.defer_camera_modalities:
        from rummagebench.sim.omnigibson import backend as backend_module
        original_build=backend_module.build_env
        original_setup=backend_module.OmniGibsonBackend.setup
        def build_rgb(scenario):
            initial=scenario.model_copy(deep=True)
            initial.robot.obs_modalities=["rgb"]
            return original_build(initial)
        def setup_deferred(backend,scenario):
            report=original_setup(backend,scenario)
            requested=set(scenario.robot.obs_modalities)
            if requested.intersection({"seg_instance","seg_instance_id","bbox_2d_loose","bbox_2d_tight","bbox_3d"}):
                requested.add("seg_semantic")
            attached=[]
            for name,sensor in backend._robot.sensors.items():
                if "rgb" not in sensor.modalities:continue
                for modality in sorted(requested,key=lambda m:(m!="seg_semantic",m)):
                    sensor.add_modality(modality)
                attached.append({"sensor":name,"modalities":sorted(sensor.modalities)})
            (out/"deferred_camera_modalities.json").write_text(json.dumps(attached,indent=2))
            for _ in range(6):backend._sim.render()
            backend.validate_physics_state()
            return report
        backend_module.build_env=build_rgb
        backend_module.OmniGibsonBackend.setup=setup_deferred
    if args.physics_only_settle:
        from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
        original_settle=OmniGibsonBackend.settle
        original_capture=OmniGibsonBackend.capture_visual_frame
        def stage(name, backend):
            backend.validate_physics_state()
            p,q=backend.robot_pose()
            with (out/"private_phases.jsonl").open("a") as f:
                f.write(json.dumps({"time":time.time(),"phase":name,"position":p,"orientation":q,"finite":True})+"\n")
        def settle(backend, steps=None):
            if backend._initial_state is None:
                return original_settle(backend, steps)
            stage("settle.begin",backend)
            scope=backend._sim.render_on_step(False)
            scope.__enter__()
            try:
                for i in range(steps if steps is not None else backend._settle_steps):
                    backend._sim.step()
                    stage("settle.physics."+str(i),backend)
            finally:
                # Installed OG context manager lacks try/finally around yield;
                # exit normally to ensure restoration even if validation raises.
                scope.__exit__(None,None,None)
            stage("settle.end",backend)
        def capture(backend):
            stage("capture.begin",backend)
            if args.trace_render:
                import sys
                log=(out/"synthetic_graph_trace.jsonl").open("a",buffering=1)
                def trace(frame,event,arg):
                    if frame.f_code.co_name != "_post_process_graph_tick":return None
                    if event=="line" and frame.f_lineno==1504:
                        log.write(json.dumps({"time":time.time(),"node":str(frame.f_locals.get("nodePath"))})+"\n")
                    return trace
                prior=sys.gettrace()
                sys.settrace(trace)
                try:
                    # Pure render drain; keeps physics unchanged and exercises
                    # the delayed frame within normal capture, without dump_state.
                    for _ in range(4):backend._sim.render()
                    frame=original_capture(backend)
                finally:
                    sys.settrace(prior);log.close()
            else:
                frame=original_capture(backend)
            stage("capture.end",backend)
            return frame
        OmniGibsonBackend.settle=settle
        OmniGibsonBackend.capture_visual_frame=capture
    session=None
    report={"driver_completed":False,"task_success":False,"steps":[]}
    try:
        session=create_session(args.scenario,run_dir=out/"private",mode="agent")
        session.set_trace(out/"private"/"events.jsonl")
        session._session.set_log(str(out/"private"/"legacy_events.jsonl"))
        obs=session.reset()
        for index in range(200):
            public=obs.model_dump(mode="json")
            (out/f"observation_{index}.json").write_text(json.dumps(public))
            (out/f"rgb_{index}.png").write_bytes(base64.b64decode(public["image_png_b64"]))
            (out/"ready.json").write_text(json.dumps({"index":index,"status":session.status(),"frame_id":public["frame_id"],"planning_step":public["planning_step"]}))
            print("[agent] ready",index,session.status(),flush=True)
            command=out/f"action_{index}.json"
            deadline=time.monotonic()+1800
            while not command.exists():
                if time.monotonic()>deadline: raise TimeoutError("No next public action")
                time.sleep(.2)
            action=json.loads(command.read_text())
            if action=={"control":"exit"}:
                report["driver_completed"]=True; break
            if action=={"control":"reset"}:
                obs=session.reset(); continue
            if action=={"control":"audit_collision"}:
                # Trusted evaluator-only file control, never an AGENT skill.
                # Reload this diagnostic so API fixes do not restart Kit.
                import importlib, diagnose_sofa_pair
                try:
                    evidence=importlib.reload(diagnose_sofa_pair).audit(session._backend)
                except Exception as error:
                    evidence={"error":repr(error)}
                (out/f"collision_audit_{index}.json").write_text(json.dumps(evidence,indent=2))
                continue
            before=snapshot(session._backend) if action.get("skill")=="OBSERVE" else None
            result=session.step(action)
            obs=result.observation
            record=result.model_dump(mode="json")
            record["observation"].pop("image_png_b64",None)
            record["observation"]["observe_views"]=[{"view_index":v["view_index"]} for v in record["observation"]["observe_views"]]
            if before is not None: record["restore_errors"]=compare(before,snapshot(session._backend))
            report["steps"].append(record)
            (out/f"result_{index}.json").write_text(json.dumps(record,indent=2))
            (out/"report.json").write_text(json.dumps(report,indent=2))
    except Exception as error:
        report["error"]=repr(error);traceback.print_exc();raise
    finally:
        report["task_success"]=session is not None and session.status()=="SUCCESS"
        (out/"report.json").write_text(json.dumps(report,indent=2))
        if session is not None: session._backend.close()

if __name__=="__main__": main()
