"""Scored VLM episodes with following exo recording and optional true attention."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_SCENARIO = ROOT / "scenarios/behavior_groceries_pilot/scenario.yaml"


def worker(a):
    from rummagebench.agents.humanclaw_vlm import HumanCLAWPolicy, VERSION, fingerprint
    from rummagebench.agents.local_vlm import request_json, source_fingerprint
    from rummagebench.agents.policy_adapter import PolicyAgent
    from rummagebench.evaluation.episode_log import _save_rgb_png
    from rummagebench.environment.env import InteractiveSearchEnv
    from follow_exo import FollowExo
    started = time.time()
    report = dict(status="initializing", run_valid=False, seed=0, task=a.task_id, scenario=str(a.scenario), max_steps=a.max_steps,
                  max_new_tokens=a.max_new_tokens, prompt_version=VERSION, prompt_sha256=fingerprint(),
                  source_sha256=source_fingerprint(), feedback_protocol="reason-feedback-v1",
                  ego_input="full panorama + successful VIEW supplement only",
                  history_policy=("multiturn_full_text_public_feedback_current_images" if a.no_context_compaction
                                  else "same_model_compaction_public_history_current_images"),
                  history_steps=None, validation_stop_after=a.validation_steps or None,
                  exo_supplied_to_model=False, attention_supplied_to_model=False,
                  attention_changes_model_output=False, model_calls=0)
    import hashlib
    report["instrumentation_sha256"]={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ["scripts/analysis/run_instrumented_demo.py","scripts/analysis/follow_exo.py","scripts/capture_qwen_attention.py",
         "scripts/analysis/compose_instrumented_video.py","scripts/analysis/publish_instrumented_demo.py"]}
    def save():
        report["elapsed_s"] = round(time.time()-started, 2)
        tmp=a.out/"report.json.tmp";tmp.write_text(json.dumps(report,indent=2));tmp.replace(a.out/"report.json")
    save()
    try:
        health = request_json(f"http://127.0.0.1:{a.port}/health")
        report["attention_available"] = bool(health.get("attention_recording"))
        report["server"] = health
        threshold = min(12288, int((health.get("context_limit_tokens") or 16384) * 0.70))
        report["context_compaction"] = dict(enabled=not a.no_context_compaction, threshold_tokens=threshold,
            summary_max_new_tokens=None, summary_generation_budget="native_context_remaining",
            recent_pairs=None, memory_content_checks=False, original_history_archived=True,
            method="same_model_text_summary", counts="exact_multimodal_tokenizer",
            trigger="min(12288, 70% native context); operational cap for shared GPU")
        env = InteractiveSearchEnv(a.scenario,
                                   seed=0, feedback_mode="reasons")
        session = env._session
        session.set_planning_budget(a.max_steps)
        session.expose_embodiment = True
        episode=a.out/"episode";episode.mkdir()
        session.set_trace(episode/"events.jsonl")
        session.model_version=health["model_id"]+"@"+health["revision"]
        obs=session.reset()
        exo=FollowExo(env._backend,a.out/"exo")
        exo.capture(0)
        _save_rgb_png(episode,0,obs)
        policy=HumanCLAWPolicy(f"http://127.0.0.1:{a.port}",health["model_id"],
                              a.out/"model_calls.jsonl",max_new_tokens=a.max_new_tokens,seed=0,
                              compact_threshold_tokens=None if a.no_context_compaction else threshold)
        agent=PolicyAgent(policy,name=a.model+"-instrumented")
        agent.reset(obs.instruction)
        report.update(status="running",instruction=obs.instruction,planning_steps_used=0)
        save()
        while session.status()=="RUNNING":
            raw=agent.act(obs)
            report["model_calls"]+=1
            report["compaction_calls"] = policy.compaction_count
            if raw is None:break
            result=session.step(raw);obs=result.observation
            _save_rgb_png(episode,result.planning_step,obs)
            exo.capture(result.planning_step)
            report.update(planning_steps_used=result.planning_step,
                          last_action=raw,last_feedback=result.feedback.model_dump(mode="json"))
            save()
            if a.validation_steps and result.planning_step >= a.validation_steps:
                report.update(status="validation_stopped",run_valid=False,validation_only=True)
                save()
                return 0
        report.update(status=session.status(),run_valid=True)
        save()
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        report.update(status="infrastructure_error",error=f"{type(e).__name__}: {e}")
        save()
        return 1


def stop(p):
    if p is None or p.poll() is not None:return
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()


def suite(a):
    from rummagebench.agents.local_vlm import source_fingerprint
    profiles=json.loads((ROOT/"configs/models/local_vlm.json").read_text())["models"]
    if len(set(a.models)) != len(a.models) or any(m not in profiles for m in a.models):
        raise ValueError("Models must be unique configured names")
    a.out.mkdir(parents=True,exist_ok=False)
    state=dict(status="running",pid=os.getpid(),models=a.models,scenario=str(a.scenario),task=a.task_id,max_steps=a.max_steps,validation_stop_after=a.validation_steps or None,source_sha256=source_fingerprint(),runs=[])
    def save():
        tmp=a.out/"suite.json.tmp";tmp.write_text(json.dumps(state,indent=2));tmp.replace(a.out/"suite.json")
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    child=publisher=None
    save()
    try:
        for model in a.models:
            if source_fingerprint()!=state["source_sha256"]:raise RuntimeError("Source changed during suite")
            out=a.out/model
            row=dict(model=model,status="running",out=str(out));state["runs"].append(row);save()
            cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"--model",model,"--out",str(out),"--port",str(a.port),"--max-steps",str(a.max_steps),"--scenario",str(a.scenario)]
            if a.no_context_compaction:cmd += ["--no-context-compaction"]
            if a.validation_steps:cmd += ["--validation-steps",str(a.validation_steps)]
            if a.max_new_tokens:cmd += ["--max-new-tokens",str(a.max_new_tokens)]
            with (a.out/(model+".log")).open("w") as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with (a.out/(model+"-publisher.log")).open("w") as log:
                publisher=subprocess.Popen([sys.executable,"-u",str(ROOT/"scripts/analysis/publish_instrumented_demo.py"),str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child.wait()
            state_path=out/"run_state.json"
            if state_path.exists():
                row.update(json.loads(state_path.read_text()))
            else:
                row.update(status="error",error="Episode launcher exited before creating state",exit_code=child.returncode)
                stop(publisher)
            save()
            try:publisher.wait(timeout=1800)
            except subprocess.TimeoutExpired:stop(publisher);row["export_error"]="Export timed out";save()
            row["publisher_exit_code"]=publisher.returncode;save()
        state["status"]="complete" if all(r.get("status")=="complete" and r.get("publisher_exit_code")==0 for r in state["runs"]) else "completed_with_errors"
    except KeyboardInterrupt:state["status"]="interrupted"
    except Exception as e:state.update(status="error",error=str(e))
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:stop(child)
        stop(publisher);save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--scenario",type=Path,default=DEFAULT_SCENARIO,help="Scenario binding; defaults to the official BEHAVIOR groceries task")
    p.add_argument("--port",type=int,default=8091)
    p.add_argument("--max-steps",type=int,default=100)
    p.add_argument("--validation-steps",type=int,default=0,help="Debug only: stop early without changing the planning budget shown to the model")
    p.add_argument("--model",default="qwen3-vl-4b")
    p.add_argument("--models",nargs="+")
    p.add_argument("--max-new-tokens",type=int,default=None)
    p.add_argument("--no-context-compaction",action="store_true",help="Replay complete text history as the baseline condition")
    p.add_argument("--worker",action="store_true")
    a=p.parse_args();a.out=a.out.resolve()
    a.scenario=a.scenario.resolve()
    from rummagebench.core.scenario import load_scenario
    a.task_id=load_scenario(a.scenario).id
    if not 0 <= a.validation_steps <= a.max_steps:p.error("validation-steps must be between 0 and max-steps")
    if a.models:
        return suite(a)
    profiles=json.loads((ROOT/"configs/models/local_vlm.json").read_text())["models"]
    profile=profiles[a.model]
    a.max_new_tokens=a.max_new_tokens or (32768 if a.model=="qwen3-vl-4b" else 4096)
    if a.worker:
        code=worker(a);sys.stdout.flush();sys.stderr.flush();os._exit(code)
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind(("127.0.0.1",a.port))
    a.out.mkdir(parents=True,exist_ok=False)
    state=dict(status="loading",pid=os.getpid(),out=str(a.out),model=a.model,scenario=str(a.scenario),task=a.task_id,max_steps=a.max_steps,validation_stop_after=a.validation_steps or None)
    def save():(a.out/"run_state.json").write_text(json.dumps(state,indent=2))
    save()
    server=child=None
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        with (a.out/"server.log").open("w") as log:
            server_env=dict(os.environ)
            if profile.get("environment"):server_env["YGBENCH_VLM_ENV"]=profile["environment"]
            command=[str(ROOT/("scripts/serve_extended_vlm.sh" if profile.get("backend") else "scripts/serve_vlm.sh")),"--model",a.model,"--port",str(a.port)]
            if a.model=="qwen3-vl-4b":command += ["--attention-out",str(a.out/"attention")]
            server=subprocess.Popen(command,cwd=ROOT,env=server_env,stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        from rummagebench.agents.local_vlm import request_json
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError("Server exited during loading")
            try:request_json(f"http://127.0.0.1:{a.port}/health",timeout=2);break
            except Exception:time.sleep(2)
        else:raise TimeoutError("Server load timeout")
        state["status"]="evaluating";save()
        with (a.out/"evaluation.log").open("w") as log:
            child=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"--worker",
                                    "--out",str(a.out),"--port",str(a.port),"--max-steps",str(a.max_steps),
                                    "--model",a.model,"--scenario",str(a.scenario),"--max-new-tokens",str(a.max_new_tokens),
                                    "--validation-steps",str(a.validation_steps)] +
                                   (["--no-context-compaction"] if a.no_context_compaction else []),
                                   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,start_new_session=True)
            code=child.wait(timeout=7200)
        state.update(status="complete" if code==0 else "error",exit_code=code)
    except KeyboardInterrupt:state["status"]="interrupted"
    except Exception as e:state.update(status="error",error=str(e))
    finally:stop(child);stop(server);save()


if __name__=="__main__":main()
