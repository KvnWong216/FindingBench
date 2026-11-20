"""Publish live trajectories, then render an exo demo after the suite releases GPU."""
import json
import os
from pathlib import Path
import subprocess
import time

from build_harness_review import ROOT, DEFAULT_OUT, DEFAULT_SUITE, publish


def main():
    state_path = DEFAULT_OUT / "publisher_state.json"
    def save(**row):
        state_path.write_text(json.dumps(dict(pid=os.getpid(), **row), indent=2))
    save(status="publishing", waiting_for_suite=str(DEFAULT_SUITE))
    while publish(DEFAULT_SUITE, DEFAULT_OUT):
        time.sleep(30)
    suite_state = json.loads((DEFAULT_SUITE / "suite.json").read_text())
    if suite_state.get("status") == "interrupted":
        save(status="stopped", note="Suite interrupted; no automatic simulator restart.")
        return
    # The serial suite writes final state only after it has stopped owned workers.
    output = DEFAULT_OUT / "exo_replay"
    if not output.exists():
        save(status="rendering_exo")
        cmd = ["/home/ubuntu/miniforge3/bin/conda", "run", "--no-capture-output", "-n", "ygbench",
               "python", str(ROOT / "scripts/analysis/render_exo_replay.py"), "--out", str(output)]
        try:
            with (DEFAULT_OUT / "exo_replay.log").open("w") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
                try:
                    code = proc.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    raise
            save(status="complete" if code == 0 else "exo_error", exit_code=code)
        except Exception as error:
            save(status="exo_error", error=str(error))
    publish(DEFAULT_SUITE, DEFAULT_OUT)


if __name__ == "__main__":
    main()
