"""Restart owned learners from completed checkpoint pairs after infrastructure failures."""

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import psutil

from smg_rl.config import load_config
from smg_rl.profiles import require_training_enabled
from smg_rl.reporting import read_json


def checkpoint_steps(path):
    path = Path(path)
    if not path.with_suffix(".intrinsic.pt").is_file():
        raise ValueError(f"Missing curiosity checkpoint for {path}")
    with zipfile.ZipFile(path.with_suffix(".intrinsic.pt")) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Corrupt curiosity checkpoint for {path}")
    with zipfile.ZipFile(path) as archive:
        archive.getinfo("data")
        data = json.loads(archive.read("data"))
    return int(data["num_timesteps"])


def newest_pair(run, fallback=None):
    candidates = [Path(fallback)] if fallback else []
    if run:
        candidates.extend(Path(run).glob("model-*.zip"))
    verified = []
    for path in candidates:
        try:
            verified.append((checkpoint_steps(path), path))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
    return max(verified, key=lambda pair: pair[0]) if verified else (0, None)


def terminate_owned(process):
    """The Popen handle identifies this supervisor's child, never arbitrary Dolphin PIDs."""
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in [*children, parent]:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs([*children, parent], timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


def supervised_train(config_path, steps=None, workers=None, resume=None):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    require_training_enabled(config)
    root = Path(config["paths"]["runs"])
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f"supervised-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    folder.mkdir()
    start = checkpoint_steps(resume) if resume else 0
    target = start + (steps or config["training"]["steps"])
    controller = {
        "pid": os.getpid(),
        "created_time": psutil.Process().create_time(),
        "config": str(config_path),
        "target_timesteps": target,
    }
    (folder / "controller.json").write_text(json.dumps(controller, indent=2), encoding="utf-8")
    print(f"Supervised training output: {folder}", flush=True)
    progress, pair, attempt, child, run = (
        start,
        Path(resume).resolve() if resume else None,
        0,
        None,
        None,
    )

    def publish(state, **extra):
        data = {
            **controller,
            "state": state,
            "attempts": attempt,
            "confirmed_checkpoint_timesteps": progress,
            "checkpoint": str(pair) if pair else None,
            "current_run": str(run) if run else None,
            "updated_at": time.time(),
            **extra,
        }
        pending = folder / "status.tmp"
        pending.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            pending.replace(folder / "status.json")
        except PermissionError:
            pass  # A Windows report/viewer may briefly hold the previous snapshot.

    try:
        while progress < target and not (folder / "STOP").exists():
            attempt += 1
            run = None
            command = [
                sys.executable,
                "-m",
                "smg_rl.cli",
                "--config",
                str(config_path),
                "train-parallel",
                "--workers",
                str(workers or config["parallel"]["workers"]),
                "--steps",
                str(target - progress),
            ]
            if pair:
                command += ["--resume", str(pair)]
            env = {
                **os.environ,
                "SMG_RL_SUPERVISOR": str(folder.resolve()),
                "SMG_RL_SUPERVISOR_ATTEMPT": str(attempt),
            }
            with (
                (folder / f"attempt-{attempt:04d}.stdout.log").open("w") as stdout,
                (folder / f"attempt-{attempt:04d}.stderr.log").open("w") as stderr,
            ):
                child = subprocess.Popen(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    env=env,
                    cwd=config["root"],
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                started, last_activity = time.time(), time.time()
                publish("starting", child_pid=child.pid)
                while child.poll() is None:
                    if run is None:
                        for record in root.glob("parallel-*/controller.json"):
                            if read_json(record, {}).get("supervisor") == str(folder.resolve()):
                                # The newest owned attempt is the currently running child.
                                data = read_json(record, {})
                                if data.get("supervisor_attempt") == attempt:
                                    run = record.parent
                                    break
                    if (folder / "STOP").exists() and run:
                        (run / "STOP").touch()
                    status = read_json(run / "status.json", {}) if run else {}
                    stamp = (
                        (run / "status.json").stat().st_mtime
                        if run and (run / "status.json").exists()
                        else started
                    )
                    last_activity = max(last_activity, stamp)
                    publish(
                        "running" if status else "starting",
                        child_pid=child.pid,
                        child_status=status,
                    )
                    if time.time() - last_activity > 240:
                        publish("recovering", reason="No learner status update for 240 seconds")
                        terminate_owned(child)
                        break
                    time.sleep(1)
                code = child.wait()
            status = read_json(run / "status.json", {}) if run else {}
            progress, pair = newest_pair(run, pair)
            if (folder / "STOP").exists() or status.get("state") == "stopped":
                publish("stopped", child_exit=code)
                return folder
            if progress >= target:
                break
            delay = min(30, 2 ** min(attempt, 5))
            publish("recovering", child_exit=code, retry_seconds=delay)
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline and not (folder / "STOP").exists():
                time.sleep(0.5)
        publish("stopped" if (folder / "STOP").exists() else "completed")
    except KeyboardInterrupt:
        (folder / "STOP").touch()
        if run:
            (run / "STOP").touch()
        if child:
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                terminate_owned(child)
        publish("stopped")
    finally:
        if child:
            terminate_owned(child)
    return folder
