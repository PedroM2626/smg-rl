"""Fault-inject one test-owned native emulator and verify supervised recovery."""

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil
import yaml

from smg_rl.config import load_config
from smg_rl.reporting import read_json
from smg_rl.supervision import newest_pair


def main():
    config = load_config("config/smg1-native.yaml")
    if config["parallel"].get("backend") != "native":
        raise ValueError("This integration check requires the native backend.")
    root = Path(config["root"])
    token = uuid.uuid4().hex[:12]
    output = root / ".local" / f"native-recovery-check-{token}"
    output.mkdir()
    config["paths"]["runs"] = str(output / "runs")
    config["training"].update(rollout_steps=32, batch_size=32, checkpoint_steps=64)
    config["parallel"]["workers"] = 1
    path = root / ".local" / f"native-recovery-config-{token}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "smg_rl.cli",
        "--config",
        str(path),
        "train-supervised",
        "--workers",
        "1",
        "--steps",
        "256",
    ]
    process = None
    supervisor = None
    injected = None
    started = time.monotonic()
    try:
        with (
            (output / "stdout.log").open("w") as stdout,
            (output / "stderr.log").open("w") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            while process.poll() is None:
                if time.monotonic() - started > 240:
                    raise TimeoutError("Recovery integration check exceeded 240 seconds.")
                managers = sorted((output / "runs").glob("supervised-*"))
                if managers:
                    supervisor = managers[-1]
                    status = read_json(supervisor / "status.json", {})
                    if status.get("current_run") and not injected:
                        run = Path(status["current_run"])
                        count, pair = newest_pair(run)
                        if count >= 64 and pair:
                            controller = read_json(run / "controller.json", {})
                            metadata = read_json(
                                Path(controller["runtime"]) / "worker-00/process.json", {}
                            )
                            emulator = psutil.Process(metadata["pid"])
                            learner = psutil.Process(controller["pid"])
                            descendants = {p.pid for p in learner.children(recursive=True)}
                            if emulator.pid not in descendants:
                                raise RuntimeError("Refusing to terminate an unowned process.")
                            if (
                                Path(emulator.exe()).resolve()
                                != Path(config["paths"]["dolphin"]).resolve()
                            ):
                                raise RuntimeError(
                                    "Test-owned process is not the configured emulator."
                                )
                            injected = {
                                "pid": emulator.pid,
                                "checkpoint_steps": count,
                                "checkpoint": str(pair),
                                "run": str(run),
                            }
                            emulator.terminate()
                time.sleep(0.2)
            status = read_json(supervisor / "status.json", {}) if supervisor else {}
            result = {
                "injected_failure": injected,
                "supervisor": status,
                "elapsed_seconds": time.monotonic() - started,
                "passed": bool(
                    injected
                    and status.get("state") == "completed"
                    and status.get("attempts", 0) >= 2
                    and status.get("confirmed_checkpoint_timesteps", 0) >= 256
                ),
            }
            (output / "recovery.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(output, flush=True)
            if not result["passed"]:
                raise RuntimeError(
                    "Native supervised recovery did not pass; inspect its test logs."
                )
    finally:
        if process and process.poll() is None:
            if supervisor:
                (supervisor / "STOP").touch()
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                from smg_rl.supervision import terminate_owned

                terminate_owned(process)


if __name__ == "__main__":
    main()
