"""Command-line entry point. All output and saved artifacts use English."""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import psutil

from smg_rl.config import load_config
from smg_rl.profiles import game_id, require_training_enabled


def main():
    parser = argparse.ArgumentParser(
        description="Visual reinforcement learning and game telemetry validation in Dolphin."
    )
    parser.add_argument("--config", default="config/default.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare", help="Create an isolated Dolphin profile and copy local savestates."
    )
    launch_parser = commands.add_parser("launch", help="Boot Dolphin with the isolated profile.")
    launch_parser.add_argument("--slot", type=int, choices=range(1, 5), default=1)
    commands.add_parser("telemetry", help="Print a read-only telemetry snapshot.")
    commands.add_parser(
        "calibrate-background", help="Validate configured scenarios with a hidden worker."
    )
    for name in ("status", "stop"):
        command = commands.add_parser(name)
        command.add_argument(
            "run", nargs="?", help="Parallel run directory; defaults to the latest run."
        )
    parallel = commands.add_parser(
        "train-parallel", help="Train with isolated hidden Dolphin processes."
    )
    parallel.add_argument("--workers", type=int)
    parallel.add_argument("--steps", type=int)
    parallel.add_argument("--resume")
    for command in ("calibrate", "train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument(
            "--attach", action="store_true", help="Use an already running configured game window."
        )
        if command == "train":
            sub.add_argument("--steps", type=int)
            sub.add_argument("--rollout-steps", type=int)
            sub.add_argument("--resume")
        if command == "evaluate":
            sub.add_argument("checkpoint")
            sub.add_argument("--episodes", type=int, default=5)
    plot = commands.add_parser("plot")
    plot.add_argument("trajectory")
    plot.add_argument("--output")
    args = parser.parse_args()
    if args.command == "plot":
        from smg_rl.plotting import plot_trajectory

        print(plot_trajectory(args.trajectory, args.output))
        return
    config = load_config(args.config)
    if args.command in ("train", "train-parallel"):
        require_training_enabled(config)
    if args.command in ("status", "stop"):
        root = Path(config["paths"]["runs"]).resolve()
        candidates = sorted(root.glob("parallel-*"))
        run = Path(args.run).resolve() if args.run else candidates[-1] if candidates else None
        if run is None or run.parent != root or not (run / "controller.json").is_file():
            parser.error("Choose an existing parallel run in the configured runs directory.")
        controller = json.loads((run / "controller.json").read_text(encoding="utf-8"))
        try:
            process = psutil.Process(controller["pid"])
            alive = (
                abs(process.create_time() - controller.get("created_time", process.create_time()))
                < 0.01
            )
        except psutil.NoSuchProcess:
            alive = False
        if args.command == "stop":
            if alive:
                (run / "STOP").touch()
                print(f"Graceful stop requested; the final checkpoint will be saved in {run}.")
            else:
                print("The training controller is no longer running.")
        else:
            status = (
                json.loads((run / "status.json").read_text(encoding="utf-8"))
                if (run / "status.json").is_file()
                else {"state": "starting"}
            )
            print(
                json.dumps(
                    {"run": str(run), "controller_running": alive, **controller, **status}, indent=2
                )
            )
        return
    if args.command == "calibrate-background":
        from smg_rl.background import BackgroundDolphin
        from smg_rl.train import new_run
        from smg_rl.validation import calibrate

        if "parallel" not in config:
            parser.error("Use --config config/parallel.yaml for background calibration.")
        output = new_run(config, "background-validation")
        isolated = deepcopy(config)
        isolated["paths"]["user"] = str(
            Path(config["paths"]["user"]).parent / "parallel-profiles" / output.name / "template"
        )
        backend = BackgroundDolphin(isolated)
        try:
            calibrate(isolated, backend, output)
            print(f"Hidden worker calibration passed. Evidence: {output}")
        finally:
            backend.close()
        return
    if args.command == "train-parallel":
        from smg_rl.parallel import parallel_train

        if "parallel" not in config:
            parser.error("Use --config config/parallel.yaml for parallel training.")
        if args.steps is not None and args.steps <= 0:
            parser.error("--steps must be positive")
        if args.workers is not None and args.workers <= 0:
            parser.error("--workers must be positive")
        print(parallel_train(config, args.steps, args.workers, args.resume))
        return
    if args.command == "telemetry":
        from smg_rl.memory import GameMemory

        print(json.dumps(GameMemory(game_id=game_id(config)).sample().to_dict(), indent=2))
        return
    from smg_rl.dolphin import DesktopDolphin, launch, prepare_profile

    if args.command == "prepare":
        print(prepare_profile(config))
        return
    if args.command == "launch":
        print(f"Dolphin process: {launch(config, args.slot).pid}")
        return
    if args.command == "train" and args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.command == "evaluate" and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not args.attach:
        process = launch(config)
        # Wait for boot and the initial savestate; adapter construction validates RAM.
        deadline = time.monotonic() + config["environment"]["reset_timeout"]
        while True:
            try:
                backend = DesktopDolphin(config)
                break
            except (RuntimeError, OSError):
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Dolphin did not boot the configured savestate in time.")
                time.sleep(0.5)
    else:
        backend = DesktopDolphin(config)
    try:
        if args.command == "calibrate":
            from smg_rl.train import new_run
            from smg_rl.validation import calibrate

            output = new_run(config, "validation")
            calibrate(config, backend, output)
            print(f"Calibration passed. Evidence: {output}")
        elif args.command == "train":
            from smg_rl.train import train

            print(train(config, backend, args.steps, args.resume, args.rollout_steps))
        else:
            from smg_rl.train import evaluate

            print(evaluate(config, backend, args.checkpoint, args.episodes))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
