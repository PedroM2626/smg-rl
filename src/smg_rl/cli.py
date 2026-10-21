"""Command-line entry point. All output and saved artifacts use English."""

import argparse
import time

from smg_rl.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Visual reinforcement learning for Twisty Trials in Dolphin."
    )
    parser.add_argument("--config", default="config/default.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare", help="Create an isolated Dolphin profile and copy local savestates."
    )
    launch_parser = commands.add_parser("launch", help="Boot Dolphin with the isolated profile.")
    launch_parser.add_argument("--slot", type=int, choices=range(1, 5), default=1)
    commands.add_parser("telemetry", help="Print a read-only telemetry snapshot.")
    for command in ("calibrate", "train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument(
            "--attach", action="store_true", help="Use an already running SB4E01 game window."
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
    if args.command == "telemetry":
        import json

        from smg_rl.memory import GameMemory

        print(json.dumps(GameMemory().sample().to_dict(), indent=2))
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
