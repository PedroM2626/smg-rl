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
    parser.add_argument("--config", default="config/impala-iqn-smg2.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare", help="Create an isolated Dolphin profile and copy local savestates."
    )
    launch_parser = commands.add_parser("launch", help="Boot Dolphin with the isolated profile.")
    launch_parser.add_argument("--slot", type=int, choices=range(1, 5), default=1)
    commands.add_parser("telemetry", help="Print a read-only telemetry snapshot.")
    report = commands.add_parser(
        "report", help="Audit training success and peak valid progress across games."
    )
    report.add_argument("--output", help="Save JSON and measured best-trajectory figures.")
    commands.add_parser(
        "calibrate-background", help="Validate configured scenarios with a hidden worker."
    )
    benchmark = commands.add_parser(
        "benchmark", help="Measure environment stages without training PPO."
    )
    benchmark.add_argument("--steps", type=int, default=64)
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
    supervised = commands.add_parser(
        "train-supervised", help="Recover failed learners from paired checkpoints."
    )
    supervised.add_argument("--workers", type=int)
    supervised.add_argument("--steps", type=int)
    supervised.add_argument("--resume")
    dqn = commands.add_parser('train-dqn', help='Finite model-free Rainbow/IQN training with TensorBoard.')
    dqn.add_argument('--steps', type=int)
    dqn.add_argument('--max-seconds', type=float, help='Stop and checkpoint at this finite wall-clock budget.')
    dqn.add_argument('--deadline-utc', help='Timezone-qualified absolute collection deadline; checkpoint afterward.')
    dqn.add_argument('--resume', help='Immutable model-free checkpoint directory, including replay.')
    q_video = commands.add_parser('record-dqn', help='Finite canonical-start Rainbow/IQN Q-value/input video.')
    q_video.add_argument('checkpoint')
    q_video.add_argument('--variation', type=int, default=0)
    q_video.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 47])
    q_video.add_argument('--max-steps', type=int, default=300)
    q_video.add_argument('--stochastic', action='store_true',
                         help='Frozen NoisyNet/random-quantile evaluation with checkpoint training epsilon; no learning.')
    for name in ("train-world-model", "fit-world-model"):
        world = commands.add_parser(name, help="Guarded visual world-model workflow (experimental).")
        world.add_argument("--resume")
        if name == "train-world-model":
            world.add_argument("--steps", type=int)
        else:
            world.add_argument("--cycles", type=int, default=100)
    commands.add_parser("inspect-replay", help="Inspect visual replay without collecting or fitting.")
    horizon = commands.add_parser("test-world-horizon", help="Paired horizon prediction audit and guarded offline actor trial.")
    horizon.add_argument("--checkpoint", required=True)
    horizon.add_argument("--updates", type=int, default=64)
    video = commands.add_parser("record-world-policy", help="Record full-start episodes without learning (ffmpeg required).")
    video.add_argument("--checkpoint", required=True)
    video.add_argument("--variation", type=int, default=0)
    video.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    video.add_argument("--max-steps", type=int, default=600)
    ablation = commands.add_parser("ablate-world-model", help="Authorized offline SMG2 objective screening; no emulator.")
    ablation.add_argument("--checkpoint", required=True)
    ablation.add_argument("--updates", type=int, default=128)
    ablation.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    ablation.add_argument("--validation-batches", type=int, default=8)
    ablation.add_argument("--arms", nargs="+", help="Named arms; must include continued_baseline.")
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
    historical = {'train-world-model', 'fit-world-model', 'test-world-horizon', 'ablate-world-model',
                  'record-world-policy', 'inspect-replay', 'train', 'train-parallel', 'train-supervised', 'evaluate'}
    if 'dqn' in config and args.command in historical:
        parser.error('This model-free profile requires train-dqn/record-dqn; select an explicit historical profile for PPO/RSSM commands.')
    if args.command == 'record-dqn':
        from smg_rl.dqn_video import record_dqn

        print(record_dqn(args.config, args.checkpoint, args.variation, args.seeds, args.max_steps, args.stochastic))
        return
    if args.command == 'train-dqn':
        from smg_rl.dqn_training import train_dqn

        print(train_dqn(config, args.steps if args.steps is not None else config['training']['steps'],
                        args.resume, args.max_seconds, args.deadline_utc))
        return
    if args.command == "test-world-horizon":
        from smg_rl.horizon_trial import horizon_trial

        print(horizon_trial(config, args.checkpoint, args.updates))
        return
    if args.command == "record-world-policy":
        from smg_rl.policy_video import record_policy

        print(record_policy(args.config, args.checkpoint, args.variation, args.seeds, args.max_steps))
        return
    if args.command == "ablate-world-model":
        from smg_rl.world_ablation import run_ablation

        print(run_ablation(config, args.checkpoint, args.updates, args.seeds, args.validation_batches, args.arms))
        return
    if args.command == "inspect-replay":
        from smg_rl.replay import SequenceReplay, replay_signature

        dataset = SequenceReplay(config["paths"]["replay"], replay_signature(config))
        print(json.dumps(dataset.refresh(), indent=2))
        return
    if args.command in ("train-world-model", "fit-world-model"):
        require_training_enabled(config)
        from smg_rl.world_training import fit_world_model, train_world_model

        print(train_world_model(config, args.steps, args.resume) if args.command == "train-world-model"
              else fit_world_model(config, args.cycles, args.resume))
        return
    if args.command == "benchmark":
        if args.steps < 1:
            parser.error("--steps must be positive")
        from smg_rl.benchmark import benchmark

        print(benchmark(config, args.steps))
        return
    if args.command == "report":
        from smg_rl.reporting import training_report

        print(json.dumps(training_report(Path(config["root"]) / "runs", args.output), indent=2))
        return
    if args.command in ("train", "train-parallel", "train-supervised"):
        require_training_enabled(config)
    if args.command == "train-supervised":
        from smg_rl.supervision import supervised_train

        if (
            args.steps is not None
            and args.steps <= 0
            or args.workers is not None
            and args.workers <= 0
        ):
            parser.error("Steps and workers must be positive.")
        print(supervised_train(args.config, args.steps, args.workers, args.resume))
        return
    if args.command in ("status", "stop"):
        root = Path(config["paths"]["runs"]).resolve()
        candidates = sorted([*root.glob("parallel-*"), *root.glob("world-model-*"), *root.glob('dqn-*')], key=lambda p: p.stat().st_mtime)
        run = Path(args.run).resolve() if args.run else candidates[-1] if candidates else None
        if run is None or run.parent != root or not (run / "controller.json").is_file():
            parser.error("Choose an existing parallel run in the configured runs directory.")
        controller = json.loads((run / "controller.json").read_text(encoding="utf-8"))
        supervisor = Path(controller["supervisor"]) if controller.get("supervisor") else None
        if supervisor and (
            supervisor.parent != root or not supervisor.name.startswith("supervised-")
        ):
            parser.error("Invalid supervisor ownership metadata.")
        try:
            process = psutil.Process(controller["pid"])
            alive = (
                abs(process.create_time() - controller.get("created_time", process.create_time()))
                < 0.01
            )
        except psutil.NoSuchProcess:
            alive = False
        if args.command == "stop":
            if supervisor:
                (supervisor / "STOP").touch()
            if alive:
                (run / "STOP").touch()
                print(f"Graceful stop requested; the final checkpoint will be saved in {run}.")
            else:
                print(
                    "Supervisor stop requested."
                    if supervisor
                    else "The training controller is no longer running."
                )
        else:
            status = (
                json.loads((run / "status.json").read_text(encoding="utf-8"))
                if (run / "status.json").is_file()
                else {"state": "starting"}
            )
            print(
                json.dumps(
                    {
                        "run": str(run),
                        "controller_running": alive,
                        **controller,
                        **status,
                        "supervisor_status": json.loads(
                            (supervisor / "status.json").read_text(encoding="utf-8")
                        )
                        if supervisor and (supervisor / "status.json").exists()
                        else None,
                    },
                    indent=2,
                )
            )
        return
    if args.command == "calibrate-background":
        from smg_rl.native import background_backend
        from smg_rl.train import new_run
        from smg_rl.validation import calibrate

        if "parallel" not in config:
            parser.error("Use --config config/parallel.yaml for background calibration.")
        output = new_run(config, "background-validation")
        isolated = deepcopy(config)
        isolated["paths"]["user"] = str(
            Path(config["paths"]["user"]).parent / "parallel-profiles" / game_id(config) / output.name / "template"
        )
        backend = background_backend(isolated)
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
