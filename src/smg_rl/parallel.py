"""Parallel recurrent PPO with private Dolphin processes and GPU optimization."""

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from smg_rl.diagnostics import atomic_json, measured
from smg_rl.env import GalaxyEnv
from smg_rl.native import background_backend
from smg_rl.profiles import require_training_enabled
from smg_rl.timed_ppo import TimedRecurrentPPO
from smg_rl.train import new_run


@dataclass
class WorkerFactory:
    config: dict
    index: int
    run_dir: str

    def __call__(self):
        torch.set_num_threads(self.config["training"]["torch_threads"])
        torch.manual_seed(self.config["training"]["seed"] + self.index)
        time.sleep(self.index * self.config.get("parallel", {}).get("startup_spacing_seconds", 1.0))
        backend = background_backend(self.config, self.index)
        try:
            return GalaxyEnv(
                self.config,
                backend,
                Path(self.run_dir) / f"worker-{self.index:02d}",
                memory=backend.memory,
                variation=self.index % len(self.config["variations"]),
            )
        except BaseException:
            backend.close()
            raise


@measured("checkpoint")
def checkpoint(model, environments, base):
    model.save(str(base))
    torch.save({"workers": environments.env_method("intrinsic_state")}, str(base) + ".intrinsic.pt")


class ProgressCallback(BaseCallback):
    def __init__(self, environments, run_dir, every):
        super().__init__()
        self.environments, self.run_dir, self.every = environments, Path(run_dir), every
        self.last_checkpoint = 0
        self.started = time.monotonic()
        self.start_steps = 0
        self.infos = []

    def _on_training_start(self):
        self.start_steps = self.num_timesteps
        self.last_checkpoint = self.num_timesteps

    def write_status(self, state="running"):
        elapsed = time.monotonic() - self.started
        status = {
            "state": state,
            "timesteps": self.num_timesteps,
            "updates": self.model._n_updates,
            "target_timesteps": self.model._total_timesteps,
            "elapsed_seconds": elapsed,
            "decisions_per_second": (self.num_timesteps - self.start_steps) / max(elapsed, 1e-6),
            "device": str(self.model.device),
            "workers": len(self.infos),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "progress": [info.get("best_progress", 0) for info in self.infos],
            "worker_timing_ms": [info.get("timing_ms", {}) for info in self.infos],
            "backend_timing_ms": [info.get("backend_timing_ms", {}) for info in self.infos],
            "advanced_frames": [info.get("advanced_frames") for info in self.infos],
            "learner_timings": self.model.profiler.snapshot()
            if hasattr(self.model, "profiler")
            else {},
        }
        atomic_json(self.run_dir / "status.json", status)

    def _on_step(self):
        self.infos = self.locals["infos"]
        if self.n_calls % 16 == 0:
            self.write_status()
        for info in self.locals["infos"]:
            name = info.get("variation", "unknown")
            for key in ("best_progress", "intrinsic_reward", "rnd_error"):
                if key in info:
                    self.logger.record_mean(f"{name}/{key}", info[key])
        if self.num_timesteps - self.last_checkpoint >= self.every:
            checkpoint(self.model, self.environments, self.run_dir / f"model-{self.num_timesteps}")
            self.last_checkpoint = self.num_timesteps
            # Retain three complete periodic pairs; long experiments stay bounded.
            pairs = sorted(
                (p for p in self.run_dir.glob("model-*.zip") if p.stem[6:].isdigit()),
                key=lambda p: int(p.stem[6:]),
            )
            for old in pairs[:-3]:
                old.unlink()
                old.with_suffix(".intrinsic.pt").unlink(missing_ok=True)
        return not (self.run_dir / "STOP").exists()


def parallel_train(config, steps=None, workers=None, resume=None):
    require_training_enabled(config)
    config = deepcopy(config)
    workers = workers or config["parallel"]["workers"]
    config["parallel"]["workers"] = workers
    config["training"]["steps"] = steps or config["training"]["steps"]
    settings = config["training"]
    if workers < 1:
        raise ValueError("At least one worker is required.")
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the documented CUDA wheel or select cpu.")
    torch.set_num_threads(settings["torch_threads"])
    torch.manual_seed(settings["seed"])
    run_dir = new_run(config, "parallel")
    print(f"Parallel training output: {run_dir}", flush=True)
    worker_config = deepcopy(config)
    runtime = Path(config["paths"]["user"]).parent / "parallel-profiles" / run_dir.name
    worker_config["paths"]["user"] = str(runtime / "template")
    (run_dir / "controller.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "created_time": psutil.Process().create_time(),
                "workers": workers,
                "requested_steps": settings["steps"],
                "resume": str(resume) if resume else None,
                "runtime": str(runtime),
                "supervisor": os.environ.get("SMG_RL_SUPERVISOR"),
                "supervisor_attempt": int(os.environ.get("SMG_RL_SUPERVISOR_ATTEMPT", "0")),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    environments = None
    model = None
    callback = None
    failed = False
    try:
        environments = SubprocVecEnv(
            [WorkerFactory(worker_config, i, str(run_dir)) for i in range(workers)],
            start_method="spawn",
        )
        environments.seed(settings["seed"])
        if resume:
            base = (
                str(Path(resume).with_suffix("")) if str(resume).endswith(".zip") else str(resume)
            )
            intrinsic = torch.load(base + ".intrinsic.pt", map_location="cpu", weights_only=True)
            states = intrinsic.get("workers", [intrinsic])
            for index in range(workers):
                environments.env_method(
                    "restore_intrinsic_state", states[index % len(states)], indices=index
                )
            model = TimedRecurrentPPO.load(
                base,
                env=environments,
                device=settings["device"],
                n_steps=settings["rollout_steps"],
                batch_size=settings["batch_size"],
            )
        else:
            model = TimedRecurrentPPO(
                "MultiInputLstmPolicy",
                environments,
                learning_rate=settings["learning_rate"],
                n_steps=settings["rollout_steps"],
                batch_size=settings["batch_size"],
                n_epochs=settings["epochs"],
                gamma=settings["gamma"],
                gae_lambda=settings["gae_lambda"],
                ent_coef=settings["entropy"],
                seed=settings["seed"],
                device=settings["device"],
                verbose=1,
                policy_kwargs={"lstm_hidden_size": 128, "net_arch": {"pi": [128], "vf": [128]}},
            )
        model.install_inference_timing()
        callback = ProgressCallback(environments, run_dir, settings["checkpoint_steps"])
        model.learn(
            total_timesteps=steps or settings["steps"],
            reset_num_timesteps=not bool(resume),
            callback=callback,
        )
    except KeyboardInterrupt:
        print("Stopping parallel training and saving a final checkpoint.", flush=True)
    except BaseException as exc:
        failed = True
        (run_dir / "error.txt").write_text(repr(exc) + "\n", encoding="utf-8")
        if model is not None:
            # Worker state may be inaccessible. Preserve the policy for inspection,
            # but do not advertise it as a paired, resumable curiosity checkpoint.
            model.save(str(run_dir / "policy-emergency"))
        raise
    finally:
        try:
            if model and not failed:
                checkpoint(model, environments, run_dir / "model-final")
            if callback:
                callback.write_status(
                    "failed"
                    if failed
                    else "stopped"
                    if (run_dir / "STOP").exists()
                    else "completed"
                )
        finally:
            if environments is not None and not failed:
                environments.close()
            children = psutil.Process().children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(children, timeout=5)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
    return run_dir
