"""Recurrent PPO training, matching curiosity checkpoints, and pixel-only evaluation."""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

from smg_rl.env import GalaxyEnv
from smg_rl.profiles import require_training_enabled


def save_checkpoint(model, env, path):
    path = Path(path)
    model.save(str(path))
    torch.save(
        {name: module.state_dict() for name, module in env.curiosity.items()},
        str(path) + ".intrinsic.pt",
    )


class Checkpoints(BaseCallback):
    def __init__(self, env, run_dir, every):
        super().__init__()
        self.env, self.run_dir, self.every = env, Path(run_dir), every

    def _on_step(self):
        info = self.locals["infos"][0]
        name = info.get("variation", "unknown")
        for key in ("best_progress", "intrinsic_reward", "rnd_error"):
            if key in info:
                self.logger.record(f"{name}/{key}", info[key])
        if self.n_calls % self.every == 0:
            save_checkpoint(self.model, self.env, self.run_dir / f"model-{self.num_timesteps}")
        return True


def new_run(config, kind):
    path = Path(config["paths"]["runs"]) / f"{kind}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def train(config, backend, steps=None, resume=None, rollout_steps=None):
    require_training_enabled(config)
    settings = config["training"]
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(settings["torch_threads"])
    run_dir = new_run(config, "train")
    env = GalaxyEnv(config, backend, run_dir)
    model = None
    try:
        if resume:
            base = (
                str(Path(resume).with_suffix("")) if str(resume).endswith(".zip") else str(resume)
            )
            state = torch.load(base + ".intrinsic.pt", map_location="cpu", weights_only=True)
            for name, module in env.curiosity.items():
                module.load_state_dict(state[name])
            model = RecurrentPPO.load(base, env=env, device=settings["device"])
        else:
            rollout = rollout_steps or settings["rollout_steps"]
            model = RecurrentPPO(
                "MultiInputLstmPolicy",
                env,
                learning_rate=settings["learning_rate"],
                n_steps=rollout,
                batch_size=min(settings["batch_size"], rollout),
                n_epochs=settings["epochs"],
                gamma=settings["gamma"],
                gae_lambda=settings["gae_lambda"],
                ent_coef=settings["entropy"],
                seed=seed,
                device=settings["device"],
                verbose=1,
                policy_kwargs={"lstm_hidden_size": 128, "net_arch": {"pi": [128], "vf": [128]}},
            )
        callback = Checkpoints(env, run_dir, settings["checkpoint_steps"])
        print(f"Training output: {run_dir}", flush=True)
        model.learn(
            total_timesteps=steps or settings["steps"],
            callback=callback,
            reset_num_timesteps=not bool(resume),
        )
    except KeyboardInterrupt:
        print("Training interrupted; saving current weights and curiosity state.", flush=True)
    finally:
        try:
            if model is not None:
                save_checkpoint(model, env, run_dir / "model-final")
        finally:
            env.close()
    return run_dir


def evaluate(config, backend, checkpoint, episodes=5):
    torch.set_num_threads(config["training"]["torch_threads"])
    run_dir = new_run(config, "eval")
    env = GalaxyEnv(config, backend, run_dir, intrinsic=False)
    model = RecurrentPPO.load(checkpoint, device=config["training"]["device"])
    results = []
    try:
        for index, variation in enumerate(config["variations"]):
            for episode in range(episodes):
                obs, _ = env.reset(
                    seed=config["training"]["seed"] + episode, options={"variation": index}
                )
                state, starts = None, np.ones((1,), dtype=bool)
                while True:
                    action, state = model.predict(
                        obs, state=state, episode_start=starts, deterministic=True
                    )
                    obs, _, terminated, truncated, info = env.step(int(action))
                    starts = np.array([terminated or truncated])
                    if terminated or truncated:
                        break
                results.append(
                    {
                        "variation": variation["name"],
                        "success": info["is_success"],
                        "best_progress": info["best_progress"],
                    }
                )
        (run_dir / "evaluation.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        for variation in config["variations"]:
            subset = [r for r in results if r["variation"] == variation["name"]]
            print(
                f"{variation['name']}: {sum(r['success'] for r in subset)}/{len(subset)} successes"
            )
    finally:
        env.close()
    return run_dir
