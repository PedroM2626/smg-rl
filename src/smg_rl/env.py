"""Gymnasium environment. Pixels and scenario identity are the only policy inputs."""

import csv
import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smg_rl.actions import ACTIONS, action_name
from smg_rl.curiosity import IntrinsicReward
from smg_rl.memory import GameMemory, MemoryFault
from smg_rl.plotting import plot_trajectory
from smg_rl.profiles import game_id
from smg_rl.rewards import DamageTracker, ProgressReward


class GalaxyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config, backend, run_dir, memory=None, intrinsic=True, variation=None):
        super().__init__()
        self.config, self.backend = config, backend
        self.memory = memory or GameMemory(game_id=game_id(config))
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        calibration = json.loads(Path(config["paths"]["calibration"]).read_text(encoding="utf-8"))
        if not calibration.get("validated"):
            raise ValueError("Run calibration and validation before training.")
        if calibration.get("game_id", "SB4E01") != game_id(config):
            raise ValueError("Calibration belongs to another game.")
        self.calibration = calibration["variations"]
        self.scenario_count = len(config["variations"])
        n, stack = config["environment"]["image_size"], config["environment"]["frame_stack"]
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, (stack, n, n), np.uint8),
                "scenario": spaces.Box(0, 1, (self.scenario_count,), np.float32),
            }
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.frames = deque(maxlen=stack)
        self.use_intrinsic = intrinsic
        self.curiosity = (
            {
                v["name"]: IntrinsicReward(len(ACTIONS), config["novelty"])
                for v in config["variations"]
            }
            if intrinsic
            else {}
        )
        self.fixed_variation = variation
        self.episode = 0
        self.rows = []
        self.closed = False

    def _observation(self):
        scenario = np.zeros(self.scenario_count, dtype=np.float32)
        scenario[self.config["variations"].index(self.variation)] = 1.0
        return {"image": np.stack(self.frames), "scenario": scenario}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.rows:
            self._save_episode("interrupted")
        options = options or {}
        index = options.get("variation", self.fixed_variation)
        if index is None:
            index = int(self.np_random.integers(self.scenario_count))
        if index not in range(self.scenario_count):
            raise ValueError("Variation index is outside the configured scenarios.")
        self.variation = self.config["variations"][index]
        self.backend.load_slot(self.variation["start_slot"])
        self.telemetry = self.memory.sample()
        if (
            self.telemetry.scenario != self.variation["scenario"]
            or self.telemetry.dead
            or self.telemetry.complete
        ):
            raise MemoryFault("Reset did not restore the requested live starting state.")
        expected = self.calibration[self.variation["name"]]
        if np.linalg.norm(np.asarray(self.telemetry.position) - expected["start"]) > 500:
            raise MemoryFault("Reset position does not match the calibrated start.")
        frame = self.backend.image()
        self.frames.clear()
        self.frames.extend(frame.copy() for _ in range(self.frames.maxlen))
        self.progress = ProgressReward(
            expected["start"],
            expected["goal"],
            self.config["environment"]["min_reward_y"],
            self.config["environment"]["max_step_distance"],
        )
        self.progress.reset(self.telemetry.position)
        self.damage = DamageTracker()
        self.damage.reset(self.telemetry.health)
        self.steps, self.idle_steps, self.total_reward = 0, 0, 0.0
        self.episode += 1
        self.rows = []
        if self.use_intrinsic:
            self.curiosity[self.variation["name"]].reset(frame)
        self._append(-1, 0.0, 0.0, "running", {})
        return self._observation(), self.telemetry.to_dict()

    def step(self, action):
        if not self.rows:
            raise RuntimeError("Call reset before stepping.")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        before = self.frames[-1]
        self.backend.advance(ACTIONS[action])
        self.telemetry = self.memory.sample()
        health_lost = self.damage.update(self.telemetry.health)
        if self.telemetry.scenario != self.variation["scenario"]:
            raise MemoryFault("Scenario changed during an episode.")
        frame = self.backend.image()
        self.frames.append(frame)
        delta, fraction = self.progress.update(
            self.telemetry.position, self.telemetry.dead, self.telemetry.complete
        )
        self.idle_steps = 0 if delta > 1e-5 else self.idle_steps + 1
        weights = self.config["rewards"]
        curiosity, metrics = 0.0, {}
        # Terminal cinematics and falls cannot earn visual novelty rewards.
        if (
            self.use_intrinsic
            and not self.telemetry.dead
            and not self.telemetry.complete
            and health_lost == 0
            and self.telemetry.position[1] >= self.config["environment"]["min_reward_y"]
        ):
            curiosity, metrics = self.curiosity[self.variation["name"]].step(before, frame, action)
        reward = weights["step"] + weights["progress"] * delta + weights["intrinsic"] * curiosity
        if self.telemetry.complete:
            reward += weights["completion"]
        elif self.telemetry.dead:
            reward += weights["death"]
        damage_penalty = (
            weights.get("damage", 0.0) * health_lost
            if not self.telemetry.dead and not self.telemetry.complete
            else 0.0
        )
        reward += damage_penalty
        self.steps += 1
        self.total_reward += reward
        terminated = self.telemetry.complete or self.telemetry.dead
        truncated = not terminated and (
            self.steps >= self.config["environment"]["max_steps"]
            or self.idle_steps >= self.config["environment"]["idle_limit"]
        )
        outcome = (
            "success"
            if self.telemetry.complete
            else "death"
            if self.telemetry.dead
            else "timeout"
            if truncated
            else "running"
        )
        metrics.update(
            progress=fraction,
            best_progress=self.progress.best,
            progress_reward=weights["progress"] * delta,
            intrinsic_reward=weights["intrinsic"] * curiosity,
            is_success=self.telemetry.complete,
            variation=self.variation["name"],
            health_lost=health_lost,
            damage_penalty=damage_penalty,
        )
        self._append(action, reward, fraction, outcome, metrics)
        info = {**self.telemetry.to_dict(), **metrics}
        if terminated or truncated:
            info["episode"] = {"r": self.total_reward, "l": self.steps}
            self._save_episode(outcome)
        return self._observation(), float(reward), terminated, truncated, info

    def _append(self, action, reward, progress, outcome, metrics):
        t = self.telemetry
        self.rows.append(
            dict(
                step=self.steps,
                time=time.time(),
                variation=self.variation["name"],
                action=action_name(action) if action >= 0 else "reset",
                x=t.position[0],
                y=t.position[1],
                z=t.position[2],
                progress=progress,
                best_progress=self.progress.best,
                reward=reward,
                dead=t.dead,
                complete=t.complete,
                actor=f"0x{t.actor:08X}",
                actor_nerve=f"0x{t.actor_nerve:08X}",
                scene_nerve=f"0x{t.scene_nerve:08X}",
                scene_frame=t.scene_frame,
                outcome=outcome,
                intrinsic_reward=metrics.get("intrinsic_reward", 0.0),
                health=t.health,
                max_health=t.max_health,
                health_lost=metrics.get("health_lost", 0),
                damage_penalty=metrics.get("damage_penalty", 0.0),
            )
        )
        if self.steps % 16 == 0 or outcome != "running":
            self._save_live()

    def _save_live(self):
        """Expose a bounded current-episode snapshot without interrupting collection."""
        path = self.run_dir / "live.csv"
        pending = path.with_suffix(".tmp")
        with pending.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        try:
            pending.replace(path)
        except PermissionError:
            # A report/viewer can briefly hold a Windows file handle. Skip this
            # diagnostic snapshot; never abort learning because a viewer is open.
            pass

    def _save_episode(self, outcome):
        if self.steps == 0:
            self.rows = []
            return
        self.rows[-1]["outcome"] = outcome
        folder = self.run_dir / self.variation["name"]
        folder.mkdir(exist_ok=True)
        path = folder / f"episode-{self.episode:06d}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        plot_every = self.config.get("parallel", {}).get("plot_every", 1)
        if outcome == "success" or self.episode % plot_every == 0 or outcome == "interrupted":
            plot_trajectory(path)
        with (self.run_dir / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "episode": self.episode,
                        "variation": self.variation["name"],
                        "outcome": outcome,
                        "reward": self.total_reward,
                        "steps": self.steps,
                        "best_progress": self.progress.best,
                    }
                )
                + "\n"
            )
        self.rows = []

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.rows:
                self._save_episode("interrupted")
        finally:
            self.backend.close()

    def intrinsic_state(self):
        return {name: module.state_dict() for name, module in self.curiosity.items()}

    def restore_intrinsic_state(self, state):
        for name, module in self.curiosity.items():
            module.load_state_dict(state[name])

    def worker_status(self):
        return {
            "pid": self.backend.process.pid,
            "port": self.backend.server.port,
            "frame": self.memory.sample().scene_frame,
            "variation": self.variation["name"] if hasattr(self, "variation") else None,
        }
