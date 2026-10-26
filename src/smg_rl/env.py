"""Gymnasium environment. Pixels and scenario identity are the only policy inputs."""

import csv
import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smg_rl.actions import action_name, action_set, advance_action
from smg_rl.checkpoints import digest
from smg_rl.curriculum import CurriculumBank
from smg_rl.diagnostics import StageProfiler, atomic_json, measured
from smg_rl.frontier import FrontierArchive
from smg_rl.memory import GameMemory, MemoryFault
from smg_rl.plotting import plot_trajectory
from smg_rl.profiles import game_id
from smg_rl.replay import ReplayRecorder, replay_signature
from smg_rl.rewards import (
    DamageTracker,
    ExplorationBudget,
    ProgressReward,
    RegionalAdvance,
    RegionalPotential,
    validate_simple_objective,
)
from smg_rl.route import RouteProgress


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
        self.actions = action_set(config)
        self.action_space = spaces.Discrete(len(self.actions))
        self.frames = deque(maxlen=stack)
        self.use_intrinsic = intrinsic
        if intrinsic:
            from smg_rl.curiosity import IntrinsicReward
        self.curiosity = (
            {
                v["name"]: IntrinsicReward(len(self.actions), config["novelty"])
                for v in config["variations"]
            }
            if intrinsic
            else {}
        )
        self.fixed_variation = variation
        self.episode = 0
        self.rows = []
        self.closed = False
        self.profiler = StageProfiler()
        self.replay = None
        self.route = None
        if config.get("paths", {}).get("route"):
            route_path = Path(config["paths"]["route"])
            spec = json.loads(route_path.read_text(encoding="utf-8"))
            if spec["game_id"] != game_id(config):
                raise ValueError("Route belongs to another game.")
            if config.get("route_reward", {}).get("enabled") and digest(route_path) != config["route_reward"]["spec_sha256"]:
                raise ValueError("Route reward specification integrity mismatch.")
            self.route = RouteProgress(spec)
        self.archives = {}
        self.regional_settings = config.get('regional_reward', {})
        self.regional = self.regional_settings.get('enabled', False)
        self.potential = None
        if self.regional:
            settings = self.regional_settings
            if (self.route is None or self.route.schema != 3
                    or digest(config['paths']['route']) != settings['spec_sha256']
                    or config.get('route_reward', {}).get('enabled')
                    or config['rewards']['progress'] != 0 or config['rewards']['intrinsic'] != 0
                    or settings['max_idle_frames'] < 1 or settings['max_episode_frames'] < 1):
                raise ValueError('Regional reward requires a signed schema3 route and separate shaping.')
            mode = settings.get('mode', 'potential')
            if mode not in ('potential', 'advance'):
                raise ValueError('Unknown regional reward mode.')
            self.potential = (RegionalAdvance(settings['scale']) if mode == 'advance'
                              else RegionalPotential(settings['scale'], settings['gamma_per_frame']))
            normalization = settings.get('normalization_distance', self.route.total_length)
            if not np.isfinite(normalization) or normalization <= 0:
                raise ValueError('Regional normalization distance must be positive and finite.')
            expected_gamma = settings['gamma_per_frame'] ** config['parallel']['frame_skip']
            learner = config['dqn'] if 'dqn' in config else config['world_model']
            if abs(learner['gamma'] - expected_gamma) > 1e-10:
                raise ValueError('Learner discount must match the regional frame discount.')
            if mode == 'advance':
                validate_simple_objective(settings, config['rewards'], learner['gamma'])
        self.curriculum_bank = None
        curriculum = config.get('curriculum', {})
        if curriculum.get('enabled', False):
            if not self.regional or config.get('frontier', {}).get('enabled'):
                raise ValueError('Curated curriculum requires regional shaping and its own start bank.')
            self.curriculum_bank = CurriculumBank(config['paths']['curriculum'], curriculum['spec_sha256'],
                game_id(config), digest(config['paths']['dolphin']), curriculum['probability'])
        self.frontier_history = deque(maxlen=6)
        self.frontier_validation_decisions = 0
        replay = config.get("replay", {})
        training_run = None
        for parent in (self.run_dir, *self.run_dir.parents):
            if parent.name.startswith(("benchmark-", "background-validation-", "damage-validation-", "eval-")):
                break
            if parent.name.startswith(("parallel-", "train-", "world-model-")):
                training_run = parent
                break
        if replay.get("enabled", False) and training_run is not None:
            root = Path(config["paths"]["replay"]) / training_run.name
            folder = root / self.run_dir.relative_to(training_run)
            self.replay = ReplayRecorder(folder, replay_signature(config),
                                         replay.get("chunk_steps", 256), replay.get("max_worker_bytes", 268435456))

    def _potential_fraction(self):
        distance = self.route.best * self.route.total_length
        normalization = self.regional_settings.get('normalization_distance', self.route.total_length)
        return float(min(1.0, distance / normalization))

    def _observation(self):
        scenario = np.zeros(self.scenario_count, dtype=np.float32)
        scenario[self.config["variations"].index(self.variation)] = 1.0
        return {"image": np.stack(self.frames), "scenario": scenario}

    @measured("reset")
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
        self.backend._motor_direction = ()
        self.start_kind, self.start_state_id = "full_start", ""
        self.archive = None
        settings = self.config.get("frontier", {})
        if settings.get("enabled", False):
            name = self.variation["name"]
            if name not in self.archives:
                identity = {"game_id": game_id(self.config), "scenario": self.variation["scenario"],
                            "emulator_sha256": digest(self.config["paths"]["dolphin"])}
                minimum = self.calibration[name]["start"][1] - 50
                self.archives[name] = FrontierArchive(
                    Path(self.config["paths"]["frontier"]) / self.run_dir.name / name,
                    identity, {"minimum_height": minimum, **settings})
            self.archive = self.archives[name]
        curated = (self.curriculum_bank.choose(self.np_random, self.variation['scenario'])
                   if self.curriculum_bank and not options.get('full_start', False) else None)
        entry = curated or (self.archive.choose(self.np_random) if self.archive else None)
        if entry:
            owner = self.curriculum_bank if curated else self.archive
            self.backend.load_state(owner.state_path(entry))
            self.start_kind, self.start_state_id = 'curriculum' if curated else 'frontier', entry['file']
        else:
            self.backend.load_slot(self.variation["start_slot"])
        self.telemetry = self.memory.sample()
        if (
            self.telemetry.scenario != self.variation["scenario"]
            or self.telemetry.dead
            or self.telemetry.complete
        ):
            raise MemoryFault("Reset did not restore the requested live starting state.")
        expected = self.calibration[self.variation["name"]]
        if curated:
            if curated['route_sha256'] != self.regional_settings['spec_sha256']:
                raise ValueError('Curriculum anchor belongs to another regional route.')
            self.curriculum_bank.verify_reset(curated, self.telemetry, expected['goal'])
        target = entry["position"] if entry else expected["start"]
        if np.linalg.norm(np.asarray(self.telemetry.position) - target) > (0.01 if entry else 500):
            raise MemoryFault("Reset position does not match the calibrated start.")
        if entry and (self.telemetry.scene_frame != entry["scene_frame"]
                      or self.telemetry.health != entry["health"]):
            raise MemoryFault("Frontier reset did not restore time/health context.")
        self.frontier_history.clear()
        self.frontier_history.append(self.telemetry)
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
        if self.route:
            self.route.reset(self.telemetry, segment=int(curated['region']) if curated else None)
        if self.potential:
            self.potential.reset(self._potential_fraction())
        self.route_start = self.route.best if self.route else 0.0
        self.damage = DamageTracker()
        self.damage.reset(self.telemetry.health)
        self.steps, self.idle_steps, self.total_reward = 0, 0, 0.0
        self.emulated_frames, self.idle_frames = 0, 0
        weights = self.config["rewards"]
        self.exploration_budget = ExplorationBudget(
            weights.get("intrinsic_episode_budget"),
            weights.get("intrinsic_decay_start"), weights.get("intrinsic_decay_end"),
        )
        self.episode += 1
        self.rows = []
        if self.replay:
            provenance = ({'start_kind': self.start_kind, 'anchor_id': self.start_state_id,
                           'route_region': self.route.segment,
                           'route_sha256': self.regional_settings['spec_sha256']} if self.regional else None)
            self.replay.begin(frame, index, provenance=provenance)
        if self.use_intrinsic:
            self.curiosity[self.variation["name"]].reset(frame)
        self._append(-1, 0.0, self.progress.best, "running", {})
        return self._observation(), {**self.telemetry.to_dict(), 'start_kind': self.start_kind,
                                     'anchor_id': self.start_state_id}

    @measured("step")
    def step(self, action):
        if not self.rows:
            raise RuntimeError("Call reset before stepping.")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        before = self.frames[-1]
        previous_telemetry = self.telemetry
        with self.profiler.measure("environment_control_and_capture"):
            advance_action(self.backend, self.actions[action], self.config)
        with self.profiler.measure("telemetry"):
            self.telemetry = self.memory.sample()
        health_lost = self.damage.update(self.telemetry.health)
        if self.telemetry.scenario != self.variation["scenario"]:
            raise MemoryFault("Scenario changed during an episode.")
        frame = self.backend.image()
        self.frames.append(frame)
        delta, fraction = self.progress.update(
            self.telemetry.position, self.telemetry.dead, self.telemetry.complete
        )
        route_delta = self.route.update(self.telemetry) if self.route else 0.0
        route_settings = self.config.get("route_reward", {})
        advance = route_delta if route_settings.get("enabled") or self.regional else delta
        self.idle_steps = 0 if advance > 1e-5 else self.idle_steps + 1
        measured_frames = getattr(self.backend, 'last_frame_count', None)
        control_frames = int(measured_frames) if measured_frames is not None else None
        if self.regional and control_frames != self.config['parallel']['frame_skip']:
            raise RuntimeError('Regional shaping requires verified native frame counts.')
        self.emulated_frames = (self.emulated_frames + control_frames
                                if control_frames is not None and self.emulated_frames is not None else None)
        self.idle_frames = 0 if advance > 1e-5 else self.idle_frames + (control_frames or 0)
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
            with self.profiler.measure("curiosity"):
                curiosity, metrics = self.curiosity[self.variation["name"]].step(
                    before, frame, action
                )
        raw_intrinsic = weights["intrinsic"] * curiosity
        intrinsic_reward, intrinsic_gate = self.exploration_budget.pay(raw_intrinsic, self.idle_steps)
        step_reward = weights['step'] * (control_frames / 12 if self.regional else 1)
        reward = step_reward + weights["progress"] * delta + intrinsic_reward
        route_reward = route_settings.get("weight", 0.0) * route_delta if route_settings.get("enabled") else 0.0
        reward += route_reward
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
        terminated = self.telemetry.complete or self.telemetry.dead
        truncated = not terminated and (
            self.steps >= self.config["environment"]["max_steps"]
            or self.idle_steps >= self.config["environment"]["idle_limit"]
        )
        timed_out = truncated
        if self.regional:
            timed_out = not terminated and (
                self.emulated_frames >= self.regional_settings['max_episode_frames']
                or self.idle_frames >= self.regional_settings['max_idle_frames'])
            terminated = terminated or timed_out
            truncated = False
        shaping_reward = self.potential.update(self._potential_fraction(), control_frames, terminated) if self.potential else 0.0
        reward += shaping_reward
        self.frontier_history.append(self.telemetry)
        if self.archive and not terminated and not truncated:
            _, probes = self.archive.consider(
                self.backend, self.telemetry, self.frontier_history, fraction,
                {"episode": self.episode, "step": self.steps, "start_kind": self.start_kind,
                 "parent_state": self.start_state_id, "run": str(self.run_dir)})
            self.frontier_validation_decisions += probes
        timeout_penalty = weights.get("timeout", 0.0) if timed_out else 0.0
        reward += timeout_penalty
        if self.replay:
            with self.profiler.measure("replay_recording"):
                self.replay.append(action, frame, reward, intrinsic_reward, terminated, truncated)
        self.total_reward += reward
        outcome = (
            "success"
            if self.telemetry.complete
            else "death"
            if self.telemetry.dead
            else "timeout"
            if timed_out
            else "running"
        )
        metrics.update(
            worker_episode=self.episode,
            episode_step=self.steps,
            start_kind=self.start_kind,
            frontier_state=self.start_state_id,
            frontier_validation_decisions=self.frontier_validation_decisions,
            progress=fraction,
            best_progress=self.progress.best,
            progress_reward=weights["progress"] * delta,
            intrinsic_reward=intrinsic_reward,
            raw_intrinsic_reward=raw_intrinsic,
            intrinsic_gate=intrinsic_gate,
            intrinsic_budget_remaining=self.exploration_budget.remaining,
            timeout_penalty=timeout_penalty,
            is_success=self.telemetry.complete,
            variation=self.variation["name"],
            health_lost=health_lost,
            damage_penalty=damage_penalty,
            route_prefix_progress=self.route.best if self.route else None,
            route_prefix_exited=self.route.exited if self.route else False,
            route_reward=route_reward,
            regional_shaping_reward=shaping_reward,
            shaping_potential=self.potential.value if self.potential else None,
            region_index=self.route.segment if self.regional else None,
            start_relative_route_progress=self.route.best - self.route_start if self.route else None,
            emulated_frames=self.emulated_frames,
            step_reward=step_reward,
        )
        with self.profiler.measure("logging"):
            self._append(action, reward, fraction, outcome, metrics)
        info = {**self.telemetry.to_dict(), **metrics}
        info["timing_ms"] = {
            name: value
            for name, value in self.profiler.last.items()
            if name not in ("step", "reset")
        }
        info["backend_timing_ms"] = dict(getattr(self.backend, "profiler", self.profiler).last)
        info["advanced_frames"] = getattr(self.backend, "last_frame_count", None)
        if hasattr(self.backend, "bridge"):
            info["native_diagnostics"] = self.backend.bridge.diagnostics()
        info["scene_frames_advanced"] = (
            self.telemetry.scene_frame - previous_telemetry.scene_frame
            if self.telemetry.scene_nerve == previous_telemetry.scene_nerve
            and self.telemetry.scene_frame >= previous_telemetry.scene_frame
            else None
        )
        if self.steps % 16 == 0 or terminated or truncated:
            atomic_json(
                self.run_dir / "timings.json",
                {
                    "environment": self.profiler.snapshot(),
                    "backend": getattr(self.backend, "profiler", self.profiler).snapshot(),
                    "note": "Nested wall-clock stages overlap; recent percentiles use at most 512 calls.",
                },
            )
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
                start_kind=self.start_kind,
                frontier_state=self.start_state_id,
                frontier_validation_decisions=self.frontier_validation_decisions,
                action=action_name(action, self.config) if action >= 0 else "reset",
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
                raw_intrinsic_reward=metrics.get("raw_intrinsic_reward", 0.0),
                progress_reward=metrics.get("progress_reward", 0.0),
                episodic_novelty=metrics.get("episodic_novelty", 0.0),
                rnd_error=metrics.get("rnd_error", 0.0),
                lifelong_novelty=metrics.get("lifelong_novelty", 0.0),
                intrinsic_gate=metrics.get("intrinsic_gate", 1.0),
                intrinsic_budget_remaining=metrics.get("intrinsic_budget_remaining"),
                timeout_penalty=metrics.get("timeout_penalty", 0.0),
                curiosity_update=metrics.get("curiosity_update", 0),
                curiosity_batch_size=metrics.get("curiosity_batch_size", 0),
                health=t.health,
                max_health=t.max_health,
                grounded=t.grounded,
                route_prefix_progress=metrics.get("route_prefix_progress"),
                route_prefix_exited=metrics.get("route_prefix_exited", False),
                route_reward=metrics.get("route_reward", 0.0),
                regional_shaping_reward=metrics.get('regional_shaping_reward', 0.0),
                shaping_potential=metrics.get('shaping_potential', self.potential.value if self.potential else None),
                region_index=metrics.get('region_index', self.route.segment if self.regional else None),
                start_relative_route_progress=metrics.get('start_relative_route_progress', 0.0 if self.route else None),
                emulated_frames=self.emulated_frames,
                step_reward=metrics.get('step_reward', 0.0),
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

    @measured("episode_output")
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
                        "intrinsic_total": sum(r["intrinsic_reward"] for r in self.rows),
                        "progress_total": sum(r["progress_reward"] for r in self.rows),
                        "timeout_total": sum(r["timeout_penalty"] for r in self.rows),
                        "reward_revision": self.config["rewards"].get("revision", "legacy"),
                        "start_kind": self.start_kind,
                        "frontier_state": self.start_state_id,
                        "route_prefix_progress": self.route.best if self.route else None,
                        "route_prefix_exited": self.route.exited if self.route else False,
                        "route_reward_total": sum(r["route_reward"] for r in self.rows),
                        "regional_shaping_total": sum(r['regional_shaping_reward'] for r in self.rows),
                        "emulated_frames": self.emulated_frames,
                        "route_start": self.route_start,
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
            try:
                if self.replay:
                    self.replay.close()
            finally:
                try:
                    for archive in self.archives.values():
                        if hasattr(archive, "close"):
                            archive.close()
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
