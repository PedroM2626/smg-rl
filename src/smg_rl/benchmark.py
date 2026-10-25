"""Repeatable environment benchmarks excluded from all training success reports."""

import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from smg_rl.actions import action_set
from smg_rl.diagnostics import atomic_json
from smg_rl.env import GalaxyEnv
from smg_rl.native import background_backend
from smg_rl.profiles import game_id
from smg_rl.train import new_run


def benchmark(config, steps=64):
    config = deepcopy(config)
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.manual_seed(42)
    output = new_run(config, "benchmark")
    config["paths"]["user"] = str(
        Path(config["paths"]["user"]).parent / "benchmark-profiles" / game_id(config) / output.name / "template"
    )
    backend = background_backend(config)
    try:
        env = GalaxyEnv(config, backend, output / "worker-00", memory=backend.memory, variation=0)
    except BaseException:
        backend.close()
        raise
    available = action_set(config)
    actions = [available.index(names) for names in ((), ("w",), ("jump",), ("s",), ("w", "jump"))
               if names in available]
    durations, frames, scene_frames, native_samples = [], [], [], []
    try:
        env.reset(seed=42)
        for index in range(steps):
            started = time.perf_counter()
            _, _, terminated, truncated, info = env.step(actions[index % len(actions)])
            durations.append(time.perf_counter() - started)
            frames.append(info.get("advanced_frames"))
            scene_frames.append(info.get("scene_frames_advanced"))
            native_samples.append(info.get("native_diagnostics", {}))
            if terminated or truncated:
                env.reset(seed=42)
        result = {
            "backend": config["parallel"].get("backend", "legacy"),
            "steps": steps,
            "game_id": config.get("game", {}).get("id", "SB4E01"),
            "frame_skip": config["parallel"]["frame_skip"],
            "emulation_speed": config["parallel"]["emulation_speed"],
            "seed": 42,
            "skip_present": config["parallel"].get("skip_present", False),
            "bridge_wakeup": config["parallel"].get("bridge_wakeup", "events"),
            "native_capture": config["parallel"].get("native_capture", "cpu"),
            "quiet_ui": config["parallel"].get("quiet_ui", True),
            "curiosity_update_every": config["novelty"].get("update_every", 1),
            "torch_threads": config["training"]["torch_threads"],
            "reward_revision": config["rewards"].get("revision", "legacy"),
            "native_diagnostics": backend.bridge.diagnostics() if hasattr(backend, "bridge") else None,
            "native_stages": {
                key: {"mean": float(np.mean([s[key] for s in native_samples])),
                      "p50": float(np.percentile([s[key] for s in native_samples], 50)),
                      "p95": float(np.percentile([s[key] for s in native_samples], 95))}
                for key in native_samples[0] if key.endswith("_us") or key == "image_bytes"
            } if native_samples else {},
            "action_profile":config.get('actions',{}).get('profile','legacy54'),
            "action_cycle": [list(available[index]) for index in actions],
            "step_wall_seconds": sum(durations),
            "decisions_per_second_excluding_resets": steps / sum(durations),
            "step_p50_ms": float(np.percentile(durations, 50) * 1000),
            "step_p95_ms": float(np.percentile(durations, 95) * 1000),
            "native_completed_frames": frames,
            "scene_frame_deltas": scene_frames,
            "environment": env.profiler.snapshot(),
            "backend_stages": backend.profiler.snapshot(),
            "note": "Fixed action cycle, one worker, curiosity enabled; startup/reset time separate.",
        }
        atomic_json(output / "benchmark.json", result)
        return output
    finally:
        env.close()
