"""Bounded full-start visual policy recordings, without any optimizer updates."""

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.env import GalaxyEnv
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.replay import replay_signature
from smg_rl.train import new_run
from smg_rl.world_model import ImaginedAgent


class VideoBackend:
    """Observe every two emulated frames; retain twelve-frame policy decisions."""

    def __init__(self, backend, destination, label):
        self.backend, self.label = backend, label
        self.writer = None
        self.destination = Path(destination)
        self.last_frame_count = 0
        self.episode, self.step = 0, 0
        self.frames_written = 0
        self.telemetry = None
        self.last_action = "reset"

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def frame(self):
        rgba = self.backend.bridge.image()
        if rgba.ndim != 3:
            raise RuntimeError("Video recording needs full RGBA capture.")
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        h, w = rgb.shape[:2]
        canvas = np.zeros((h + 100 + h % 2, w + w % 2, 3), np.uint8)
        canvas[:h, :w] = rgb
        lines = [self.label, f"Episode {self.episode} | decision {self.step} | {self.last_action}"]
        if self.telemetry:
            t = self.telemetry
            lines.append(f"XYZ: {t.position[0]:.1f}, {t.position[1]:.1f}, {t.position[2]:.1f} | HP: {t.health}")
            lines.append(f"Dead: {t.dead} | RAM completion: {t.complete} | 1x game time; no audio")
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (8, h + 19 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (235, 235, 235), 1, cv2.LINE_AA)
        if self.writer is None:
            encoder = shutil.which("ffmpeg")
            if not encoder:
                raise RuntimeError("Install ffmpeg to encode policy videos.")
            self.writer = subprocess.Popen([
                encoder, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{canvas.shape[1]}x{canvas.shape[0]}",
                "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.destination)],
                stdin=subprocess.PIPE)
        self.writer.stdin.write(canvas.tobytes())
        self.frames_written += 1

    def advance(self, names):
        self.last_action = "+".join(names) or "idle"
        self.step += 1
        for _ in range(6):
            self.backend.advance(names)
            self.telemetry = self.backend.memory.sample()
            self.frame()
        self.last_frame_count = 12

    def close(self):
        try:
            if self.writer:
                self.writer.stdin.close()
                if self.writer.wait(timeout=60) != 0:
                    raise RuntimeError("Video encoder failed.")
        finally:
            self.backend.close()


def record_policy(config_path, checkpoint, variation, seeds=(11, 23, 47), maximum_steps=600):
    """Record predeclared episodes, including failures; never resume training."""
    if not seeds or maximum_steps < 1:
        raise ValueError("A recording needs seeds and a positive step budget.")
    config = deepcopy(load_config(config_path))
    if variation not in range(len(config["variations"])):
        raise ValueError("Variation index is outside the configured scenarios.")
    checkpoint = Path(checkpoint).resolve()
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    config["world_model"] = metadata["settings"]
    config["replay"]["enabled"] = False
    config["frontier"] = {"enabled": False}
    config["world_model_run"] = "evaluation"
    config["environment"]["max_steps"] = maximum_steps
    torch.set_num_threads(1)
    agent = ImaginedAgent.load(checkpoint, replay_signature(config), config["world_model"],
                               config["training"]["device"])
    original_counts = (agent.real_decisions, agent.model_updates, agent.actor_updates, agent.imagined_transitions)
    run = new_run(config, "eval-world-video")
    config["paths"]["user"] = str(runtime_directory(config, run) / "template")
    native_config = deepcopy(config)
    native_config["parallel"].update(native_capture="full", frame_skip=2)
    backend = NativeDolphin(native_config)
    name = config["variations"][variation]["name"]
    video = VideoBackend(backend, run / "policy.mp4", f"{config['game']['id']} | {name} | {checkpoint.parent.parent.name}")
    env = None
    result = {"scope": "Full-start policy evaluation; no learning; no training-success accounting",
              "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
              "variation": name, "seeds": list(seeds), "episodes": [],
              "capture": "Native full RGBA, 30 samples/s, policy every 12 frames; silent 1x game time",
              "observation_note": "Full RGBA capture uses the existing OpenCV gray/crop/resize path.",
              "video": str((run / "policy.mp4").resolve())}
    try:
        env = GalaxyEnv(config, video, run / "worker-00", memory=backend.memory,
                        intrinsic=False, variation=variation)
        for episode, seed in enumerate(seeds, 1):
            torch.manual_seed(seed)
            observation, _ = env.reset(seed=seed)
            latent, previous, first = None, None, np.ones(1, dtype=bool)
            video.episode, video.step, video.last_action = episode, 0, "reset"
            video.telemetry = env.telemetry
            video.frame()
            for step in range(maximum_steps):
                batch = {key: value[None] for key, value in observation.items()}
                actions, latent, previous = agent.act(batch, latent, previous, first, epsilon=0.05)
                observation, _, terminated, truncated, info = env.step(int(actions[0]))
                first[:] = False
                if terminated or truncated:
                    break
            result["episodes"].append({"seed": seed, "decisions": step + 1,
                                       "outcome": "success" if info["complete"] else "death" if info["dead"] else "timeout",
                                       "geometric_projection_percent": 100 * info["best_progress"],
                                       "final_telemetry": env.telemetry.to_dict()})
            atomic_json(run / "recording.json", result)
            print(json.dumps({"episode": episode, **result["episodes"][-1]}), flush=True)
        assert original_counts == (agent.real_decisions, agent.model_updates, agent.actor_updates, agent.imagined_transitions)
    finally:
        if env:
            env.close()
        else:
            video.close()
    result["frames_written"] = video.frames_written
    result["state"] = "completed"
    atomic_json(run / "recording.json", result)
    return run
