"""Bounded full-start visual policy recordings, without any optimizer updates."""

import hashlib
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


class VideoBackend:
    """Observe every two emulated frames with signed four/twelve-frame decisions."""

    def __init__(self, backend, destination, label, decision_frames=12):
        self.backend, self.label = backend, label
        self.writer = None
        self.destination = Path(destination)
        self.last_frame_count = 0
        self.episode, self.step = 0, 0
        self.frames_written = 0
        self.telemetry = None
        self.last_action = "reset"
        self.last_controls = ()
        self.policy_image = None
        self.policy_probabilities = None
        self.policy_q_values = None
        self.input_log = None
        if decision_frames not in (4, 12):
            raise ValueError('Policy video supports signed four/twelve-frame decisions.')
        self.decision_frames = decision_frames

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def compose_frame(self):
        rgba = self.backend.bridge.image()
        if rgba.ndim != 3:
            raise RuntimeError("Video recording needs full RGBA capture.")
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        h, w = rgb.shape[:2]
        body_height = max(h, 560)
        canvas = np.zeros((body_height + 100 + body_height % 2, w + 340 + w % 2, 3), np.uint8)
        canvas[:h, :w] = rgb
        x = w + 14
        cv2.putText(canvas, "Actual policy input", (x, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (235,235,235), 1, cv2.LINE_AA)
        if self.policy_image is not None:
            tile = cv2.resize(self.policy_image, (252,252), interpolation=cv2.INTER_NEAREST)
            canvas[36:288, x:x+252] = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
        shape = self.policy_image.shape if self.policy_image is not None else (0,0)
        input_label = f"{shape[1]}x{shape[0]} grayscale; held for this decision" if shape[0] else "No policy sample (control fixture)"
        cv2.putText(canvas, input_label, (x, 308),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, (190,190,190), 1, cv2.LINE_AA)
        labels = ((None,"None"),("d","Right (D)"),("a","Left (A)"),("w","Forward (W)"),
                  ("s","Backward (S)"),("jump","Jump (LMB)"),("spin","Spin (MMB)"),("crouch","Crouch (Shift)"))
        for i,(control,label) in enumerate(labels):
            y = 329 + 27*i
            pressed = not self.last_controls if control is None else control in self.last_controls
            color = (238,187,46) if pressed else (70,68,90)
            cv2.rectangle(canvas, (x,y), (x+310,y+22), color, -1)
            if self.policy_probabilities is not None and len(self.policy_probabilities) == 8:
                probability = float(self.policy_probabilities[i])
                cv2.rectangle(canvas, (x,y+19), (x+round(310*probability),y+22), (90,210,160), -1)
                label += f"  {100*probability:.1f}%"
            elif self.policy_q_values is not None:
                value = float(self.policy_q_values[i])
                label += f'  Q {value:.3f}'
            cv2.putText(canvas, label, (x+6,y+16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (12,12,12) if pressed else (235,235,235), 1, cv2.LINE_AA)
        lines = [self.label, f"Episode {self.episode} | decision {self.step} | {self.last_action}"]
        if self.telemetry:
            t = self.telemetry
            lines.append(f"XYZ: {t.position[0]:.1f}, {t.position[1]:.1f}, {t.position[2]:.1f} | HP: {t.health}")
            lines.append(f"Dead: {t.dead} | RAM completion: {t.complete} | 1x game time; no audio")
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (8, body_height + 19 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (235, 235, 235), 1, cv2.LINE_AA)
        return canvas

    def frame(self):
        canvas = self.compose_frame()
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
        if self.input_log is None:
            self.input_log = self.destination.with_suffix('.inputs.jsonl').open('x', encoding='utf-8')
        self.input_log.write(json.dumps({'frame':self.frames_written, 'episode':self.episode, 'decision':self.step,
            'selected_action':self.last_action, 'applied_controls':self.last_controls,
            'policy_expected_returns': self.policy_q_values.tolist() if self.policy_q_values is not None else None,
            'policy_input_sha256':hashlib.sha256(self.policy_image.tobytes()).hexdigest()
                if self.policy_image is not None else None})+'\n')
        self.frames_written += 1

    def advance(self, names):
        self.last_action = self.__dict__.get('logical_action', '+'.join(names) or 'none')
        self.step += 1
        for _ in range(self.decision_frames // 2):
            self.backend.advance(names)
            self.last_controls = tuple(names)
            self.telemetry = self.backend.memory.sample()
            self.frame()
        self.last_frame_count = self.decision_frames

    def advance_pulse(self, names, pressed, released):
        if pressed + released != self.decision_frames or pressed % 2 or released % 2:
            raise ValueError('Video pulses must match the decision in two-frame chunks.')
        self.last_action = self.__dict__.get('logical_action', '+'.join(names))
        self.step += 1
        for controls, count in ((names, pressed), (tuple(n for n in names if n not in ("jump", "spin")), released)):
            for _ in range(count // 2):
                self.backend.advance(controls)
                self.last_controls = tuple(controls)
                self.telemetry = self.backend.memory.sample()
                self.frame()
        self.last_frame_count = self.decision_frames

    def close(self):
        try:
            if self.writer:
                self.writer.stdin.close()
                if self.writer.wait(timeout=60) != 0:
                    raise RuntimeError("Video encoder failed.")
        finally:
            try:
                if self.input_log:
                    self.input_log.close()
            finally:
                self.backend.close()


def record_policy(config_path, checkpoint, variation, seeds=(11, 23, 47), maximum_steps=600):
    """Record predeclared episodes, including failures; never resume training."""
    from smg_rl.world_model import ImaginedAgent

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
    policy_stage = 'UNTRAINED POLICY PRIOR' if agent.actor_updates == 0 else 'POLICY EVALUATION'
    video = VideoBackend(backend, run / "policy.mp4",
                         f"{config['game']['id']} | {name} | {policy_stage} | real {agent.real_decisions} | actor updates {agent.actor_updates}",
                         decision_frames=config['parallel']['frame_skip'])
    env = None
    result = {"scope": "Full-start policy evaluation; no learning; no training-success accounting",
              "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
              "variation": name, "seeds": list(seeds), "episodes": [],
              "capture": f"Native full RGBA, 30 samples/s, policy every {config['parallel']['frame_skip']} frames; silent 1x game time",
              "observation_note": "Full RGBA capture uses the existing OpenCV gray/crop/resize path.",
              "policy_input_overlay": "Exact cropped grayscale frame used by the actor, held for each decision; recurrent latent memory is not visualized.",
              "button_overlay": "Applied controls per two-frame sample; policy probabilities are epsilon-mixture probabilities, not Q values.",
              "source_actor_updates":agent.actor_updates,
              "video": str((run / "policy.mp4").resolve())}
    try:
        env = GalaxyEnv(config, video, run / "worker-00", memory=backend.memory,
                        intrinsic=False, variation=variation)
        for episode, seed in enumerate(seeds, 1):
            torch.manual_seed(seed)
            observation, _ = env.reset(seed=seed, options={'full_start': True})
            latent, previous, first = None, None, np.ones(1, dtype=bool)
            video.episode, video.step, video.last_action = episode, 0, "reset"
            video.policy_image = observation['image'][-1].copy()
            video.last_controls, video.policy_probabilities = (), None
            video.telemetry = env.telemetry
            video.frame()
            for step in range(maximum_steps):
                batch = {key: value[None] for key, value in observation.items()}
                video.policy_image = observation['image'][-1].copy()
                actions, latent, previous = agent.act(batch, latent, previous, first, epsilon=0.05)
                with torch.no_grad():
                    scene = torch.as_tensor(batch['scenario'], device=agent.device)
                    probabilities = agent.actor(agent.world.feature(latent,scene)).softmax(-1)[0]
                    video.policy_probabilities = (0.95*probabilities + 0.05/len(probabilities)).cpu().numpy()
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
