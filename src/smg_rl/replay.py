"""Bounded visual transition recording and episode-separated sequence replay.

Only pixels, action/scenario identity, scalar reward labels and boundary flags
enter these archives. No RAM coordinates/health or serialized Python objects.
"""

import hashlib
import json
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np

from smg_rl.actions import action_name, action_set, execution_signature
from smg_rl.profiles import game_id

SCHEMA = 1
ARRAYS = {"images", "actions", "rewards", "intrinsic_rewards", "terminated", "truncated",
          "is_first", "metadata"}
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


def replay_signature(config):
    result = {"schema": SCHEMA, "game_id": game_id(config),
            "scenario_names": [v["name"] for v in config["variations"]],
            "action_names": [action_name(i, config) for i in range(len(action_set(config)))],
            "image_size": config["environment"]["image_size"],
            "crop": config["environment"]["crop"],
            "frame_skip": config["parallel"]["frame_skip"],
            "reward_revision": config["rewards"].get("revision", "legacy"),
            "reward_weights": {k: v for k, v in config["rewards"].items()
                               if k in ("completion", "death", "damage", "progress", "step", "timeout")}}
    if config.get("actions", {}).get("profile", "legacy54") != "legacy54":
        result["action_execution"] = execution_signature(config)
    if config.get("route_reward", {}).get("enabled"):
        result["route_reward"] = config["route_reward"]
    if config.get('regional_reward', {}).get('enabled'):
        result['regional_reward'] = config['regional_reward']
        result['termination_revision'] = ('budget-truncation-bootstrap-v2'
            if config['regional_reward'].get('timeout_mode') == 'truncation_bootstrap'
            else 'idle-terminal-global-truncation-v3'
            if config['regional_reward'].get('timeout_mode') == 'idle_terminal_penalty'
            else 'absorbing-frame-budget-v1')
    if config.get('curriculum', {}).get('enabled'):
        result['curriculum_distribution'] = config['curriculum']
    if config.get('position_novelty', {}).get('enabled'):
        result['position_novelty'] = config['position_novelty']
    return result


class ReplayRecorder:
    def __init__(self, folder, signature, chunk_steps=256, max_bytes=256 * 1024 * 1024):
        estimated_bytes = (chunk_steps + 1) * signature["image_size"] ** 2 + 20 * chunk_steps + 16384
        if not 1 <= chunk_steps <= 4096 or max_bytes < estimated_bytes or estimated_bytes > MAX_ARCHIVE_BYTES:
            raise ValueError("Replay chunk/budget cannot hold a complete visual chunk.")
        self.folder, self.signature = Path(folder), signature
        self.folder.mkdir(parents=True, exist_ok=True)
        self.chunk_steps, self.max_bytes = chunk_steps, max_bytes
        self.images, self.transitions, self.first = [], [], []
        self.episode_id, self.scenario, self.chunk = None, None, 0

    def _image(self, image):
        image = np.asarray(image)
        size = self.signature["image_size"]
        if image.dtype != np.uint8 or image.shape != (size, size):
            raise ValueError("Replay requires the final cropped uint8 observation frame.")
        return image.copy()

    def begin(self, image, scenario, provenance=None):
        self.flush()
        if not 0 <= scenario < len(self.signature["scenario_names"]):
            raise ValueError("Replay scenario is outside its signature.")
        self.episode_id, self.scenario, self.chunk = uuid.uuid4().hex, scenario, 0
        self.provenance = provenance or {}
        if set(self.provenance) - {'start_kind', 'anchor_id', 'route_region', 'route_sha256'}:
            raise ValueError('Invalid replay reset provenance fields.')
        self.images, self.first, self.transitions = [self._image(image)], [True], []

    def append(self, action, image, reward, intrinsic_reward, terminated, truncated):
        if not self.images or not 0 <= action < len(self.signature["action_names"]):
            raise ValueError("Replay transition has no reset or an invalid action.")
        if not np.isfinite([reward, intrinsic_reward]).all() or (terminated and truncated):
            raise ValueError("Replay reward/boundary labels are invalid.")
        self.images.append(self._image(image))
        self.first.append(False)
        self.transitions.append((action, reward - intrinsic_reward, intrinsic_reward,
                                 bool(terminated), bool(truncated)))
        if len(self.transitions) >= self.chunk_steps or terminated or truncated:
            self.flush()
        if terminated or truncated:
            self.images, self.first = [], []

    def flush(self):
        if not self.transitions:
            return
        actions, rewards, intrinsic, terminated, truncated = zip(*self.transitions)
        metadata = {"signature": self.signature, "episode_id": self.episode_id,
                    "scenario": self.scenario, "chunk": self.chunk,
                    "source": "real_training", "reward_target": "reward_minus_intrinsic",
                    **self.provenance}
        path = self.folder / f"{time.time_ns()}-{self.episode_id}-{self.chunk:05d}.npz"
        pending = path.with_suffix(".tmp")
        try:
            with pending.open("wb") as stream:
                # No compression on the emulator's critical path. uint8 storage
                # is already four times smaller than float32 training tensors.
                np.savez(stream, images=np.stack(self.images), actions=np.asarray(actions, np.int64),
                         rewards=np.asarray(rewards, np.float32), intrinsic_rewards=np.asarray(intrinsic, np.float32),
                         terminated=np.asarray(terminated, np.bool_), truncated=np.asarray(truncated, np.bool_),
                         is_first=np.asarray(self.first, np.bool_), metadata=np.asarray(json.dumps(metadata)))
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        self.images, self.first, self.transitions = [self.images[-1]], [False], []
        self.chunk += 1
        files = sorted(self.folder.glob("*.npz"))
        total = sum(p.stat().st_size for p in files)
        for old in files:
            if total <= self.max_bytes:
                break
            size = old.stat().st_size
            old.unlink()
            total -= size

    def close(self):
        self.flush()


def load_chunk(path, signature=None):
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Replay archive exceeds its size limit.")
    with zipfile.ZipFile(path) as archive:
        if (len(archive.infolist()) != len(ARRAYS)
                or {p.filename for p in archive.infolist()} != {name + ".npy" for name in ARRAYS}
                or sum(p.file_size for p in archive.infolist()) > MAX_ARCHIVE_BYTES):
            raise ValueError("Replay archive has unexpected members or excessive expanded size.")
    with np.load(path, allow_pickle=False) as data:
        result = {name: data[name] for name in ARRAYS}
    meta = json.loads(str(result.pop("metadata")))
    expected = meta["signature"]
    if expected.get("schema") != SCHEMA or signature is not None and expected != signature:
        raise ValueError("Replay signature does not match the model/configuration.")
    if meta.get("source") != "real_training" or meta.get("reward_target") != "reward_minus_intrinsic":
        raise ValueError("Replay source/reward semantics are incompatible.")
    n = len(result["actions"])
    size = expected["image_size"]
    if not 1 <= n <= 4096 or result["images"].shape != (n + 1, size, size) or result["images"].dtype != np.uint8:
        raise ValueError("Replay observation layout is invalid.")
    for name in ("actions", "rewards", "intrinsic_rewards", "terminated", "truncated"):
        if result[name].shape != (n,):
            raise ValueError("Replay transition arrays are misaligned.")
    if (result["actions"].dtype != np.int64 or result["is_first"].shape != (n + 1,)
            or result["is_first"].dtype != np.bool_
            or any(result[name].dtype != np.bool_ for name in ("terminated", "truncated"))
            or any(result[name].dtype != np.float32 for name in ("rewards", "intrinsic_rewards"))
            or not np.isfinite(result["rewards"]).all() or not np.isfinite(result["intrinsic_rewards"]).all()
            or (result["actions"] < 0).any() or (result["actions"] >= len(expected["action_names"])).any()
            or result["is_first"][1:].any()
            or (result["terminated"] & result["truncated"]).any()
            or (result["terminated"][:-1] | result["truncated"][:-1]).any()
            or not 0 <= meta["scenario"] < len(expected["scenario_names"])):
        raise ValueError("Replay values or episode boundaries are invalid.")
    result["metadata"] = meta
    return result


class SequenceReplay:
    def __init__(self, root, signature, seed=42, validation_fraction=0.2):
        if not 0 < validation_fraction < 1:
            raise ValueError("Replay requires a separate validation episode split.")
        self.root, self.signature = Path(root), signature
        self.rng = np.random.default_rng(seed)
        self.validation_fraction = validation_fraction
        self.index, self.rejected = [], []
        self._known = {}

    def refresh(self):
        current, rejected = {}, []
        for path in sorted(self.root.rglob("*.npz")) if self.root.exists() else []:
            try:
                item = self._known.get(path)
                if item is None:
                    chunk = load_chunk(path, self.signature)
                    meta = chunk["metadata"]
                    split = int(hashlib.sha256(meta["episode_id"].encode()).hexdigest()[:8], 16) / 2**32
                    item = {"path": path, "episode_id": meta["episode_id"], "scenario": meta["scenario"],
                            "steps": len(chunk["actions"]), "terminal": bool(chunk["terminated"][-1]),
                            "validation": split < self.validation_fraction,
                            'start_kind': meta.get('start_kind', 'unspecified_legacy')}
                current[path] = item
            except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
                rejected.append({"path": str(path), "error": str(exc)})
        self._known, self.index, self.rejected = current, list(current.values()), rejected
        return self.summary()

    def summary(self):
        return {"chunks": len(self.index), "transitions": sum(x["steps"] for x in self.index),
                "start_kinds": {name: sum(x['steps'] for x in self.index if x['start_kind'] == name)
                                for name in sorted({x['start_kind'] for x in self.index})},
                "train_episodes": len({x["episode_id"] for x in self.index if not x["validation"]}),
                "validation_episodes": len({x["episode_id"] for x in self.index if x["validation"]}),
                "scenarios": {name: sum(x["steps"] for x in self.index if x["scenario"] == i)
                              for i, name in enumerate(self.signature["scenario_names"])},
                "train_scenarios": {name: sum(x["steps"] for x in self.index if x["scenario"] == i and not x["validation"])
                                    for i, name in enumerate(self.signature["scenario_names"])},
                "validation_scenarios": {name: sum(x["steps"] for x in self.index if x["scenario"] == i and x["validation"])
                                         for i, name in enumerate(self.signature["scenario_names"])},
                "rejected": self.rejected.copy()}

    def sample(self, batch_size, length, burn_in=8, validation=False, terminal_fraction=0.0):
        if min(batch_size, length) < 1 or burn_in < 0 or not 0 <= terminal_fraction <= 1:
            raise ValueError("Invalid replay sequence sampling parameters.")
        candidates = [x for x in self.index if x["validation"] == validation]
        if not candidates:
            raise ValueError("No compatible episodes in the requested replay split.")
        samples = []
        for i in range(batch_size):
            # Stratify by scenario so the short/easier variant cannot dominate.
            scenarios = sorted({x["scenario"] for x in candidates})
            pool = [x for x in candidates if x["scenario"] == scenarios[i % len(scenarios)]]
            terminal_pool = [x for x in pool if x["terminal"]]
            tail = bool(terminal_pool and self.rng.random() < terminal_fraction)
            pool = terminal_pool if tail else pool
            item = pool[int(self.rng.integers(len(pool)))]
            chunk = load_chunk(item["path"], self.signature)
            n = item["steps"]
            learn_start = max(0, n - length) if tail else int(self.rng.integers(n))
            if not chunk["is_first"][0]:
                learn_start = max(learn_start, min(burn_in, n - 1))
            start = max(0, learn_start - burn_in)
            width = length + burn_in
            available = min(n - start, width)
            sample = {}
            frames = chunk["images"][start:start + available + 1]
            sample["images"] = np.concatenate([frames, np.repeat(frames[-1:], width - available, axis=0)])
            for name in ("actions", "rewards", "terminated", "truncated"):
                sample[name] = np.pad(chunk[name][start:start + available], (0, width - available))
            sample["is_first"] = np.pad(chunk["is_first"][start:start + available + 1], (0, width - available))
            sample["mask"] = np.zeros(width, np.float32)
            sample["mask"][learn_start - start:available] = 1
            sample["scenario"] = np.eye(len(self.signature["scenario_names"]), dtype=np.float32)[item["scenario"]]
            samples.append(sample)
        return {key: np.stack([x[key] for x in samples]) for key in samples[0]}
