"""Bounded savestate exploration archive with explicit assisted-start accounting."""

import json
import os
import re
import uuid
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json
from smg_rl.memory import MemoryFault


class FrontierArchive:
    """A local worker owns its index; archive states never replace user slots 1-4."""

    def __init__(self, root, identity, settings):
        self.root, self.identity, self.settings = Path(root), identity, settings
        if (not 0 <= settings.get("reset_probability", 0.5) <= 1
                or not 1 <= settings.get("probe_decisions", 5) <= 10
                or not 1 <= settings.get("capacity", 16) <= settings.get("max_candidates", 32) <= 128
                or not np.isfinite(settings.get("cell_size", 300)) or settings.get("cell_size", 300) <= 0):
            raise ValueError("Invalid bounded frontier settings.")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / ".writer.lock").open("a+b")
        try:
            self._lock.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.close()
            raise RuntimeError("Another worker owns this frontier archive.") from exc
        self.index = self.root / "index.json"
        self.entries = []
        if self.index.exists():
            data = json.loads(self.index.read_text(encoding="utf-8"))
            if data["identity"] != identity:
                raise ValueError("Frontier game/scenario/emulator identity mismatch.")
            self.entries = data["entries"]

    def close(self):
        if getattr(self, "_lock", None) is not None:
            self._lock.close()  # OS releases the writer lock, including on process exit.
            self._lock = None

    def __del__(self):
        self.close()

    def save_index(self):
        atomic_json(self.index, {"identity": self.identity, "entries": self.entries})

    def state_path(self, entry):
        if not re.fullmatch(r"[0-9a-f]{32}\.state", entry["file"]):
            raise ValueError("Invalid archive filename.")
        path = self.root / entry["file"]
        if path.resolve().parent != self.root.resolve() or digest(path) != entry["sha256"]:
            raise ValueError("Frontier state integrity mismatch.")
        return path

    def choose(self, rng):
        eligible = [entry for entry in self.entries if entry["accepted"]]
        if not eligible or rng.random() >= self.settings.get("reset_probability", 0.5):
            return None
        weights = np.array([1 / np.sqrt(1 + e.get("visits", 0)) for e in eligible])
        entry = eligible[int(rng.choice(len(eligible), p=weights / weights.sum()))]
        self.state_path(entry)
        entry["visits"] = entry.get("visits", 0) + 1
        self.save_index()
        return entry

    def consider(self, backend, telemetry, history, fraction, source):
        """Screen stability, then idle-probe a copy and restore its exact state.

        The probe is real interaction, separately counted, never learner replay.
        Passing is evidence for this short stability probe, not a grounded flag.
        """
        maximum = int(self.settings.get("max_candidates", 32))
        if len(self.entries) >= maximum or sum(e["accepted"] for e in self.entries) >= self.settings.get("capacity", 16):
            return None, 0
        cell = tuple(np.floor(np.asarray(telemetry.position)[[0, 2]] /
                              self.settings.get("cell_size", 300)).astype(int).tolist())
        if (len(history) < 6 or any(tuple(e["cell"]) == cell for e in self.entries)
                or fraction < self.settings.get("minimum_projection", 0.005)
                or telemetry.dead or telemetry.complete
                or telemetry.position[1] < self.settings["minimum_height"]
                or np.ptp([t.position[1] for t in history]) > 20
                or any(t.dead or t.complete or t.health != telemetry.health for t in history)):
            return None, 0
        filename = uuid.uuid4().hex + ".state"
        path = self.root / filename
        backend.save_state(path)
        entry = {"file": filename, "sha256": digest(path), "cell": cell,
                 "position": list(telemetry.position), "scenario": telemetry.scenario,
                 "projection": fraction, "health": telemetry.health, "source": source,
                 "scene_frame": telemetry.scene_frame,
                 "accepted": True, "visits": 0, "probe_decisions": 0,
                 "screen": "Six-observation vertical stability and bounded idle probe; not native contact verification"}
        count = 0
        try:
            for _ in range(int(self.settings.get("probe_decisions", 5))):
                backend.advance(())
                count += 1
                probe = backend.memory.sample()
                if (probe.dead or probe.complete or probe.scenario != telemetry.scenario
                        or probe.health != telemetry.health
                        or abs(probe.position[1] - telemetry.position[1]) > 25
                        or np.linalg.norm(np.subtract(probe.position, telemetry.position)) > 250):
                    entry["accepted"] = False
                    break
        except MemoryFault as exc:
            entry["accepted"] = False
            entry["probe_error"] = str(exc)
        finally:
            backend.load_state(path)
        restored = backend.memory.sample()
        if (restored.scenario != telemetry.scenario or restored.health != telemetry.health
                or restored.scene_frame != telemetry.scene_frame or restored.scene_nerve != telemetry.scene_nerve
                or restored.dead or restored.complete
                or np.linalg.norm(np.subtract(restored.position, telemetry.position)) > 0.01):
            raise RuntimeError("Frontier probe did not restore its exact starting state.")
        entry["probe_decisions"] = count
        self.entries.append(entry)
        self.save_index()
        return entry, count
