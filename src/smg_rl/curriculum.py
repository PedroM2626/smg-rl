"""Immutable curated intermediate starts with explicit assisted-start provenance."""

import json
import re
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest


class CurriculumBank:
    def __init__(self, path, expected_hash, game_id, emulator_hash, probability):
        self.path = Path(path)
        if not 0 <= probability <= .5 or digest(self.path) != expected_hash:
            raise ValueError('Curriculum probability/specification integrity mismatch.')
        self.probability = probability
        self.data = json.loads(self.path.read_text())
        if (self.data.get('schema') != 1 or self.data['game_id'] != game_id
                or self.data['emulator_sha256'] != emulator_hash):
            raise ValueError('Curriculum executable/emulator identity mismatch.')
        self.entries = self.data['entries']
        for entry in self.entries:
            if (entry.get('accepted') is not True or entry['origin_slot'] not in (1, 3)
                    or entry.get('near_star_fixture') is not False
                    or entry['validation']['passed'] is not True
                    or entry['validation']['idle_frames'] < 60):
                raise ValueError('Unvalidated/near-star state cannot enter the curriculum bank.')
            self.state_path(entry)

    def state_path(self, entry):
        if not re.fullmatch(r'[a-z0-9-]{1,60}\.state', entry['file']):
            raise ValueError('Invalid curriculum state filename.')
        path = self.path.parent / entry['file']
        if path.resolve().parent != self.path.parent.resolve() or digest(path) != entry['sha256']:
            raise ValueError('Curriculum state integrity mismatch.')
        return path

    def choose(self, rng, scenario):
        entries = [e for e in self.entries if e['scenario'] == scenario]
        if not entries or rng.random() >= self.probability:
            return None
        return entries[int(rng.integers(len(entries)))]

    @staticmethod
    def verify_reset(entry, telemetry, goal):
        if (not np.isfinite(telemetry.position).all() or not np.isfinite(goal).all()
                or telemetry.dead or telemetry.complete or telemetry.grounded is not True
                or telemetry.scenario != entry['scenario'] or telemetry.health != entry['health']
                or telemetry.scene_frame != entry['scene_frame']
                or np.linalg.norm(np.subtract(telemetry.position, entry['position'])) > .01
                or np.linalg.norm(np.subtract(telemetry.position, goal)) < 2000):
            raise ValueError('Curriculum reset did not reproduce its verified live context.')
