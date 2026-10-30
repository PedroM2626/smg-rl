"""Immutable curated intermediate starts with explicit assisted-start provenance."""

import json
import re
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json


class CurriculumBank:
    def __init__(self, path, expected_hash, game_id, emulator_hash, probability, progress_path=None):
        self.path = Path(path)
        if not 0 <= probability <= .5 or digest(self.path) != expected_hash:
            raise ValueError('Curriculum probability/specification integrity mismatch.')
        self.probability = probability
        self.data = json.loads(self.path.read_text())
        if (self.data.get('schema') not in (1, 2) or self.data['game_id'] != game_id
                or self.data['emulator_sha256'] != emulator_hash):
            raise ValueError('Curriculum executable/emulator identity mismatch.')
        self.entries = self.data['entries']
        self.expected_hash = expected_hash
        self.progress_path = Path(progress_path) if progress_path else None
        if self.data['schema'] == 2:
            indices = [e.get('route_waypoint') for e in self.entries]
            if (not self.data.get('source_full_start_ram_star') or not indices
                    or any(type(i) is not int or i < 1 for i in indices)
                    or indices != sorted(set(indices)) or self.progress_path is None):
                raise ValueError('Human backward curriculum requires ordered verified anchors.')
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
        if self.data['schema'] == 2:
            frontier = len(entries) - 1
            if self.progress_path.exists():
                progress = json.loads(self.progress_path.read_text())
                frontier = progress['frontier']
                if (progress['bank_sha256'] != self.expected_hash or type(frontier) is not int
                        or not 0 <= frontier < len(entries)):
                    raise ValueError('Backward curriculum frontier identity is invalid.')
            if frontier + 1 < len(entries) and rng.random() < .2:
                return entries[int(rng.integers(frontier + 1, len(entries)))]
            return entries[frontier]
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


class BackwardCurriculum:
    """The learner alone advances a signed frontier after eight of ten RAM stars."""

    def __init__(self, bank_path, expected_hash, progress_path):
        self.path = Path(progress_path)
        bank_path = Path(bank_path)
        if digest(bank_path) != expected_hash:
            raise ValueError('Backward scheduler bank hash mismatch.')
        bank = json.loads(bank_path.read_text())
        if bank.get('schema') != 2 or not bank.get('source_full_start_ram_star'):
            raise ValueError('Backward scheduling requires a verified demonstration bank.')
        self.entries = bank['entries']
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if (self.state['bank_sha256'] != expected_hash or type(self.state['frontier']) is not int
                    or not 0 <= self.state['frontier'] < len(self.entries)):
                raise ValueError('Resumed curriculum schedule identity mismatch.')
        else:
            self.state = {'schema': 1, 'bank_sha256': expected_hash,
                          'frontier': len(self.entries) - 1, 'recent': [], 'events': [],
                          'assisted_attempts': 0, 'assisted_ram_stars': 0}
            self.save()

    def save(self):
        if not atomic_json(self.path, self.state):
            raise OSError('Could not publish the backward start frontier.')

    def record(self, anchor, ram_star, real_decisions):
        if anchor not in {e['file'] for e in self.entries}:
            raise ValueError('An assisted attempt lacks a verified anchor.')
        self.state['assisted_attempts'] += 1
        self.state['assisted_ram_stars'] += int(bool(ram_star))
        current = self.entries[self.state['frontier']]['file']
        if anchor == current:
            self.state['recent'] = (self.state['recent'] + [bool(ram_star)])[-10:]
            if len(self.state['recent']) == 10 and sum(self.state['recent']) >= 8 and self.state['frontier'] > 0:
                previous = self.state['frontier']
                self.state['frontier'] -= 1
                self.state['events'].append({'real_decisions': real_decisions,
                    'from_frontier': previous, 'to_frontier': self.state['frontier'],
                    'ram_stars_last10': sum(self.state['recent'])})
                self.state['recent'] = []
        self.save()
