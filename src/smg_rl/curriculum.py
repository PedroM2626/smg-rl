"""Immutable curated intermediate starts with explicit assisted-start provenance."""

import json
import re
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json


class CurriculumBank:
    def __init__(self, path, expected_hash, game_id, emulator_hash, probability, progress_path=None,
                 anchor_diagnostic=False):
        self.path = Path(path)
        # Mixed training keeps full starts; only a labeled anchor diagnostic may start every attempt assisted.
        if not 0 <= probability <= (1. if anchor_diagnostic else .5) or digest(self.path) != expected_hash:
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
    """The learner alone advances a signed frontier after eight of ten RAM stars.

    The optional hybrid rule instead needs consecutive frozen greedy probe stars from the
    frontier plus a minimum training-episode star rate over a longer window.
    """

    def __init__(self, bank_path, expected_hash, progress_path, rule=None):
        self.path = Path(progress_path)
        self.rule = validate_frontier_rule(rule)
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
        if self.rule:
            self.state.setdefault('probes', [])

    def save(self):
        if not atomic_json(self.path, self.state):
            raise OSError('Could not publish the backward start frontier.')

    def record(self, anchor, ram_star, real_decisions):
        if anchor not in {e['file'] for e in self.entries}:
            raise ValueError('An assisted attempt lacks a verified anchor.')
        self.state['assisted_attempts'] += 1
        self.state['assisted_ram_stars'] += int(bool(ram_star))
        if anchor == self.current():
            window = self.rule['robust_window'] if self.rule else 10
            self.state['recent'] = (self.state['recent'] + [bool(ram_star)])[-window:]
            self._advance(real_decisions)
        self.save()

    def current(self):
        return self.entries[self.state['frontier']]['file']

    def record_probe(self, anchor, ram_star, real_decisions):
        """Record a frozen greedy frontier probe; it never counts as a training attempt."""
        if not self.rule:
            raise ValueError('Greedy frontier probes require the hybrid frontier rule.')
        if anchor not in {e['file'] for e in self.entries}:
            raise ValueError('A frontier probe lacks a verified anchor.')
        if anchor == self.current():
            self.state['probes'] = (self.state['probes'] + [bool(ram_star)])[-self.rule['probe_streak']:]
            self._advance(real_decisions)
        self.save()

    def _advance(self, real_decisions):
        recent = self.state['recent']
        if self.rule:
            streak = self.rule['probe_streak']
            ready = (len(recent) == self.rule['robust_window']
                     and sum(recent) >= self.rule['robust_min'] * len(recent)
                     and len(self.state['probes']) == streak and all(self.state['probes']))
        else:
            ready = len(recent) == 10 and sum(recent) >= 8
        if not ready or self.state['frontier'] == 0:
            return
        previous = self.state['frontier']
        self.state['frontier'] -= 1
        event = {'real_decisions': real_decisions, 'from_frontier': previous,
                 'to_frontier': self.state['frontier']}
        if self.rule:
            event.update(rule=self.rule['mode'], ram_stars_recent=sum(recent), recent_window=len(recent),
                         greedy_probe_streak=len(self.state['probes']))
            self.state['probes'] = []
        else:
            event['ram_stars_last10'] = sum(recent)
        self.state['events'].append(event)
        self.state['recent'] = []


def validate_frontier_rule(rule):
    """None keeps eight of ten training stars; 'greedy_hybrid' adds frozen greedy probes."""
    if rule is None:
        return None
    keys = {'mode', 'probe_every', 'probe_streak', 'robust_window', 'robust_min'}
    if (set(rule) != keys or rule['mode'] != 'greedy_hybrid'
            or any(type(rule[k]) is not int for k in ('probe_every', 'probe_streak', 'robust_window'))
            or not 2 <= rule['probe_every'] <= 100 or not 1 <= rule['probe_streak'] <= 10
            or not 10 <= rule['robust_window'] <= 100
            or type(rule['robust_min']) is not float or not .05 <= rule['robust_min'] <= .8):
        raise ValueError('Invalid hybrid frontier rule.')
    return dict(rule)


def canonical_start(config, options, rng):
    """True when a reset must use the canonical start instead of a curriculum anchor.

    `start_mix.full_start_fraction` is a collection-only setting kept outside the signed
    `curriculum` block, so checkpoints resume across changes to it. No random number is drawn
    unless the fraction is positive, which keeps earlier runs' random streams unchanged."""
    if options.get('full_start', False):
        return True
    fraction = config.get('start_mix', {}).get('full_start_fraction', 0.)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError('The canonical-start fraction must be a number between zero and one.')
    return bool(fraction) and bool(rng.random() < fraction)
