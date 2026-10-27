"""Shared read-only physical trajectory checks for genetic experiment reports."""

import csv
from pathlib import Path

import numpy as np

from smg_rl.genetic import GroundedAdvance


def audited_trajectory(record, bounds, max_step):
    if not record.get('trajectory'):
        if record['decisions']:
            raise ValueError('A nonempty genetic trial lacks a trajectory.')
        return [], []
    with Path(record['trajectory']).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) - 1 != record['decisions']:
        raise ValueError('Genetic trial/CSV decision counts differ.')
    score = GroundedAdvance(bounds['start'], bounds['goal'], max_step)
    advances = [0.]
    for row in rows[1:]:
        score.update({'position': [float(row[k]) for k in ('x', 'y', 'z')],
                      'grounded': row['grounded'] == 'True', 'dead': row['dead'] == 'True'})
        advances.append(100 * score.best / score.length)
    if score.invalid or not np.isclose(score.best, record['fitness_world_units'], atol=1e-6, rtol=0):
        raise ValueError('Recomputed grounded trajectory fitness differs from the trial ledger.')
    if record['ram_star'] != (rows[-1]['complete'] == 'True'):
        raise ValueError('Genetic success differs from native completion CSV.')
    return rows, advances


def counts(trials):
    """Count actual ledger entries; inherited fitness is deliberately not an entry."""
    if any('inherited_from_run' in t for t in trials):
        raise ValueError('Inherited fitness metadata must not be counted as new physical trials.')
    return {'real_decisions': sum(t['decisions'] for t in trials),
            'completed_trials': sum(bool(t['fitness_eligible']) for t in trials),
            'partial_trials': sum(not t['fitness_eligible'] for t in trials),
            'partial_decisions': sum(t['decisions'] for t in trials if not t['fitness_eligible']),
            'full_start_stars': sum(bool(t['ram_star']) and t['fitness_eligible'] for t in trials)}
