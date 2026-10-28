import numpy as np
import pytest

from smg_rl.genetic import sample_genetic_action
from smg_rl.robust_genetic import (
    aggregate_draws,
    calibration_decision,
    draw_seed,
    verified_stage_review,
)


def test_argmax_is_identical_and_epsilon_draws_are_reproducible():
    rng = np.random.default_rng(10)
    state = rng.bit_generator.state
    assert sample_genetic_action(3, 'argmax', 0, rng, 8) == (3, False)
    assert rng.bit_generator.state == state
    rng_a, rng_b = np.random.default_rng(11), np.random.default_rng(11)
    a = [sample_genetic_action(3, 'epsilon', .05, rng_a, 8) for _ in range(1000)]
    b = [sample_genetic_action(3, 'epsilon', .05, rng_b, 8) for _ in range(1000)]
    assert a == b
    assert 20 < sum(sampled for _, sampled in a) < 90
    with pytest.raises(ValueError):
        sample_genetic_action(0, 'softmax', .05, rng, 8)


def test_aggregation_rejects_missing_or_mixed_draws():
    def entry(i, draw, value, digest='hash'):
        return ({'individual': i, 'draw': draw, 'fitness_eligible': True,
                 'genome_sha256': digest, 'fitness_world_units': value}, None)
    records = [entry(i, d, i + d) for i in range(2) for d in range(3)]
    assert aggregate_draws(records, 2, 3).tolist() == [1, 2]
    with pytest.raises(ValueError):
        aggregate_draws(records[:-1], 2, 3)
    with pytest.raises(ValueError):
        aggregate_draws(records[:-1] + [entry(1, 2, 3, 'different')], 2, 3)
    assert draw_seed(0, 0) == draw_seed(0, 0)
    assert draw_seed(0, 0) != draw_seed(0, 1)


def test_calibration_chooses_largest_valid_scale_and_counts_stars():
    base = {'action_prefix60_sha256': 'base'}
    rows = []
    for sigma in (.02, .05, .10):
        for i in range(16):
            rows.append({'sigma': sigma, 'fitness_eligible': True,
                         'fitness_world_units': 100, 'decisions': 70 if i < (8 if sigma < .1 else 7) else 20,
                         'ram_star': sigma == .1 and i == 7,
                         'action_prefix60_sha256': 'changed' if i < 4 else 'base'})
    assert calibration_decision(rows, base)['selected_sigma'] == .1
    rows[-1]['fitness_eligible'] = False
    with pytest.raises(ValueError, match='16 complete'):
        calibration_decision(rows, base)


def test_blocked_or_stale_calibration_cannot_authorize_a_training_arm(tmp_path):
    import json

    from smg_rl.checkpoints import digest

    run = tmp_path / 'calibration'
    run.mkdir()
    (run / 'summary.json').write_text('{}')
    (run / 'trials.jsonl').write_text('')
    review = tmp_path / 'review.json'
    data = {'audit_passed': True, 'stage': 'calibration', 'state': 'blocked', 'run': str(run),
            'summary_sha256': digest(run / 'summary.json'), 'trials_sha256': digest(run / 'trials.jsonl')}
    review.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='blocked or stale'):
        verified_stage_review(review, 'calibration')
    data['state'] = 'completed'
    review.write_text(json.dumps(data))
    assert verified_stage_review(review, 'calibration')['state'] == 'completed'
    (run / 'trials.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='blocked or stale'):
        verified_stage_review(review, 'calibration')
