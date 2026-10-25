"""Reward invariants for an isolated schema-2 route, not learned completions."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.replay import replay_signature
from smg_rl.route import RouteProgress


def sample(position, **kwargs):
    return SimpleNamespace(position=position, grounded=True, health=3,
                           dead=False, complete=False, **kwargs)


def specification(schema):
    spec = json.loads(Path('config/routes/smg1-initial-exit.json').read_text())
    return {**spec, 'schema': schema}


def test_endpoint_candidate_removes_stationary_segment_credit_preserving_history():
    point = [146, 45, -295]  # 70% along the first segment, inside its endpoint radius.
    results = []
    for schema in (1, 2):
        route = RouteProgress(specification(schema))
        route.reset(sample(route.points[0]))
        first = route.update(sample(point))
        second = route.update(sample(point))
        results.append((first, second, route.segment))
    assert results[0][0] == pytest.approx(results[1][0])
    assert results[0][1] > 0.1  # Reproduces historical phantom credit.
    assert results[0][2] == 1
    assert results[1][1:] == (0, 0)


def test_candidate_resolves_crossing_once_and_requires_endpoint_for_exit():
    route = RouteProgress(specification(2))
    route.reset(sample(route.points[0]))
    total = 0
    for point in route.points[1:-1]:
        total += route.update(sample(point))
        assert route.update(sample(point)) == 0
    near = route.points[-1] + [0, 0, 100]
    total += route.update(sample(near))
    assert route.update(sample(near)) == 0 and not route.exited
    total += route.update(sample(route.points[-1]))
    assert not route.exited
    assert route.update(sample(route.points[-1])) == 0 and route.exited
    assert total == pytest.approx(1)


def test_dense_candidate_trace_is_bounded_and_does_not_pay_backtracking():
    route = RouteProgress(specification(2))
    route.reset(sample(route.points[0]))
    total = 0
    for start, end in zip(route.points[:-1], route.points[1:]):
        for t in np.linspace(0, 1, 41):
            point = start + t * (end - start)
            total += route.update(sample(point))
            assert route.update(sample(point)) == 0
    assert total == pytest.approx(1)
    assert route.update(sample(route.points[-2])) == 0
    route.reset(sample(route.points[-1]))
    assert route.update(sample(route.points[-1])) == 0 and not route.exited


def test_candidate_rejects_ground_contact_damage_and_corner_shortcuts():
    route = RouteProgress(specification(2))
    route.reset(sample(route.points[0]))
    for attr, value in [('grounded', False), ('health', 2), ('dead', True), ('complete', True)]:
        point = sample(route.points[1])
        setattr(point, attr, value)
        assert route.update(point) == 0
    assert route.update(sample(route.points[-1])) == 0
    with pytest.raises(ValueError, match='schema'):
        RouteProgress(specification(3))


def test_candidate_keeps_historical_file_and_requires_a_distinct_replay_signature():
    historical = Path('config/routes/smg1-initial-exit.json')
    candidate = Path('config/routes/smg1-initial-exit-v2-candidate.json')
    config = load_config('config/motor8-smg1.yaml')
    original = replay_signature(config)
    assert digest(historical) == config['route_reward']['spec_sha256']
    assert Path(config['paths']['route']).resolve() == historical.resolve()
    config['route_reward'] = {**config['route_reward'], 'spec_sha256': digest(candidate)}
    config['paths']['route'] = str(candidate)
    config['rewards']['revision'] = 'endpoint-candidate-offline-only'
    assert replay_signature(config) != original
