import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_search():
    spec = importlib.util.spec_from_file_location('chocolate_search', Path('scripts/chocolate_frontier_search.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_seeded_exclusive_schedules():
    module = load_search()
    first = module.schedule(np.random.default_rng(9000))
    assert first == module.schedule(np.random.default_rng(9000))
    assert len(first) == 48 and set(first).issubset(set(range(8)))
    assert first != module.schedule(np.random.default_rng(9001))


def test_candidate_needs_live_grounded_undamaged_state_beyond_opening():
    module = load_search()
    sample = SimpleNamespace(position=(500., -50., -2110.), dead=False, complete=False, grounded=True, health=3)
    endpoint = (500., -50., -1950.)
    assert module.interesting(sample, endpoint)
    for field, value in [('grounded', False), ('dead', True), ('complete', True), ('health', 2)]:
        invalid = SimpleNamespace(**vars(sample))
        setattr(invalid, field, value)
        assert not module.interesting(invalid, endpoint)
    sample.position = (500., -50., -2000.)
    assert not module.interesting(sample, endpoint)


def test_validation_demands_exact_time_position_health_and_ground():
    spec = importlib.util.spec_from_file_location('chocolate_validation', Path('scripts/validate_chocolate_frontier.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sample = SimpleNamespace(position=(500., -50., -2100.), dead=False, complete=False,
                             grounded=True, health=3, scene_frame=42)
    assert module.matching(sample, sample.position, 42)
    assert not module.matching(sample, sample.position, 43)
    assert not module.matching(sample, (500., -50., -2100.1), 42)
    sample.grounded = False
    assert not module.matching(sample, sample.position, 42)
