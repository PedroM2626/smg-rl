import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    spec = importlib.util.spec_from_file_location('sweet_search', Path('scripts/solve_sweet_frontier.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_search_is_seeded_bounded_and_exclusive():
    m = module()
    first = m.action_schedule(np.random.default_rng(10))
    assert first == m.action_schedule(np.random.default_rng(10))
    assert len(first) == m.HORIZON and all(0 <= value < 8 for value in first)
    assert first != m.action_schedule(np.random.default_rng(11))


def test_cells_are_serializable_and_reject_nonfinite_positions():
    m = module()
    assert m.cell([450., -50., -2328.]) == (1, -1, -10)
    assert all(type(value) is int for value in m.cell([450., -50., -2328.]))
    with pytest.raises(ValueError, match='nonfinite'):
        m.cell([0., float('nan'), 0.])
