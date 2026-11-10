import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('prepare_anchor_bank', Path(__file__).resolve().parents[1] / 'scripts/prepare_anchor_bank.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_anchors_are_spaced_grounded_and_far_from_the_goal():
    points = np.stack([np.linspace(0, 30000, 600), np.zeros(600), np.zeros(600)], axis=1)
    grounded = np.ones(600, dtype=bool)
    grounded[100:140] = False
    chosen = module.choose_samples(points, grounded, np.array([30000., 0., 0.]), 5)
    assert chosen == sorted(chosen) and len(chosen) == 5
    assert all(grounded[i:i + 30].all() for i in chosen)
    assert all(np.linalg.norm(points[i] - [30000, 0, 0]) >= 2000 for i in chosen)
    assert np.diff([points[i][0] for i in chosen]).min() > 3000


def test_nearest_waypoint_indices_follow_the_route_order():
    route = {'waypoints': [[x, 0, 0] for x in range(0, 1000, 100)]}
    assert module.nearest_waypoints(route, np.array([[0, 0, 0], [420, 5, 0], [905, 0, 0]])) == [0, 4, 9]
