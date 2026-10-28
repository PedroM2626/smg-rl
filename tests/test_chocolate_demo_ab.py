import importlib.util
import sys
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest


def test_demonstration_replay_keeps_full_start_cut_nonterminal(tmp_path):
    spec = importlib.util.spec_from_file_location('chocolate_demo', Path('scripts/run_chocolate_demo_ab.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    episodes = []
    for index in range(2):
        path = tmp_path / f'demo-{index}.npz'
        frames = np.stack([np.full((40, 40), value, np.uint8) for value in range(5)])
        np.savez_compressed(path, frames=frames, actions=np.array([3, 3, 5, 3]), rewards=np.zeros(4, np.float32))
        episodes.append({'purpose': 'training', 'file': str(path), 'sha256': digest(path),
                         'outcome': 'nonterminal_collector_cut', 'decisions': 4})
    cfg = {'environment': {'image_size': 40}, 'dqn': {'frame_stack': 4, 'n_step': 3, 'gamma': 1., 'alpha': .5}}
    replay = module.demo_replay(cfg, {'episodes': episodes}, seed=42)
    assert replay.emitted == replay.size == 8 and not any(replay.pending)
    assert not replay.arrays['terminal'][:8].any() and not replay.arrays['assisted'][:8].any()
    assert replay.arrays['horizon'][:4].tolist() == [3, 3, 2, 1]
    assert replay.arrays['discount'][:8].tolist() == [1.] * 8


def test_gate_requires_projection_reachable_region_and_no_star_regression():
    sys.path.insert(0, str(Path('scripts').resolve()))
    try:
        spec = importlib.util.spec_from_file_location('chocolate_audit', Path('scripts/audit_chocolate_demo_ab.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    assert module.pair_gate(.5, 1, 0, 0)
    assert not module.pair_gate(.499, 1, 0, 0)
    assert not module.pair_gate(.6, 0, 0, 0)
    assert not module.pair_gate(.6, 1, 1, 0)
