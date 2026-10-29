"""Create an immutable, explicitly audited worker-layout migration checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_training import prepare_dqn


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--config', required=True, help='Original source worker-count profile.')
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, target = Path(args.checkpoint), Path(args.output)
    if target.exists():
        raise FileExistsError('Never overwrite or repeat a migration checkpoint.')
    config = load_config(args.config)
    agent, original = prepare_dqn(config, source)
    if args.workers > original.workers:
        expanded = original.expanded_workers(args.workers)
    elif args.workers < original.workers:
        expanded = original.reduced_workers(args.workers)
    else:
        raise ValueError('The target worker count must differ.')
    for key in original.arrays:
        np.testing.assert_array_equal(original.arrays[key], expanded.arrays[key])
    for start in range(0, len(original.frame_ids), 4096):
        slots = np.arange(start, min(start + 4096, len(original.frame_ids)))
        slots = slots[original.frame_ids[slots] >= 0]
        targets = original.frame_ids[slots] % expanded.frame_capacity
        np.testing.assert_array_equal(original.frames[slots], expanded.frames[targets])
        np.testing.assert_array_equal(original.previous[slots], expanded.previous[targets])
    agent.save(target, expanded)
    before = torch.load(source / 'agent.pt', map_location='cpu', weights_only=True)
    after = torch.load(target / 'agent.pt', map_location='cpu', weights_only=True)
    if not same(before, after):
        raise ValueError('Worker migration changed learner/optimizer/RNG/counter state.')
    report = {'scope': 'Collector layout only; no learning, replay relabeling or counter reset.',
              'target_agent_and_replay_sha256': {'agent': digest(target / 'agent.pt'), 'replay': digest(target / 'replay.npz')},
              'source': str(source.resolve()), 'target': str(target.resolve()),
              'source_workers': original.workers, 'target_workers': expanded.workers,
              'real_decisions': agent.real_decisions, 'optimizer_updates': agent.updates,
              'completed_attempts': sum(v['completed'] for v in agent.attempts.values()),
              'retained_transitions': expanded.size, 'frame_capacity_before': original.frame_capacity,
              'frame_capacity_after': expanded.frame_capacity, 'all_retained_frames_links_and_transitions_equal': True,
              'learner_optimizer_rng_and_counters_equal': True,
              'source_agent_sha256': digest(source / 'agent.pt'), 'target_agent_sha256': digest(target / 'agent.pt')}
    atomic_json(target / 'worker-migration.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
