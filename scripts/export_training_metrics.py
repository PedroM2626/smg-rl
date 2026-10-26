"""Backfill and follow real episode reward/distance without restarting learners."""

import argparse

from smg_rl.training_metrics import export_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--interval', type=float, default=15.)
    parser.add_argument('--training-state', help='Explicit finite training supervisor state; defaults to the original IQN trial.')
    args = parser.parse_args()
    kwargs = {'training_state': args.training_state} if args.training_state else {}
    export_metrics(args.watch, args.max_seconds, args.interval, **kwargs)
