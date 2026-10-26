"""Backfill and follow real episode reward/distance without restarting learners."""

import argparse

from smg_rl.training_metrics import export_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--interval', type=float, default=15.)
    args = parser.parse_args()
    export_metrics(args.watch, args.max_seconds, args.interval)
