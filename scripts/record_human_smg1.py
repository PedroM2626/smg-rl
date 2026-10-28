"""Start an interactive human recorder for a configured Mario Galaxy mission."""

import argparse

from smg_rl.human_recorder import record_human

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-minutes', type=int, default=120)
    parser.add_argument('--max-attempts', type=int, default=100)
    parser.add_argument('--game', choices=('smg1', 'smg2'), default='smg1')
    parser.add_argument('--variation', choices=('sweet_sweet', 'spinning', 'double_time'))
    args = parser.parse_args()
    print(record_human(args.max_minutes, args.max_attempts, args.game, args.variation), flush=True)
