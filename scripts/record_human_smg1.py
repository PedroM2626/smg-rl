"""Start the interactive Sweet Sweet human recorder on Windows."""

import argparse

from smg_rl.human_recorder import record_human

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-minutes', type=int, default=120)
    parser.add_argument('--max-attempts', type=int, default=100)
    args = parser.parse_args()
    print(record_human(args.max_minutes, args.max_attempts), flush=True)
