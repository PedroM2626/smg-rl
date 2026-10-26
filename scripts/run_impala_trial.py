"""Start health, supervise reviewed bounded blocks, or inspect the six-hour trial."""

import argparse

from smg_rl.dqn_trial import snapshot, start_health, supervise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['health', 'supervise', 'snapshot'])
    args = parser.parse_args()
    {'health': start_health, 'supervise': supervise, 'snapshot': snapshot}[args.mode]()
