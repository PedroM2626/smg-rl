"""Run the bounded genetic action-head pilot with exclusive six-emulator ownership."""

import argparse

from smg_rl.config import load_config
from smg_rl.genetic import train_genetic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--resume-run')
    parser.add_argument('--confirmation-seconds', type=float, default=300)
    parser.add_argument('--policy-mode', choices=['head', 'full_visual'], default='head')
    parser.add_argument('--fitness-mode', choices=['goal-projection-xz-v1', 'goal-distance-xyz-v1'],
                        default='goal-projection-xz-v1')
    parser.add_argument('--initial-head', help='Verified earlier head champion; fitness is evaluated again.')
    parser.add_argument('--population-seed', type=int, default=142,
                        help='Seed for a fresh population; does not change the restored Wii state.')
    args = parser.parse_args()
    train_genetic(load_config(args.config), args.source, args.seconds, deadline_utc=args.deadline_utc,
                  resume_run=args.resume_run, confirmation_seconds=args.confirmation_seconds, policy_mode=args.policy_mode,
                  fitness_mode=args.fitness_mode, initial_head=args.initial_head,
                  population_seed=args.population_seed)
