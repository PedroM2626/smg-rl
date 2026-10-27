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
    args = parser.parse_args()
    train_genetic(load_config(args.config), args.source, args.seconds, deadline_utc=args.deadline_utc,
                  resume_run=args.resume_run, confirmation_seconds=args.confirmation_seconds, policy_mode=args.policy_mode)
