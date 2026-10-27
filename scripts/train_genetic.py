"""Run the bounded genetic action-head pilot with exclusive six-emulator ownership."""

import argparse

from smg_rl.config import load_config
from smg_rl.genetic import train_genetic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--seconds', type=float, default=3600)
    args = parser.parse_args()
    train_genetic(load_config(args.config), args.source, args.seconds)
