"""Report canonical global goal projection separately from the mapped initial opening."""

import argparse
import json

from smg_rl.iqn_progress import report_iqn_progress

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(report_iqn_progress(args.state, args.output), indent=2))
