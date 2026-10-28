"""Run one explicitly bounded robust genetic arm or held-out confirmation."""

import argparse

from smg_rl.robust_genetic import run_arm, run_confirmation, run_sampling_smoke

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['smoke', 'B', 'C', 'confirmation'])
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--calibration-review')
    parser.add_argument('--population-seed', type=int, default=20261027)
    parser.add_argument('--arm-b')
    parser.add_argument('--arm-c')
    parser.add_argument('--arm-b-review')
    parser.add_argument('--arm-c-review')
    args = parser.parse_args()
    if args.stage == 'smoke':
        print(run_sampling_smoke(calibration_review=args.calibration_review))
    elif args.stage == 'confirmation':
        if not args.arm_b or not args.arm_c:
            parser.error('confirmation requires --arm-b and --arm-c')
        print(run_confirmation(args.arm_b, args.arm_c,
                               arm_b_review=args.arm_b_review, arm_c_review=args.arm_c_review))
    else:
        if args.sigma is None:
            parser.error('training arms require --sigma')
        print(run_arm(args.stage, args.sigma, population_seed=args.population_seed,
                      calibration_review=args.calibration_review,
                      prior_arm_review=args.arm_b_review if args.stage == 'C' else None))
