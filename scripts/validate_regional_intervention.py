"""Audit recorded regional fixtures and restore every accepted start in Dolphin."""

import argparse
import json
from pathlib import Path

import numpy as np

from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.env import GalaxyEnv
from smg_rl.memory import Telemetry
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.rewards import RegionalPotential
from smg_rl.route import RouteProgress
from smg_rl.train import new_run


def audit(config, preparation):
    report = json.loads(Path(preparation).read_text())
    spec = json.loads(Path(config['paths']['route']).read_text())
    settings = config['regional_reward']
    results = []
    for mission in report['missions']:
        traces = json.loads((Path(preparation).parent / f"{mission['variation']}-traces.json").read_text())
        for name, trace in traces.items():
            if '-idle' in name or name == 'near-star-detector-only':
                continue
            route = RouteProgress(spec)
            route.reset(Telemetry(**trace[0]['telemetry']))
            potential = RegionalPotential(settings['scale'], settings['gamma_per_frame'])
            potential.reset(0.)
            discount, total, increments = 1., 0., []
            for sample in trace[1:]:
                telemetry = Telemetry(**sample['telemetry'])
                before = route.best
                route.update(telemetry)
                if telemetry.grounded is not True:
                    assert route.best == before
                value = min(1., route.best * route.total_length / settings['normalization_distance'])
                amount = potential.update(value, sample['frames'], telemetry.dead or telemetry.complete)
                total += discount * amount
                discount *= settings['gamma_per_frame'] ** sample['frames']
                increments.append(amount)
            # Algebraic closure is an offline failure counterfactual, not an extra game transition.
            closure = potential.update(value, 4, terminal=True)
            total += discount * closure
            assert abs(total) < 1e-10
            before = route.best
            for _ in range(10):
                assert route.update(telemetry) == 0.
                assert route.best == before
            results.append({'variation': mission['variation'], 'trace': name,
                            'confirmed_opening_fraction': route.best,
                            'confirmed_distance': route.best * route.total_length,
                            'closed_discounted_shaping': total,
                            'partial_undiscounted_shaping': sum(increments),
                            'ram_death': telemetry.dead})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', choices=['smg1', 'smg2'], required=True)
    parser.add_argument('--preparation', required=True)
    args = parser.parse_args()
    config = load_config(f'config/regional-{args.game}-12.yaml')
    config['replay']['enabled'] = False
    config['paths']['runs'] = 'runs/regional-intervention'
    run = new_run(config, f'validate-{args.game}')
    report = {'state': 'running', 'run': str(run), 'game': args.game,
              'trace_audit': audit(config, args.preparation), 'restorations': [],
              'scope': 'Engineering validation only. No optimizer or training replay.',
              'maximum_idle_frames_per_restoration': 60, 'frames': 0}
    atomic_json(run / 'result.json', report)
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    backend = NativeDolphin(config)
    env = None
    try:
        env = GalaxyEnv(config, backend, run / 'worker-00', memory=backend.memory, intrinsic=False)
        bank = env.curriculum_bank
        for entry in bank.entries:
            index = next(i for i, v in enumerate(config['variations']) if v['scenario'] == entry['scenario'])
            seed = next(s for s in range(10000)
                        if (chosen := bank.choose(np.random.default_rng(s), entry['scenario']))
                        and chosen['file'] == entry['file'])
            env.reset(seed=seed, options={'variation': index})
            assert env.start_kind == 'curriculum' and env.start_state_id == entry['file']
            assert env.rows[0]['reward'] == 0
            start_potential = env.potential.value
            initial_position = np.asarray(env.telemetry.position, dtype=float).copy()
            for _ in range(5):
                _, _, terminal, truncated, _ = env.step(0)
                report['frames'] += 12
                assert not terminal and not truncated
                assert env.telemetry.grounded is True and env.telemetry.health == entry['health']
            displacement = np.asarray(env.telemetry.position, dtype=float) - initial_position
            report['restorations'].append({'file': entry['file'], 'seed': seed, 'region': entry['region'],
                                           'passed': True, 'initial_potential': start_potential,
                                           'reset_reward': 0., 'final': env.telemetry.to_dict(),
                                           'idle_displacement_xyz': float(np.linalg.norm(displacement)),
                                           'idle_displacement_xz': float(np.linalg.norm(displacement[[0, 2]])),
                                           'within_frontier_250_unit_drift_screen': bool(np.linalg.norm(displacement) <= 250)})
            env.reset(seed=seed, options={'variation': index, 'full_start': True})
            assert env.start_kind == 'full_start' and env.route_start == 0
            atomic_json(run / 'result.json', report)
        report['state'] = 'completed'
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        if env:
            env.close()
        else:
            backend.close()
        atomic_json(run / 'result.json', report)
    print(json.dumps({'state': report['state'], 'run': str(run), 'frames': report['frames'],
                      'restorations': len(report['restorations'])}), flush=True)


if __name__ == '__main__':
    main()
