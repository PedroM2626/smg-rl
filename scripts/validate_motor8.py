"""Bounded live tests of eight exclusive controls in all three supported missions."""

import json
from pathlib import Path

import numpy as np

from smg_rl.actions import action_set
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.env import GalaxyEnv
from smg_rl.native import NativeDolphin
from smg_rl.train import new_run


def main():
    result = {'scope':'Scripted control fixtures; no learning, replay or trained mission-success accounting',
              'profile':'motor8-v1', 'physical_decisions':0, 'scenarios':[]}
    for game in ('smg1','smg2'):
        config = load_config(f'config/motor8-{game}.yaml')
        root = new_run(config,'background-validation-motor8')
        config['paths']['user'] = str(root/'profile'/'template')
        config['replay']['enabled'] = False
        backend = NativeDolphin(config)
        env = GalaxyEnv(config,backend,root/'worker-00',backend.memory,intrinsic=False)
        try:
            for variation in range(len(config['variations'])):
                row = {'game':game, 'variation':config['variations'][variation]['name'],
                       'single_action_traces':[], 'exit_trials':[]}
                for action, names in enumerate(action_set(config)):
                    env.reset(seed=11,options={'variation':variation})
                    _,_,_,_,info = env.step(action)
                    result['physical_decisions'] += 1
                    assert info['advanced_frames'] == info['scene_frames_advanced'] == 12
                    assert len(names) <= 1
                    row['single_action_traces'].append({'controls':names,'telemetry':env.telemetry.to_dict()})
                env.reset(seed=11,options={'variation':variation})
                for _ in range(3):
                    env.step(action_set(config).index(('w',)))
                    result['physical_decisions'] += 1
                before = env.telemetry.to_dict()
                env.step(action_set(config).index(('jump',)))
                result['physical_decisions'] += 1
                after = env.telemetry.to_dict()
                row['jump_after_walk'] = {'before':before,'after':after,
                    'horizontal_displacement':float(np.linalg.norm(np.asarray(after['position'])[[0,2]]-np.asarray(before['position'])[[0,2]])),
                    'height_gain':after['position'][1]-before['position'][1]}
                if game == 'smg1':
                    schedules = [[('d',)]*n+[('w',)]*16 for n in (2,3,4)]
                    schedules += [[('d',),('w',)]*4+[('w',)]*12]
                    for schedule in schedules:
                        env.reset(seed=11,options={'variation':variation})
                        trace = []
                        for names in schedule:
                            _,_,terminated,truncated,info = env.step(action_set(config).index(names))
                            result['physical_decisions'] += 1
                            trace.append({'controls':names,'telemetry':env.telemetry.to_dict(),
                                          'prefix':info['route_prefix_progress'], 'exited':info['route_prefix_exited']})
                            if terminated or truncated:
                                break
                        passed = info['route_prefix_exited'] and not info['dead'] and info['health'] == 3
                        row['exit_trials'].append({'schedule':schedule,'passed':passed,'trace':trace})
                        if passed:
                            break
                result['scenarios'].append(row)
        finally:
            env.close()
    result['checks'] = {'three_missions':len(result['scenarios']) == 3,
        'all_jump_height_gain':all(s['jump_after_walk']['height_gain'] > 20 for s in result['scenarios']),
        'all_jump_inertia':all(s['jump_after_walk']['horizontal_displacement'] > 1 for s in result['scenarios']),
        'smg1_safe_exclusive_exit':any(t['passed'] for t in result['scenarios'][0]['exit_trials'])}
    result['state'] = 'passed' if all(result['checks'].values()) else 'failed'
    output = Path('docs/assets/performance/motor8-control-validation-20261025.json')
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output,result)
    print(json.dumps({'state':result['state'],'decisions':result['physical_decisions'],'checks':result['checks']}))
    if result['state'] != 'passed':
        raise RuntimeError('The exclusive-control fixture needs review.')


if __name__ == '__main__':
    main()
