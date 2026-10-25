"""Validate native motor16 pulse timing and the scripted exit with actual RAM."""

import json

from smg_rl.actions import action_set
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.env import GalaxyEnv
from smg_rl.native import NativeDolphin
from smg_rl.train import new_run


def main():
    config = load_config('config/motor16-smg1-route.yaml')
    root = new_run(config, 'background-validation-motor16')
    config['paths']['user'] = str(root / 'profile' / 'template')
    config['replay']['enabled'] = False
    backend = NativeDolphin(config)
    env = GalaxyEnv(config, backend, root / 'worker-00', backend.memory, intrinsic=False)
    result = {'run':str(root), 'scope':'Scripted input fixture, never learned navigation or mission success', 'trace':[]}
    try:
        env.reset(seed=11)
        for names in [('jump',)]*3 + [()]*10:
            _, _, _, _, info = env.step(action_set(config).index(names))
            result['trace'].append(info)
        result['pulse_frames_exact'] = all(r['scene_frames_advanced'] == 12 and r['advanced_frames'] == 12 for r in result['trace'])
        result['pulse_released_between_decisions'] = backend.bridge.get(7) == 0
        env.reset(seed=11)
        for names in [('d','w')]*5 + [('w',)]*11:
            _, _, terminated, truncated, info = env.step(action_set(config).index(names))
            result['trace'].append(info)
            if terminated or truncated:
                break
        result['safe_exit'] = info['route_prefix_exited'] and not info['dead'] and info['health'] == 3
        result['no_mission_completion'] = not info['complete']
    finally:
        env.close()
    result['state'] = 'passed' if all(result[k] for k in ('pulse_frames_exact','pulse_released_between_decisions','safe_exit','no_mission_completion')) else 'failed'
    atomic_json(root / 'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'trace'}))
    if result['state'] != 'passed':
        raise RuntimeError('Motor validation failed.')


if __name__ == '__main__':
    main()
