"""Capture real visual transitions of one reproducible, partial full-start path."""

import json
import os
from pathlib import Path

import numpy as np

from smg_rl.actions import action_name
from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.env import GalaxyEnv
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.replay import replay_signature
from smg_rl.train import new_run

os.chdir(Path(__file__).resolve().parents[1])
VALIDATION = Path('runs/chocolate-frontier-validation-20261028/validate-20261028-093109-247390/result.json')
INITIAL = [1] * 3 + [3] * 16 + [1] * 2 + [3] * 8 + [0] * 2


def main():
    validation = json.loads(VALIDATION.read_text(encoding='utf-8'))
    if validation['state'] != 'completed':
        raise ValueError('Demonstration needs a completed physical validation.')
    candidate = validation['candidates'][0]
    if (not candidate['full_start_replay']['candidate_exact']
            or not all(r['exact_candidate'] for r in candidate['assisted_replays'])
            or not candidate['timing_perturbation']['completed']):
        raise ValueError('The selected full-start path has not passed its recorded replay checks.')
    config = load_config('config/smg1-six-two-hour.yaml')
    config['model_free_run'] = 'evaluation'
    config['replay']['enabled'] = False
    config['paths']['runs'] = str(Path(config['root']) / 'runs/chocolate-demonstrations-20261028')
    run = new_run(config, 'capture')
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    result = {'state': 'running', 'run': str(run), 'validation': str(VALIDATION),
              'validation_sha256': digest(VALIDATION), 'signature': replay_signature(config),
              'scope': 'Scripted full-start partial demonstrations; no training, no star claim.',
              'source_state_stable': False, 'episodes': [], 'physical_decisions': 0}
    atomic_json(run / 'result.json', result)
    print(json.dumps({'run': str(run)}), flush=True)
    backend = NativeDolphin(config)
    env = None
    try:
        env = GalaxyEnv(config, backend, run / 'worker-00', memory=backend.memory, intrinsic=False, variation=0)
        for delay, purpose in [(0, 'training'), (2, 'training'), (1, 'held_out_timing_check')]:
            observation, _ = env.reset(seed=11, options={'variation': 0, 'full_start': True})
            if env.start_kind != 'full_start':
                raise ValueError('Demonstration did not restore the canonical initial slot.')
            actions = INITIAL + [0] * delay + candidate['candidate']['actions']
            frames = [observation['image'][-1].copy()]
            rewards, telemetry = [], []
            for action in actions:
                observation, reward, terminal, truncated, info = env.step(action)
                result['physical_decisions'] += 1
                frames.append(observation['image'][-1].copy())
                rewards.append(reward)
                telemetry.append(env.telemetry.to_dict())
                if terminal or truncated:
                    raise ValueError(f'Demonstration interrupted before candidate: {purpose}, delay={delay}.')
            x, y, z = env.telemetry.position
            if not (env.telemetry.grounded is True and env.telemetry.health == 3 and z <= -2100
                    and abs(x - 500) <= 600 and abs(y + 50) <= 200):
                raise ValueError('Path did not end at the declared alive-grounded beyond-opening screen.')
            path = run / f'demonstration-delay-{delay}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, frames=np.stack(frames), actions=np.asarray(actions, np.int64),
                                    rewards=np.asarray(rewards, np.float32))
            result['episodes'].append({'purpose': purpose, 'delay_decisions': delay, 'file': str(path),
                'sha256': digest(path), 'decisions': len(actions), 'final': env.telemetry.to_dict(),
                'action_names': [action_name(a, config) for a in actions],
                'outcome': 'nonterminal_collector_cut', 'full_start_ram_star': False})
            atomic_json(run / 'result.json', result)
        result['state'] = 'completed'
    except BaseException as exc:
        result.update(state='failed', error=repr(exc))
        raise
    finally:
        (env or backend).close()
        atomic_json(run / 'result.json', result)
    print(json.dumps({'state': result['state'], 'run': str(run), 'episodes': len(result['episodes']),
                      'physical_decisions': result['physical_decisions']}))


if __name__ == '__main__':
    main()
