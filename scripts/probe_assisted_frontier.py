"""Small frozen-policy probe from a validated, drifting intermediate state."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn import DistributionalAgent
from smg_rl.env import GalaxyEnv
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.replay import replay_signature
from smg_rl.train import new_run

os.chdir(Path(__file__).resolve().parents[1])
CONFIG = 'config/smg1-six-two-hour.yaml'
CHECKPOINT = 'runs/model-free/smg1-six-two-hour/dqn-20261027-114555-611503/checkpoint-final'
ANCHOR = 'sweet-sweet-first-chocolate.state'


def main():
    config = load_config(CONFIG)
    config['model_free_run'] = 'evaluation'
    agent = DistributionalAgent.load_policy(CHECKPOINT, replay_signature(config), config['training']['device'])
    before = digest(Path(CHECKPOINT) / 'agent.pt')
    config['paths']['runs'] = str(Path(config['root']) / 'runs/model-free/assisted-frontier-probe-20261028')
    config['replay']['enabled'] = False
    torch.set_num_threads(config['training']['torch_threads'])
    run = new_run(config, 'probe')
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    backend = NativeDolphin(config)
    env = None
    report = {'state': 'running', 'anchor': ANCHOR, 'scope': 'Assisted frozen-policy probe; never full-start success.',
              'anchor_warning': 'Prior 60-frame idle probe drifted about 478 world units on a conveyor.',
              'maximum_decisions_per_attempt': 200, 'source_agent_sha256': before, 'episodes': []}
    atomic_json(run / 'result.json', report)
    print(json.dumps({'run': str(run)}), flush=True)
    try:
        env = GalaxyEnv(config, backend, run / 'worker-00', memory=backend.memory, intrinsic=False, variation=0)
        seed = next(i for i in range(10000)
                    if (entry := env.curriculum_bank.choose(np.random.default_rng(i), 1))
                    and entry['file'] == ANCHOR)
        for mode, draw in [('greedy', 11), ('stochastic', 11), ('stochastic', 23), ('stochastic', 47)]:
            agent.rng = np.random.default_rng(draw)
            torch.manual_seed(draw)
            observation, initial = env.reset(seed=seed, options={'variation': 0})
            if env.start_kind != 'curriculum' or env.start_state_id != ANCHOR:
                raise ValueError('Probe did not restore the declared assisted anchor.')
            info, terminal = {}, False
            for step in range(200):
                action, _ = agent.act({key: value[None] for key, value in observation.items()},
                                      evaluate=mode == 'greedy',
                                      epsilon_override=.1 if mode == 'stochastic' else None)
                observation, _, ended, truncated, info = env.step(int(action[0]))
                if ended or truncated:
                    terminal = True
                    break
            report['episodes'].append({'mode': mode, 'action_seed': draw, 'reset_seed': seed,
                'decisions': step + 1, 'outcome': 'star' if info['complete'] else 'death' if info['dead']
                else 'timeout' if terminal else 'evaluation_budget_cut',
                'assisted_ram_star': bool(info['complete'] and not info['dead']),
                'route_prefix_exited': bool(info['route_prefix_exited']),
                'start_relative_route_progress': info['start_relative_route_progress'],
                'initial': initial, 'final': env.telemetry.to_dict()})
            atomic_json(run / 'result.json', report)
        report['state'] = 'completed'
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        raise
    finally:
        (env or backend).close()
        report['source_agent_unchanged'] = digest(Path(CHECKPOINT) / 'agent.pt') == before
        atomic_json(run / 'result.json', report)
    print(json.dumps({'run': str(run), 'state': report['state'], 'episodes': len(report['episodes'])}))


if __name__ == '__main__':
    main()
