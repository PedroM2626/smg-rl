"""Bounded actual-game input sanity check, excluded from training evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from smg_rl.actions import ACTIONS, DIRECTIONS, action_name
from smg_rl.config import load_config
from smg_rl.native import NativeDolphin
from smg_rl.rewards import ProgressReward
from smg_rl.world_model import ImaginedAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/assets/performance/control-probe-20261025.json')
    options = parser.parse_args()
    output = Path(options.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'Bounded input/telemetry fixtures; no learning, replay or trained completion evidence',
              'maximum_physical_decisions': 810, 'physical_decisions': 0, 'scenarios': []}
    cases = [
        ('config/staged-smg1-control.yaml', 'runs/staged-smg1-exploration/control/world-model-20261025-004551-267957/world-final.pt'),
        ('config/staged-smg2-terminal_reward_weight.yaml', 'runs/staged-smg2/terminal_reward_weight/world-model-20261025-010019-701413/world-final.pt')]
    torch.set_num_threads(1)
    for config_path, checkpoint in cases:
        config = load_config(config_path)
        config['paths']['user'] = str(Path('.local/control-probe') / config['game']['id'] / 'template')
        metadata = json.loads(Path(checkpoint).with_suffix('.json').read_text())
        agent = ImaginedAgent.load(checkpoint, metadata['signature'], metadata['settings'], 'cuda')
        calibration = json.loads(Path(config['paths']['calibration']).read_text())['variations']
        backend = NativeDolphin(config)
        try:
            for scenario_id, variation in enumerate(config['variations']):
                name = variation['name']
                scenario = torch.eye(len(config['variations']), device=agent.device)[scenario_id:scenario_id+1]
                row = {'game': config['game']['id'], 'scenario': name, 'single_actions': [], 'held_directions': []}
                for action, names in enumerate(ACTIONS):
                    backend.load_slot(variation['start_slot'])
                    initial = backend.memory.sample()
                    with torch.no_grad():
                        embedding = agent.world.encode(torch.as_tensor(backend.image()[None, None], device=agent.device))
                        state, _, _ = agent.world.observe(embedding, torch.zeros(1, dtype=torch.long, device=agent.device),
                            agent.world.initial(1), scenario, torch.ones(1, dtype=torch.bool, device=agent.device), sample=False)
                        next_state = agent.world.imagine_step(state, torch.tensor([action], device=agent.device), scenario, sample=False)
                        predicted = float(agent.world.predict_reward(agent.world.feature(next_state, scenario)))
                    backend.advance(names)
                    result['physical_decisions'] += 1
                    end = backend.memory.sample()
                    progress = ProgressReward(calibration[name]['start'], calibration[name]['goal'],
                        config['environment']['min_reward_y'], config['environment']['max_step_distance'])
                    progress.reset(initial.position)
                    delta, _ = progress.update(end.position, end.dead, end.complete)
                    actual = config['rewards']['step'] + config['rewards']['progress'] * delta
                    row['single_actions'].append({'action': action_name(action), 'prediction_mean_latent': predicted,
                        'actual_one_step_reward': actual, 'initial': initial.to_dict(), 'end': end.to_dict()})
                for direction in DIRECTIONS:
                    backend.load_slot(variation['start_slot'])
                    initial = backend.memory.sample()
                    trace = []
                    for _ in range(24):
                        backend.advance(direction)
                        result['physical_decisions'] += 1
                        end = backend.memory.sample()
                        trace.append(end.to_dict())
                        if end.dead or end.complete:
                            break
                    row['held_directions'].append({'direction': '+'.join(direction) or 'idle',
                        'initial': initial.to_dict(), 'trace': trace,
                        'maximum_xz_distance': max(float(np.linalg.norm(np.asarray(t['position'])[[0,2]] - np.asarray(initial.position)[[0,2]])) for t in trace)})
                result['scenarios'].append(row)
                output.write_text(json.dumps(result, indent=2), encoding='utf-8')
                print(json.dumps({'game': row['game'], 'scenario': name, 'decisions': result['physical_decisions']}), flush=True)
        finally:
            backend.close()
    result['state'] = 'completed'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
