"""Observe latent disagreement during fixed SMG1 inputs without training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from smg_rl.actions import ACTIONS
from smg_rl.config import load_config
from smg_rl.native import NativeDolphin
from smg_rl.world_model import ImaginedAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/assets/performance/smg1-stationary-bonus-20261025.json')
    output = Path(parser.parse_args().output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = load_config('config/staged-smg1-control.yaml')
    config['paths']['user'] = str(Path('.local/stationary-probe/RMGE01/template'))
    checkpoint = Path('runs/staged-smg1-exploration/control/world-model-20261025-004551-267957/world-final.pt')
    metadata = json.loads(checkpoint.with_suffix('.json').read_text())
    agent = ImaginedAgent.load(checkpoint, metadata['signature'], metadata['settings'], 'cuda')
    result = {'scope': 'Fixed-input visual/RAM fixture, deterministic posterior, no optimizer or training evidence',
              'physical_decisions': 0, 'sequences': []}
    backend = NativeDolphin(config)
    try:
        for names in [(), ('jump',), ('spin',), ('crouch',)]:
            backend.load_slot(1)
            initial = backend.memory.sample()
            previous_image = backend.image().copy()
            scenario = torch.ones(1, 1, device=agent.device)
            state = agent.world.initial(1)
            previous_action = torch.zeros(1, dtype=torch.long, device=agent.device)
            action = torch.tensor([ACTIONS.index(names)], device=agent.device)
            row = {'input': '+'.join(names) or 'idle', 'initial': initial.to_dict(), 'trace': []}
            for step in range(24):
                with torch.no_grad():
                    image = torch.as_tensor(backend.image()[None, None], device=agent.device)
                    state, _, _ = agent.world.observe(agent.world.encode(image), previous_action, state,
                        scenario, torch.tensor([step == 0], device=agent.device), sample=False)
                    bonus = float(agent.world.disagreement(agent.world.feature(state, scenario), action).clamp(0, 1))
                backend.advance(names)
                result['physical_decisions'] += 1
                current = backend.memory.sample()
                frame = backend.image().copy()
                row['trace'].append({'telemetry': current.to_dict(), 'disagreement': bonus,
                    'weighted_bonus': 0.01 * bonus,
                    'successive_image_mse': float(np.mean(((frame.astype(float) - previous_image) / 255)**2)),
                    'xz_distance_from_start': float(np.linalg.norm(np.asarray(current.position)[[0, 2]] - np.asarray(initial.position)[[0, 2]]))})
                previous_action, previous_image = action, frame
                if current.dead or current.complete:
                    break
            result['sequences'].append(row)
    finally:
        backend.close()
    result['state'] = 'completed'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'decisions': result['physical_decisions'], 'completed': True}))


if __name__ == '__main__':
    main()
