"""Explicit legacy-to-motor weight projection; never silently resumes old replay."""

import argparse
import json
from pathlib import Path

import torch

from smg_rl.actions import action_set
from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.replay import replay_signature
from smg_rl.world_model import ImaginedAgent


def project_motor(source, signature, settings, seed=142):
    actions = len(signature['action_names'])
    if len(source.signature['action_names']) != 54 or actions not in (8,16):
        raise ValueError('This explicit transfer supports legacy54 to motor8-v1 or motor16-v1.')
    if (signature['game_id'] != source.signature['game_id']
            or signature['scenario_names'] != source.signature['scenario_names']
            or signature['image_size'] != source.signature['image_size']
            or signature['crop'] != source.signature['crop']
            or signature['frame_skip'] != source.signature['frame_skip']):
        raise ValueError('Motor transfer cannot change the game, scenario, image or frame layout.')
    indices = [source.signature['action_names'].index('idle' if name == 'none' else name)
               for name in signature['action_names']]
    torch.manual_seed(seed)
    result = ImaginedAgent(signature, settings, str(source.device))
    weights = result.world.state_dict()
    for name, value in source.world.state_dict().items():
        if name.startswith('reward.') or name in ('reward_low','reward_high'):
            continue  # Fresh reward predictor/support; new targets need real fitting.
        if name == 'gru.weight_ih':
            columns = (list(range(source.world.stoch)) + [source.world.stoch + i for i in indices]
                       + list(range(source.world.stoch + 54, value.shape[1])))
            value = value[:, columns]
        elif name.startswith('ensemble.') and name.endswith('.0.weight'):
            value = value[:, list(range(source.world.features)) + [source.world.features + i for i in indices]]
        if value.shape != weights[name].shape:
            raise ValueError(f'Unmapped world-model dimension: {name}')
        weights[name] = value.clone()
    result.world.load_state_dict(weights)
    # Representation and dynamics are transferred. Policy/value/optimizers start fresh.
    with torch.no_grad():
        result.actor[-1].weight.zero_()
        probabilities = ([0.8 / 5] * 5 + [0.15,0.025,0.025] if actions == 8
                         else [0.8 / 9] * 9 + [0.15 / 5] * 5 + [0.025,0.025])
        prior = torch.tensor(probabilities, device=result.device)
        result.actor[-1].bias.copy_(prior.log())
    return result


def create(config, source_path, destination):
    if len(action_set(config)) not in (8,16):
        raise ValueError('Select a versioned motor action profile explicitly.')
    source_path, destination = Path(source_path), Path(destination)
    metadata = json.loads(source_path.with_suffix('.json').read_text())
    source = ImaginedAgent.load(source_path, metadata['signature'], metadata['settings'], config['training']['device'])
    agent = project_motor(source, replay_signature(config), config['world_model'], config['training']['seed'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    agent.save(destination)
    atomic_json(destination.with_suffix('.provenance.json'), {
        'source':str(source_path), 'source_sha256':digest(source_path),
        'source_real_decisions':source.real_decisions, 'new_real_decisions':0,
        'transfer':'Project GRU/ensemble action columns; retain visual/latent dynamics. Fresh reward, actor, critic and optimizer states.',
        'initial_action_prior':'80% walking/idle, 15% jump, 2.5% spin, 2.5% crouch',
        'limitation':'Pulse semantics differ; original prediction gates require fresh compatible real replay.'})
    assert digest(source_path) == metadata['sha256']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    create(load_config(args.config), args.source, args.output)
