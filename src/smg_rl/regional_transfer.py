"""Explicit same-control visual warm start across a regional reward revision."""

import argparse
import json
from pathlib import Path

import torch

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.replay import replay_signature
from smg_rl.world_model import ImaginedAgent


def rebase_regions(source, signature, settings, seed):
    same = ('game_id', 'scenario_names', 'action_names', 'image_size', 'crop',
            'frame_skip', 'action_execution')
    if (any(source.signature.get(k) != signature.get(k) for k in same)
            or len(signature['action_names']) != 8 or signature['frame_skip'] != 12
            or not signature.get('regional_reward', {}).get('enabled')
            or signature == source.signature):
        raise ValueError('Regional warm start requires unchanged eight-action twelve-frame interfaces and a new reward identity.')
    torch.manual_seed(seed)
    result = ImaginedAgent(signature, settings, str(source.device))
    weights = result.world.state_dict()
    original = source.world.state_dict()
    if weights.keys() != original.keys():
        raise ValueError('Visual world-model architecture differs.')
    for name in weights:
        if name.startswith(('reward.', 'continuation.')) or name in ('reward_low', 'reward_high'):
            continue
        if original[name].shape != weights[name].shape:
            raise ValueError(f'Incompatible visual dynamics: {name}')
        weights[name] = original[name].clone()
    result.world.load_state_dict(weights)
    with torch.no_grad():
        result.actor[-1].weight.zero_()
        prior = torch.tensor([.16] * 5 + [.15, .025, .025], device=result.device)
        result.actor[-1].bias.copy_(prior.log())
    return result


def create(config, source_path, destination):
    source_path, destination = Path(source_path), Path(destination)
    metadata = json.loads(source_path.with_suffix('.json').read_text())
    source = ImaginedAgent.load(source_path, metadata['signature'], metadata['settings'], config['training']['device'])
    result = rebase_regions(source, replay_signature(config), config['world_model'], config['training']['seed'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination)
    counts = ('real_decisions', 'model_updates', 'actor_updates', 'imagined_transitions')
    atomic_json(destination.with_suffix('.provenance.json'), {
        'source': str(source_path), 'source_sha256': digest(source_path),
        'source_counts': {k: getattr(source, k) for k in counts},
        'new_branch_counts': {k: getattr(result, k) for k in counts},
        'transfer': 'Retain encoder, recurrent latent dynamics, decoder and ensemble under identical physical controls.',
        'reset': 'Reward/support, continuation, actor, critics, optimizers, replay RNG and new-branch counters.',
        'initial_action_prior': '80% walking/idle,15% jump,2.5% spin,2.5% crouch; not a trained policy.',
        'gate': 'Original thresholds unchanged. No readiness inherited; fresh compatible real replay required.'})
    assert digest(source_path) == metadata['sha256']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    create(load_config(args.config), args.source, args.output)
