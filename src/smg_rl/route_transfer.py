"""Create a fresh signed IQN branch from verified visual-policy weights."""

import json
from pathlib import Path

import torch

from smg_rl.checkpoints import digest
from smg_rl.dqn import DistributionalAgent
from smg_rl.dqn_replay import PrioritizedFrames
from smg_rl.replay import replay_signature


def transfer_policy(source, old_config, new_config, destination, *, allow_settings_change=False):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('A transfer must not replace an existing checkpoint.')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest['agent_sha256'] != digest(source / 'agent.pt')
            or manifest['replay_sha256'] != digest(source / 'replay.npz')
            or (not allow_settings_change and old_config['dqn'] != new_config['dqn'])):
        raise ValueError('Source checkpoint or model settings do not match.')
    architecture = ('algorithm', 'encoder', 'hidden', 'channels', 'atoms', 'quantiles',
                    'cosine_embedding', 'frame_stack')
    if any(old_config['dqn'].get(k) != new_config['dqn'].get(k) for k in architecture):
        raise ValueError('Explicit settings transfer must retain the network architecture.')
    original = DistributionalAgent.load_policy(source, replay_signature(old_config), 'cpu')
    signature = replay_signature(new_config)
    if signature == replay_signature(old_config):
        raise ValueError('Transfer is only for an explicitly changed reward signature.')
    settings = new_config['dqn']
    agent = DistributionalAgent(signature, settings, 'cpu', new_config['training']['seed'])
    agent.online.load_state_dict(original.online.state_dict())
    agent.target.load_state_dict(original.online.state_dict())
    parameters = {'capacity': settings['capacity'], 'image_size': new_config['environment']['image_size'],
                  'stack': settings['frame_stack'], 'workers': new_config['parallel']['workers'],
                  'n_step': settings['n_step'], 'gamma': settings['gamma'], 'alpha': settings['alpha']}
    replay = PrioritizedFrames(**parameters, seed=new_config['training']['seed'])
    agent.save(destination, replay)
    if not all(torch.equal(a.cpu(), agent.online.state_dict()[k].cpu())
               for k, a in original.online.state_dict().items()):
        raise ValueError('Online policy transfer changed source weights.')
    return {'source': str(source.resolve()), 'source_agent_sha256': manifest['agent_sha256'],
            'source_replay_sha256': manifest['replay_sha256'], 'destination': str(destination.resolve()),
            'destination_agent_sha256': digest(destination / 'agent.pt'),
            'source_real_decisions': manifest['real_decisions'], 'new_real_decisions': 0,
            'optimizer_reset': True, 'replay_reset': True, 'target_synchronized': True,
            'online_weights_transferred': True, 'old_and_new_signature_differ': True}
