"""Model-free distributional Q-learning and immutable resumable checkpoints."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json
from smg_rl.distributional import DistributionalQ, categorical_projection, quantile_huber
from smg_rl.dqn_replay import PrioritizedFrames


class DistributionalAgent:
    def __init__(self, signature, settings, device='cpu', seed=42):
        self.signature, self.settings, self.device = signature, settings, torch.device(device)
        torch.manual_seed(seed)
        self.online = DistributionalQ(signature, settings).to(self.device)
        self.target = deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=settings['learning_rate'],
                                           eps=settings.get('adam_epsilon', .00015))
        self.real_decisions, self.updates, self.target_updates = 0, 0, 0
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, observation, evaluate=False, return_values=False):
        images = torch.as_tensor(observation['image'], device=self.device)
        scenario = torch.as_tensor(observation['scenario'], device=self.device)
        self.online.train(not evaluate)
        if not evaluate:
            self.online.reset_noise()
        values = self.online.expected(images, scenario, self.settings['quantiles'], deterministic=evaluate)
        actions = values.argmax(-1).cpu().numpy()
        epsilon = self.settings.get('evaluation_epsilon', 0.) if evaluate else (1. if self.real_decisions < self.settings['learning_starts']
                                      else self.settings['epsilon'])
        random = self.rng.random(len(actions)) < epsilon
        actions[random] = self.rng.integers(len(self.signature['action_names']), size=int(random.sum()))
        metrics = {'q_mean': float(values.mean()), 'q_max': float(values.max()), 'epsilon': epsilon}
        if return_values:
            metrics['q_values'] = values.cpu().numpy()
        return actions, metrics

    def update(self, replay):
        settings = self.settings
        beta = min(1., settings['beta_start'] + (1 - settings['beta_start']) *
                   self.real_decisions / settings['beta_steps'])
        batch = replay.sample(settings['batch_size'], beta)
        arrays = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items() if k != 'indices'}
        scenario = torch.nn.functional.one_hot(arrays['scenario'], len(self.signature['scenario_names'])).float()
        images, successor = arrays['images'], arrays['next_images']
        actions, rewards, discounts = arrays['action'], arrays['reward'], arrays['discount']
        rows = torch.arange(len(images), device=self.device)
        self.online.train()
        self.target.train()  # Independent factorized noise; target parameters have no gradients.
        self.online.reset_noise()
        self.target.reset_noise()
        metrics = {}
        if settings['algorithm'] == 'rainbow':
            with torch.no_grad():
                selected = self.online.expected(successor, scenario).argmax(-1)  # Double selection.
                distribution = self.target(successor, scenario).softmax(-1)[rows, selected]
                target = categorical_projection(distribution, rewards, discounts, self.online.support)
                unbounded = rewards[:, None] + discounts[:, None] * self.online.support
                metrics['support_clipped_mass'] = float((distribution * (
                    (unbounded < self.online.support[0]) | (unbounded > self.online.support[-1]))).sum(-1).mean())
            predicted = self.online(images, scenario).log_softmax(-1)[rows, actions]
            per_sample = -(target * predicted).sum(-1)
            priority = (target * (target.clamp_min(1e-8).log() - predicted.detach())).sum(-1).clamp_min(0)
        else:
            count = settings['quantiles']
            with torch.no_grad():
                selected = self.online.expected(successor, scenario, count).argmax(-1)
                target_taus = torch.rand(len(images), count, device=self.device)
                next_values = self.target(successor, scenario, target_taus)
                next_values = next_values.gather(-1, selected[:, None, None].expand(-1, count, 1)).squeeze(-1)
                target = rewards[:, None] + discounts[:, None] * next_values
            taus = torch.rand(len(images), count, device=self.device)
            predicted = self.online(images, scenario, taus).gather(
                -1, actions[:, None, None].expand(-1, count, 1)).squeeze(-1)
            per_sample = quantile_huber(predicted, target, taus)
            priority = (target[:, None] - predicted.detach()[:, :, None]).abs().mean((1, 2))
        loss = (per_sample * arrays['weights']).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite distributional Q loss.')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), settings['max_grad_norm'], error_if_nonfinite=True)
        self.optimizer.step()
        replay.priorities(batch['indices'], priority.detach().cpu().numpy() + settings['priority_epsilon'])
        self.updates += 1
        if self.updates % settings['target_update_interval'] == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target_updates += 1
        metrics.update(loss=float(loss.detach()), gradient_norm=float(norm), beta=beta,
                       mean_priority=float(priority.detach().mean()),
                       effective_sample_size=float(arrays['weights'].sum().square() / arrays['weights'].square().sum()))
        return metrics

    def save(self, folder, replay):
        if replay.emitted != self.real_decisions or any(replay.pending):
            raise ValueError('Flush pending cuts and reconcile real/replay counters before checkpointing.')
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        marker = folder / '.incomplete'
        marker.touch()
        torch.save({'online': self.online.state_dict(), 'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(), 'real_decisions': self.real_decisions,
            'updates': self.updates, 'target_updates': self.target_updates,
            'torch_rng': torch.get_rng_state(), 'rng': self.rng.bit_generator.state,
            'cuda_rng': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else []}, folder / 'agent.pt')
        replay.save(folder / 'replay.npz')
        atomic_json(folder / 'manifest.json', {'schema': 1, 'algorithm': self.settings['algorithm'],
            'signature': self.signature, 'settings': self.settings,
            'agent_sha256': digest(folder / 'agent.pt'), 'replay_sha256': digest(folder / 'replay.npz'),
            'replay_emitted': replay.emitted, 'real_decisions': self.real_decisions,
            'note': 'New run resumes with canonical/curriculum resets; collector cuts retain bootstrap.'})
        marker.unlink()

    @classmethod
    def load_policy(cls, folder, signature, device='cpu'):
        folder = Path(folder)
        meta = json.loads((folder / 'manifest.json').read_text())
        if ((folder / '.incomplete').exists() or meta.get('schema') != 1 or meta['signature'] != signature
                or meta['agent_sha256'] != digest(folder / 'agent.pt')):
            raise ValueError('DQN policy identity/integrity mismatch.')
        # Loading a viewer/evaluator must not reseed a learner sharing this process.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            result = cls(signature, meta['settings'], device)
            data = torch.load(folder / 'agent.pt', map_location=result.device, weights_only=True)
            result.online.load_state_dict(data['online'])
            result.real_decisions, result.updates = data['real_decisions'], data['updates']
        return result

    @classmethod
    def load(cls, folder, signature, settings, replay_parameters, device='cpu'):
        folder = Path(folder)
        meta = json.loads((folder / 'manifest.json').read_text())
        if ((folder / '.incomplete').exists() or meta.get('schema') != 1 or meta['signature'] != signature
                or meta['settings'] != settings or meta['agent_sha256'] != digest(folder / 'agent.pt')
                or meta['replay_sha256'] != digest(folder / 'replay.npz')):
            raise ValueError('DQN checkpoint identity/settings/integrity mismatch.')
        result = cls(signature, settings, device)
        data = torch.load(folder / 'agent.pt', map_location=result.device, weights_only=True)
        result.online.load_state_dict(data['online'])
        result.target.load_state_dict(data['target'])
        result.optimizer.load_state_dict(data['optimizer'])
        result.real_decisions, result.updates, result.target_updates = data['real_decisions'], data['updates'], data['target_updates']
        torch.set_rng_state(data['torch_rng'].cpu())
        if data['cuda_rng'] and result.device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in data['cuda_rng']])
        result.rng.bit_generator.state = data['rng']
        replay = PrioritizedFrames.load(folder / 'replay.npz', replay_parameters)
        if (replay.emitted != meta['replay_emitted'] or result.real_decisions != meta['real_decisions']
                or replay.emitted != result.real_decisions):
            raise ValueError('DQN checkpoint counters do not reconcile.')
        return result, replay
