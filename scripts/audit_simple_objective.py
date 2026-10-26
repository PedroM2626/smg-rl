"""Analytic reward controls and synthetic encoder-cost measurements, not learning."""

import json
import time
from copy import deepcopy

import numpy as np
import torch

from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.distributional import DistributionalQ
from smg_rl.replay import replay_signature
from smg_rl.rewards import RegionalAdvance, validate_simple_objective


def main():
    config = load_config('config/impala-iqn-smg1.yaml')
    for game in ('smg1', 'smg2'):
        c = load_config(f'config/impala-iqn-{game}.yaml')
        validate_simple_objective(c['regional_reward'], c['rewards'], c['dqn']['gamma'])
    returns = []
    for duration in (1, 14, 180, 3000):
        old = -.001 * (1 - .997 ** duration) / (1 - .997) - 2 * .997 ** (duration - 1)
        returns.append({'failure_decisions': duration, 'old_discounted_return_no_progress': old,
                        'new_return_no_progress': -2., 'new_return_half_opening': -1.5})
    controls = {}
    for name, start, trace in [('stationary', 0., [0.] * 20), ('revisit', 0., [0., .2, 0., .2]),
                               ('assisted_stationary', .4, [.4] * 20),
                               ('assisted_advance', .4, [.4, .6, .8])]:
        reward = RegionalAdvance()
        reward.reset(start)
        controls[name] = sum(reward.update(p, 12, terminal=i == len(trace) - 1) for i, p in enumerate(trace))
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    costs = {}
    for encoder in ('nature', 'impala'):
        settings = deepcopy(config['dqn'])
        settings['encoder'] = encoder
        net = DistributionalQ(replay_signature(config), settings).to(device)
        image = torch.randint(0, 256, (32, 4, 84, 84), device=device, dtype=torch.uint8)
        scenario = torch.ones(32, 1, device=device)
        def synchronize():
            if device == 'cuda':
                torch.cuda.synchronize()
        for _ in range(3):
            net.zero_grad(set_to_none=True)
            net.expected(image, scenario, 32).square().mean().backward()
        samples = []
        for _ in range(10):
            net.zero_grad(set_to_none=True)
            synchronize()
            started = time.perf_counter()
            net.expected(image, scenario, 32).square().mean().backward()
            synchronize()
            samples.append((time.perf_counter() - started) * 1000)
        net.eval()
        inference = []
        with torch.no_grad():
            for _ in range(20):
                synchronize()
                started = time.perf_counter()
                net.expected(image[:2], scenario[:2], 32, deterministic=True)
                synchronize()
                inference.append((time.perf_counter() - started) * 1000)
        costs[encoder] = {'parameters': sum(p.numel() for p in net.parameters()),
                          'synthetic_forward_backward_batch32_median_ms': float(np.median(samples)),
                          'inference_batch2_median_ms': float(np.median(inference))}
        del net, image, scenario
    report = {'scope': 'Analytic controls and random-pixel runtime diagnostic; no game learning or efficacy ranking.',
              'failure_returns': returns, 'credit_controls': controls, 'encoder_cost': costs, 'device': device,
              'choice': 'Fresh IQN with original IMPALA-style residual encoder, simple advance reward and gradual exploration.',
              'genetic_assessment': 'Not selected: no local evidence of superior learning; population rollouts multiply the emulator cost. No GA experiment was run.',
              'limitations': ['Duration neutrality does not force movement.', 'Route maps cover verified openings only.',
                             'Gamma1 is finite-episode undiscounted Q-learning, not a full BTR reproduction.',
                             'Multiple deliberate changes form a new intervention, not an attribution ablation.']}
    atomic_json('docs/assets/performance/simple-impala-objective-audit-20261026.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
