"""Finite IQN optimizer smoke on copied Rainbow health replay; no new game steps."""

import json
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn import DistributionalAgent
from smg_rl.dqn_replay import PrioritizedFrames
from smg_rl.replay import replay_signature
from smg_rl.train import new_run


def main():
    torch.set_num_threads(1)
    state = json.loads(Path('.local/model-free-state.json').read_text())
    reports = []
    for game in ('smg1', 'smg2'):
        job = state['current_jobs'][game]
        assert job['review']['state'] == 'passed'
        config = load_config(f'config/iqn-{game}.yaml')
        checkpoint = Path(job['run']) / 'checkpoint-final'
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        assert manifest['replay_sha256'] == digest(checkpoint / 'replay.npz')
        s = config['dqn']
        replay = PrioritizedFrames.load(checkpoint / 'replay.npz', {
            'capacity': s['capacity'], 'image_size': config['environment']['image_size'],
            'stack': s['frame_stack'], 'workers': config['parallel']['workers'],
            'n_step': s['n_step'], 'gamma': s['gamma'], 'alpha': s['alpha']})
        agent = DistributionalAgent(replay_signature(config), s, config['training']['device'], config['training']['seed'])
        agent.real_decisions = replay.emitted  # Reused source data, never new physical collection.
        config['model_free_run'] = 'offline_implementation_smoke'
        run = new_run(config, 'dqn-offline-smoke')
        writer = SummaryWriter(str(Path(config['paths']['tensorboard']) / 'iqn-offline-smoke' / game / run.name))
        started = time.perf_counter()
        try:
            for update in range(64):
                metrics = agent.update(replay)
                for key, value in metrics.items():
                    writer.add_scalar(f'learning/{key}', value, update + 1)
            elapsed = time.perf_counter() - started
            agent.save(run / 'checkpoint-final', replay)
            record = {'game': game, 'state': 'completed', 'run': str(run), 'updates': 64,
                'physical_new_real_decisions': 0, 'imagined_transitions': 0,
                'reused_source_real_decisions': replay.emitted,
                'source_checkpoint': str(checkpoint), 'source_replay_sha256': manifest['replay_sha256'],
                'optimizer_wall_seconds': elapsed, 'last_metrics': metrics,
                'scope': 'Implementation/finite-gradient smoke on reused real data, not unseen evaluation or a Rainbow/IQN performance comparison.'}
            atomic_json(run / 'smoke.json', record)
            reports.append(record)
        finally:
            writer.flush()
            writer.close()
    atomic_json('docs/assets/performance/model-free-iqn-smoke-20261026.json', {'reports': reports})
    print(json.dumps(reports))


if __name__ == '__main__':
    main()
