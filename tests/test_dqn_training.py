from collections import deque

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from smg_rl import dqn_training
from smg_rl.config import load_config
from smg_rl.dqn import DistributionalAgent
from smg_rl.replay import replay_signature


@pytest.fixture(autouse=True)
def portable_memory_headroom(monkeypatch):
    # Tiny vector fixtures must not require a production six-emulator RAM budget.
    monkeypatch.setattr(dqn_training, 'require_memory_headroom', lambda: None)
    monkeypatch.setattr(dqn_training, 'memory_snapshot', lambda: {
        'commit_available_bytes': 16 * 2**30, 'physical_available_bytes': 8 * 2**30})
    monkeypatch.setattr(dqn_training, 'require_disk_headroom', lambda _: None)
    monkeypatch.setattr(dqn_training, 'disk_snapshot', lambda _: {'runs': {'free_bytes': 8 * 2**30}})


class BoundaryVector:
    """Deterministic collector fixture; never a learned/game completion."""
    def __init__(self):
        self.episode = 0

    def seed(self, seed):
        pass

    def observation(self):
        return {'image': np.stack(self.frames)[None], 'scenario': np.ones((1, 1), np.float32)}

    def reset(self):
        self.episode += 1
        self.steps = 0
        self.frames = deque([np.full((40, 40), self.episode * 10, np.uint8)] * 4, maxlen=4)
        self.reset_infos = [{'start_kind': 'curriculum' if self.episode % 2 == 0 else 'full_start'}]
        return self.observation()

    def step(self, actions):
        self.steps += 1
        self.frames.append(np.full((40, 40), self.episode * 10 + self.steps, np.uint8))
        done = self.steps == 3
        info = {'worker_episode': self.episode, 'episode_step': self.steps,
                'variation': 'sweet_sweet', 'start_kind': self.reset_infos[0]['start_kind'],
                'dead': done, 'complete': False, 'scene_nerve': 0x806A1BD0,
                'health_lost': 0, 'start_relative_route_progress': .01, 'TimeLimit.truncated': False}
        observation = self.observation()
        if done:
            info['episode'] = {'r': -2., 'l': self.steps}
            info['terminal_observation'] = {k: v[0] for k, v in observation.items()}
            observation = self.reset()
        return observation, np.array([-2. if done else -.001]), np.array([done]), (info,)

    def close(self):
        pass


@pytest.mark.parametrize('algorithm', ['rainbow', 'iqn'])
def test_full_model_free_loop_records_tensorboard_boundaries_and_resumable_replay(tmp_path, monkeypatch, algorithm):
    config = load_config(f'config/{algorithm}-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=32, learning_starts=2, batch_size=2, channels=2, hidden=32,
                         train_every=1, target_update_interval=2, quantiles=4,
                         encoder='impala' if algorithm == 'iqn' else 'nature')
    vector = BoundaryVector()
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: vector)
    run = dqn_training.train_dqn(config, 5)
    agent, replay = DistributionalAgent.load(run / 'checkpoint-final', replay_signature(config), config['dqn'],
        {'capacity': 32, 'image_size': 40, 'stack': 4, 'workers': 1, 'n_step': 3, 'gamma': .997, 'alpha': .5})
    assert replay.emitted == agent.real_decisions == 5
    assert agent.attempts['sweet_sweet/full_start']['completed'] == 1
    assert agent.updates > 0 and not any(replay.pending)
    import json

    status = json.loads((run / 'status.json').read_text())
    assert status['visual_encoder_trainable']
    assert status['visual_encoder_first_update_probe']['weights_changed']
    assert status['visual_encoder_first_update_probe']['gradient_l2'] > 0
    assert replay.arrays['terminal'][:3].all()
    assert not replay.arrays['terminal'][3:5].any()  # Collector cut bootstraps.
    assert replay.observation(replay.arrays['next'][0])[-1, 0, 0] == 13  # Final observation, not autoreset20.
    events = EventAccumulator(str(next((tmp_path / 'tensorboard').rglob('events.out*')).parent))
    events.Reload()
    assert 'learning/loss' in events.Tags()['scalars']
    assert events.Scalars('training/optimizer_updates')[-1].value > 0
    assert events.Scalars('attempt_axis/sweet_sweet/full_start/opening_advance')[-1].step == 1


def test_wall_clock_budget_stops_before_an_unallocated_decision(tmp_path, monkeypatch):
    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=32, learning_starts=2, batch_size=2, channels=2, hidden=32,
                         train_every=1, target_update_interval=2, quantiles=4)
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: BoundaryVector())
    run = dqn_training.train_dqn(config, 100, max_seconds=1e-9)
    import json
    status = json.loads((run / 'status.json').read_text())
    assert status['real_decisions'] == 0 and status['state'] == 'completed'
    assert status['stop_reason'] == 'wall_clock_budget'


def test_memory_pressure_stops_cleanly_and_flushes_committed_replay(tmp_path, monkeypatch):
    import json

    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=128, learning_starts=100, batch_size=2, channels=2, hidden=32)
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: BoundaryVector())
    monkeypatch.setattr(dqn_training, 'memory_safe', lambda _: False)
    run = dqn_training.train_dqn(config, 100)
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'stopped' and status['stop_reason'] == 'memory_pressure'
    assert status['real_decisions'] == status['replay_emitted'] == 64
    assert status['error'] is None


def test_absolute_deadline_saves_clean_checkpoint_before_next_decision(tmp_path, monkeypatch):
    import json

    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=32, learning_starts=20, batch_size=2, channels=2, hidden=32)
    wall = [1000.]
    vector = BoundaryVector()
    original_reset = vector.reset

    def reset_after_deadline():
        wall[0] = 2001.
        return original_reset()

    vector.reset = reset_after_deadline
    monkeypatch.setattr(dqn_training.time, 'time', lambda: wall[0])
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: vector)
    run = dqn_training.train_dqn(config, 100, deadline_utc='1970-01-01T00:33:20+00:00')
    status = json.loads((run / 'status.json').read_text())
    assert status['real_decisions'] == 0 and status['state'] == 'completed'
    assert status['stop_reason'] == 'scheduled_deadline'
    assert (run / 'checkpoint-final' / 'agent.pt').exists()


def test_disk_pressure_stops_before_checkpoint_space_is_exhausted(tmp_path, monkeypatch):
    import json

    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=128, learning_starts=100, batch_size=2, channels=2, hidden=32)
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: BoundaryVector())
    monkeypatch.setattr(dqn_training, 'disk_safe', lambda _: False)
    run = dqn_training.train_dqn(config, 100)
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'stopped' and status['stop_reason'] == 'disk_pressure'
    assert status['real_decisions'] == 64
    assert (run / 'checkpoint-final/manifest.json').exists()


def test_failed_checkpoint_reports_failure_instead_of_stale_running_state(tmp_path, monkeypatch):
    import json

    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=32, learning_starts=20, batch_size=2, channels=2, hidden=32)
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: BoundaryVector())

    def cannot_save(*args):
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(DistributionalAgent, 'save', cannot_save)
    with pytest.raises(OSError, match='No space'):
        dqn_training.train_dqn(config, 2)
    run = next((tmp_path / 'runs').glob('dqn-*'))
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'failed' and status['stop_reason'] == 'checkpoint_failure'
    assert 'No space' in status['error']
