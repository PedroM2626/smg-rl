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

    def step_async(self, actions):
        self.pending = actions

    def step_wait(self):
        return self.step(self.pending)

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
    monkeypatch.setattr(dqn_training, 'memory_safe', lambda _, **kwargs: False)
    run = dqn_training.train_dqn(config, 100)
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'stopped' and status['stop_reason'] == 'memory_pressure'
    assert status['real_decisions'] == status['replay_emitted'] == 64
    assert status['error'] is None
    assert status['memory_guard_snapshot']['physical_available_bytes'] > 0


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


class AnchorVector(BoundaryVector):
    """Every start is the same assisted anchor; records the controls actually sent."""
    def __init__(self):
        super().__init__()
        self.sent = []

    def reset(self):
        observation = super().reset()
        self.reset_infos = [{'start_kind': 'curriculum', 'anchor_id': 'anchor.state'}]
        return observation

    def step(self, actions):
        self.sent.append((self.episode, int(actions[0])))
        return super().step(actions)


def test_scripted_demonstration_episodes_are_sparse_exact_and_never_learned_success(tmp_path, monkeypatch):
    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=32, learning_starts=2, batch_size=2, channels=2, hidden=32,
                         train_every=1, target_update_interval=2, quantiles=4)
    config['curriculum'] = {'enabled': True}
    config['demonstration'] = {'anchor': 'anchor.state', 'actions': [5, 6, 7], 'every_episodes': 2,
                               'margin': .8}
    vector = AnchorVector()
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: vector)
    run = dqn_training.train_dqn(config, 12)
    import json

    status = json.loads((run / 'status.json').read_text())
    # Starts 1 and 3 replay the demonstration exactly; starts 2 and 4 are policy-driven.
    assert [a for e, a in vector.sent if e in (1, 3)] == [5, 6, 7, 5, 6, 7]
    assert status['scripted_demonstration_episodes_this_run'] == 2
    assert 'sweet_sweet/scripted_demonstration' in status['attempts']
    assert status['attempts']['sweet_sweet/curriculum']['completed'] == 2
    assert status['assisted_stars_this_run'] == 0
    with np.load(run / 'checkpoint-final' / 'replay.npz') as archive:
        # Exactly the six scripted transitions carry the label used by the margin loss.
        assert int(archive['demonstration'][:12].sum()) == 6
    config['demonstration']['every_episodes'] = 1
    with pytest.raises(ValueError, match='sparse episode interval'):
        dqn_training.demonstration_schedule(config)


def test_scripted_episodes_replay_the_main_or_a_recovery_sequence(tmp_path, monkeypatch):
    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=64, learning_starts=2, batch_size=2, channels=2, hidden=32,
                         train_every=1, target_update_interval=2, quantiles=4)
    config['curriculum'] = {'enabled': True}
    config['demonstration'] = {'anchor': 'anchor.state', 'actions': [5, 6, 7], 'every_episodes': 2,
                               'recovery_sequences': {'anchor.state': [[7, 6, 5]]}}
    vector = AnchorVector()
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: vector)
    dqn_training.train_dqn(config, 24)
    # Episodes 1, 3, 5 and 7 are scripted; each takes the whole sequence it drew, never a mixture.
    for episode in (1, 3, 5, 7):
        assert [a for e, a in vector.sent if e == episode][:3] in ([5, 6, 7], [7, 6, 5])


def test_collection_epsilon_overrides_acting_without_changing_signed_settings(tmp_path, monkeypatch):
    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=32, learning_starts=2, batch_size=2, channels=2, hidden=32,
                         train_every=1, target_update_interval=2, quantiles=4)
    config['curriculum'] = {'enabled': True}
    config['exploration'] = {'epsilon': 0.0}
    seen = []
    original = dqn_training.DistributionalAgent.act

    def spy(self, observation, evaluate=False, return_values=False, epsilon_override=None):
        seen.append((evaluate, epsilon_override))
        return original(self, observation, evaluate, return_values, epsilon_override)

    monkeypatch.setattr(dqn_training.DistributionalAgent, 'act', spy)
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: AnchorVector())
    run = dqn_training.train_dqn(config, 4)
    import json

    assert {override for evaluate, override in seen if not evaluate} == {0.0}
    assert json.loads((run / 'status.json').read_text())['epsilon'] == 0.0
    config['exploration'] = {'epsilon': 1.5}
    with pytest.raises(ValueError, match='collection epsilon'):
        dqn_training.train_dqn(config, 4)


def test_demonstration_mapping_validates_every_anchor_sequence():
    config = load_config('config/iqn-smg1.yaml')
    config['curriculum'] = {'enabled': True}
    config['demonstration'] = {'sequences': {'a.state': [3, 5], 'b.state': [7]}, 'every_episodes': 10}
    assert dqn_training.demonstration_schedule(config)['sequences']['b.state'] == [7]
    config['demonstration']['sequences']['b.state'] = [8]
    with pytest.raises(ValueError, match='valid actions'):
        dqn_training.demonstration_schedule(config)


def test_recovery_sequences_join_the_pool_of_a_main_demonstration_anchor():
    config = load_config('config/iqn-smg1.yaml')
    config['curriculum'] = {'enabled': True}
    config['demonstration'] = {'sequences': {'a.state': [3, 5]}, 'every_episodes': 10,
                               'recovery_sequences': {'a.state': [[3, 1, 5], [3, 2, 5]]}}
    assert dqn_training.demonstration_schedule(config)['recoveries']['a.state'] == [[3, 1, 5], [3, 2, 5]]
    assert dqn_training.demonstration_schedule({**config, 'demonstration': {
        'sequences': {'a.state': [3]}, 'every_episodes': 10}})['recoveries'] == {}
    config['demonstration']['recovery_sequences'] = {'b.state': [[3]]}
    with pytest.raises(ValueError, match='Recovery sequences'):
        dqn_training.demonstration_schedule(config)
    config['demonstration']['recovery_sequences'] = {'a.state': [[8]]}
    with pytest.raises(ValueError, match='Recovery sequences'):
        dqn_training.demonstration_schedule(config)


def test_margin_loss_raises_the_demonstrated_label_above_alternatives():
    import torch

    from smg_rl.dqn_replay import PrioritizedFrames

    config = load_config('config/iqn-smg1.yaml')
    config['dqn'].update(capacity=16, batch_size=8, channels=2, hidden=32, quantiles=4, learning_rate=1e-3)
    config['environment']['image_size'] = 40
    torch.manual_seed(0)
    agent = DistributionalAgent(replay_signature(config), config['dqn'], 'cpu', 3)
    replay = PrioritizedFrames(16, 40, workers=1, n_step=1, seed=3)
    current = replay.add_frame(np.zeros((40, 40), np.uint8))
    for step in range(8):
        successor = replay.add_frame(np.full((40, 40), step + 1, np.uint8), current)
        replay.append(0, current, 5, 0., successor, 0, False, False, True, demonstration=True)
        current = successor
    image = np.stack([replay.observation(current)])
    scenario = torch.ones(1, len(agent.signature['scenario_names']))

    def gap():
        with torch.no_grad():
            values = agent.online.expected(torch.as_tensor(image), scenario, 4, deterministic=True)[0]
        return float(values[5] - torch.cat([values[:5], values[6:]]).max())

    before = gap()
    for _ in range(60):
        metrics = agent.update(replay, .8)
    assert metrics['demonstration_rows'] > 0 and gap() > before


def test_policy_continues_after_a_short_demonstration_without_margin_labels(tmp_path, monkeypatch):
    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=32, learning_starts=2, batch_size=2, channels=2, hidden=32,
                         train_every=1, target_update_interval=2, quantiles=4)
    config['curriculum'] = {'enabled': True}
    config['demonstration'] = {'sequences': {'anchor.state': [5]}, 'every_episodes': 2}
    vector = AnchorVector()
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: vector)
    run = dqn_training.train_dqn(config, 12)
    import json

    ledger = [json.loads(line) for line in (run / 'collector-steps.jsonl').read_text().splitlines()]
    # Each scripted three-step episode has one demonstrated decision, then two policy decisions.
    assert sum(row['demonstrated'][0] for row in ledger) == 2
    assert sum(row['scripted'][0] for row in ledger) == 6
    with np.load(run / 'checkpoint-final' / 'replay.npz') as archive:
        assert int(archive['demonstration'][:12].sum()) == 2


def test_overlapped_learning_collects_identically_and_defers_at_most_one_tick(tmp_path, monkeypatch):
    import json

    results = {}
    for overlap in (False, True):
        config = load_config('config/iqn-smg1.yaml')
        config['paths'].update(runs=str(tmp_path / str(overlap) / 'runs'),
                               tensorboard=str(tmp_path / str(overlap) / 'tensorboard'))
        config['environment']['image_size'] = 40
        config['parallel'].update(workers=1, overlap_learning=overlap)
        config['training']['device'] = 'cpu'
        config['dqn'].update(capacity=32, learning_starts=2, batch_size=2, channels=2, hidden=32,
                             train_every=1, target_update_interval=2, quantiles=4)
        vector = BoundaryVector()
        monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: vector)
        run = dqn_training.train_dqn(config, 9)
        results[overlap] = json.loads((run / 'status.json').read_text())
    assert results[True]['real_decisions'] == results[False]['real_decisions'] == 9
    assert results[True]['episodes_this_run'] == results[False]['episodes_this_run']
    assert 0 < results[False]['optimizer_updates'] - results[True]['optimizer_updates'] <= 1


class JitterVector(AnchorVector):
    """Even starts are exact anchor restores; odd starts report start jitter."""
    def reset(self):
        observation = super().reset()
        self.reset_infos[0]['start_jitter_frames'] = 0 if self.episode % 2 == 0 else 7
        return observation


def test_jittered_starts_never_run_open_loop_demonstrations(tmp_path, monkeypatch):
    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment'].update(image_size=40, assisted_start_jitter_frames=60)
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=32, learning_starts=2, batch_size=2, channels=2, hidden=32,
                         train_every=1, target_update_interval=2, quantiles=4)
    config['curriculum'] = {'enabled': True}
    config['demonstration'] = {'sequences': {'anchor.state': [5, 6, 7]}, 'every_episodes': 10}
    vector = JitterVector()
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: vector)
    run = dqn_training.train_dqn(config, 12)
    import json

    ledger = [json.loads(line) for line in (run / 'collector-steps.jsonl').read_text().splitlines()]
    scripted_episodes = {row['worker_episodes'][0] for row in ledger if row['scripted'][0]}
    assert scripted_episodes == {2, 4}
    assert [a for e, a in vector.sent if e in (2, 4)] == [5, 6, 7, 5, 6, 7]


def test_uniform_prefix_handover_replays_a_prefix_then_the_policy_continues(tmp_path, monkeypatch):
    import json

    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=64, learning_starts=2, batch_size=2, channels=2, hidden=32,
                         train_every=1, target_update_interval=2, quantiles=4)
    config['curriculum'] = {'enabled': True}
    config['demonstration'] = {'sequences': {'anchor.state': [5, 6, 7]}, 'every_episodes': 2,
                               'handover': 'uniform_prefix'}
    vector = AnchorVector()
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: vector)
    run = dqn_training.train_dqn(config, 48)
    ledger = [json.loads(line) for line in (run / 'collector-steps.jsonl').read_text().splitlines()]
    attempts = [json.loads(line) for line in (run / 'attempts.jsonl').read_text().splitlines()]
    prefixes = {a['worker_episode']: a['script_prefix'] for a in attempts if a['start_kind'] == 'scripted_demonstration'}
    assert prefixes and all('script_prefix' not in a for a in attempts if a['start_kind'] != 'scripted_demonstration')
    for episode, prefix in prefixes.items():
        rows = [r for r in ledger if r['worker_episodes'][0] == episode]
        # Only the replayed prefix is demonstrated; it matches the start of the sequence.
        assert [r['actions'][0] for r in rows if r['demonstrated'][0]] == [5, 6, 7][:prefix]
        assert all(r['scripted'][0] for r in rows)
    assert len(set(prefixes.values())) > 1
    config['demonstration']['handover'] = 'random'
    with pytest.raises(ValueError, match='handover'):
        dqn_training.demonstration_schedule(config)


class FrontierVector(AnchorVector):
    def step(self, actions):
        observation, rewards, done, (info,) = super().step(actions)
        return observation, rewards, done, ({**info, 'frontier_state': 'anchor.state'},)


def test_hybrid_frontier_probes_act_greedily_and_are_kept_out_of_training_attempts(tmp_path, monkeypatch):
    import json

    from smg_rl.checkpoints import digest

    config = load_config('config/iqn-smg1.yaml')
    config['paths'].update(runs=str(tmp_path / 'runs'), tensorboard=str(tmp_path / 'tensorboard'))
    bank = tmp_path / 'bank.json'
    bank.write_text(json.dumps({'schema': 2, 'source_full_start_ram_star': True,
                                'entries': [{'file': 'anchor.state'}]}))
    config['paths'].update(curriculum=str(bank), curriculum_progress=str(tmp_path / 'progress.json'))
    config['environment']['image_size'] = 40
    config['parallel']['workers'] = 1
    config['training']['device'] = 'cpu'
    config['dqn'].update(capacity=64, learning_starts=2, batch_size=2, channels=2, hidden=32,
                         train_every=1, target_update_interval=2, quantiles=4)
    config['curriculum'] = {'enabled': True, 'mode': 'human_backward_v2', 'spec_sha256': digest(bank)}
    config['frontier_rule'] = {'mode': 'greedy_hybrid', 'probe_every': 2, 'probe_streak': 2,
                               'robust_window': 10, 'robust_min': .2}
    original = DistributionalAgent.act

    def labelled(self, observation, evaluate=False, **kwargs):
        actions, metrics = original(self, observation, evaluate=evaluate, **kwargs)
        return np.full_like(actions, 7 if evaluate else 0), metrics

    monkeypatch.setattr(DistributionalAgent, 'act', labelled)
    vector = FrontierVector()
    monkeypatch.setattr(dqn_training, 'MeasuredSubprocVecEnv', lambda *a, **k: vector)
    run = dqn_training.train_dqn(config, 24)
    ledger = [json.loads(line) for line in (run / 'collector-steps.jsonl').read_text().splitlines()]
    attempts = [json.loads(line) for line in (run / 'attempts.jsonl').read_text().splitlines()]
    probed = {a['worker_episode'] for a in attempts if a['start_kind'] == 'frontier_probe'}
    assert probed == {2, 4, 6, 8}
    assert all(r['actions'][0] == (7 if r['worker_episodes'][0] in probed else 0) for r in ledger)
    status = json.loads((run / 'status.json').read_text())
    assert status['frontier_probe_episodes_this_run'] == 4
    progress = json.loads((tmp_path / 'progress.json').read_text())
    assert progress['assisted_attempts'] == 4 and progress['probes'] == [False, False]
