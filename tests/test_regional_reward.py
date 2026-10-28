import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smg_rl.actions import EIGHT_ACTIONS, advance_action, execution_signature
from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.curriculum import CurriculumBank
from smg_rl.env import GalaxyEnv
from smg_rl.memory import Telemetry
from smg_rl.replay import load_chunk, replay_signature
from smg_rl.rewards import RegionalPotential
from smg_rl.route import RouteProgress


@pytest.mark.parametrize('frames', [4, 12])
@pytest.mark.parametrize('start', [0., .4])
def test_potential_telescope_closes_failure_without_profitable_partial_progress(frames, start):
    shaping = RegionalPotential()
    shaping.reset(start)
    rewards, discount = [], 1.
    total = 0.
    for index, progress in enumerate([start, .6, .9, .9, .9]):
        amount = shaping.update(progress, frames, terminal=index == 4)
        rewards.append(amount)
        total += discount * amount
        discount *= shaping.gamma_per_frame ** frames
    assert total == pytest.approx(-start)
    assert sum(rewards) <= 0
    assert shaping.value == 0


def test_four_frame_held_actions_reproduce_legacy_pulses_without_extra_actions():
    old = load_config('config/motor8-smg1.yaml')
    calls = []
    backend = SimpleNamespace(config=old, bridge=object(), advance=lambda names: calls.append((old['parallel']['frame_skip'], names)))
    advance_action(backend, ('jump',), old)
    expected = [names for frames, names in calls for _ in range(frames)]
    new = {**old, 'actions': {'profile': 'motor8-timed-v2', 'decision_frames': 4,
                            'pulse_frames': 4, 'jump_direction': 'inertia'},
           'parallel': {**old['parallel'], 'frame_skip': 4}}
    calls.clear()
    backend.config = new
    backend.advance = lambda names: calls.append((new['parallel']['frame_skip'], names))
    for names in [('jump',), ('jump',), ()]:
        advance_action(backend, names, new)
    assert [names for frames, names in calls for _ in range(frames)] == expected
    assert len(EIGHT_ACTIONS) == 8
    assert execution_signature(old) != execution_signature(new)
    new['actions']['jump_direction'] = 'carry'
    advance_action(backend, ('w',), new)
    advance_action(backend, ('jump',), new)
    assert calls[-1] == (4, ('w', 'jump'))
    advance_action(backend, (), new)
    advance_action(backend, ('jump',), new)
    assert calls[-1] == (4, ('jump',))


class NativeFixture:
    def __init__(self):
        self.position = (0., -1150., 500.)
        self.steps = 0
        self.bridge = SimpleNamespace(diagnostics=lambda: {})
        self.last_frame_count = 4
        self.dead = False

    def load_slot(self, slot):
        self.position, self.dead, self.steps = (0., -1150., 500.), False, 0

    def advance(self, names):
        self.steps += 1
        if names == ('w',):
            self.position = (0., -1150., self.position[2] - 100)

    def sample(self):
        return Telemetry(self.position, 1, 'SecretAthleticGalaxy', 1, 1, 1, self.steps,
                         self.dead, False, grounded=not self.dead)

    def image(self):
        return np.full((84, 84), self.steps, np.uint8)

    def close(self):
        pass


def regional_environment(tmp_path):
    config = load_config()
    config['parallel'] = {'backend': 'native', 'frame_skip': 4}
    config['actions'] = {'profile': 'motor8-timed-v2', 'decision_frames': 4,
                         'pulse_frames': 4, 'jump_direction': 'inertia'}
    spec = {'schema': 3, 'game_id': 'SB4E01', 'waypoints': [[0, -1150, 500], [0, -1150, -500]],
            'waypoint_radius': 150, 'corridor_radius': 200, 'height_tolerance': 100,
            'exit_confirmations': 2, 'requires_grounded': True, 'requires_full_health': False}
    route = tmp_path / 'route.json'
    route.write_text(json.dumps(spec))
    config['paths']['route'] = str(route)
    config['rewards'].update(progress=0., intrinsic=0., step=-.001, death=-2., timeout=-2.)
    gamma = .997 ** (1 / 12)
    config['regional_reward'] = {'enabled': True, 'spec_sha256': digest(route), 'scale': 1.,
        'gamma_per_frame': gamma, 'max_idle_frames': 8, 'max_episode_frames': 120}
    config['world_model'] = {'gamma': gamma ** 4}
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps({'validated': True, 'variations': {
        name: {'start': [0, -1150, 500], 'goal': [0, -450, -10000]}
        for name in ('spinning', 'double_time')}}))
    config['paths']['calibration'] = str(calibration)
    config['paths']['replay'] = str(tmp_path / 'replay')
    config['replay'] = {'enabled': True, 'chunk_steps': 256}
    backend = NativeFixture()
    backend.config = config
    env = GalaxyEnv(config, backend, tmp_path / 'world-model-fixture' / 'worker-00', backend, intrinsic=False)
    return env, backend


def test_regional_timeout_is_absorbing_and_replay_keeps_start_provenance(tmp_path):
    env, backend = regional_environment(tmp_path)
    try:
        env.reset(options={'variation': 0})
        _, _, terminated, _, info = env.step(3)
        assert not terminated and info['shaping_potential'] > 0
        env.step(0)
        _, _, terminated, truncated, info = env.step(0)
        assert terminated and not truncated and not info['dead']
        assert info['timeout_penalty'] == -2 and info['shaping_potential'] == 0
        chunk = load_chunk(next((tmp_path / 'replay').rglob('*.npz')), replay_signature(env.config))
        assert chunk['terminated'][-1] and not chunk['truncated'][-1]
        assert chunk['metadata']['start_kind'] == 'full_start'
    finally:
        env.close()


def test_regional_curriculum_reset_initializes_potential_without_teleport_credit(tmp_path):
    env, backend = regional_environment(tmp_path)
    try:
        env.reset(options={'variation': 0})
        backend._motor_direction = ('w',)
        env.reset(options={'variation': 0})
        assert backend._motor_direction == ()
        route = RouteProgress(json.loads(Path(env.config['paths']['route']).read_text()))
        route.reset(Telemetry((0, -1150, 300), 1, 'fixture', 1, 1, 1, 0,
                              False, False, grounded=True), segment=0)
        potential = RegionalPotential()
        potential.reset(route.best)
        assert potential.update(route.best, 4) < 0
    finally:
        env.close()


def test_curated_bank_rejects_goal_states_corruption_and_unverified_metadata(tmp_path):
    file = tmp_path / 'anchor.state'
    file.write_bytes(b'portable state fixture')
    entry = {'file': file.name, 'sha256': digest(file), 'accepted': True, 'origin_slot': 1,
             'near_star_fixture': False, 'validation': {'passed': True, 'idle_frames': 60},
             'scenario': 1}
    path = tmp_path / 'bank.json'
    data = {'schema': 1, 'game_id': 'fixture', 'emulator_sha256': 'emulator', 'entries': [entry]}
    path.write_text(json.dumps(data))
    bank = CurriculumBank(path, digest(path), 'fixture', 'emulator', .5)
    assert bank.state_path(entry) == file
    with pytest.raises(ValueError):
        CurriculumBank(path, digest(path), 'fixture', 'emulator', 1.)
    data['entries'][0]['origin_slot'] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='near-star'):
        CurriculumBank(path, digest(path), 'fixture', 'emulator', .5)
    file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='integrity'):
        bank.state_path(entry)


def test_schema3_keeps_grounded_progress_after_damage_and_rejects_airborne_shortcuts():
    spec = json.loads(Path('config/routes/smg1-opening-regional-v3.json').read_text())
    route = RouteProgress(spec)
    initial = Telemetry(tuple(spec['waypoints'][0]), 1, 'fixture', 1, 1, 1, 0,
                        False, False, health=3, grounded=True)
    route.reset(initial)
    moving = replace(initial, position=(0., 45., -100.), health=2)
    assert route.update(replace(moving, grounded=False)) == 0
    assert route.update(moving) > 0
    assert route.update(moving) == 0
    assert route.update(initial) == 0
    assert route.update(moving) == 0


def test_distance_normalization_does_not_pay_entire_opening_as_a_mission(tmp_path):
    env, _ = regional_environment(tmp_path)
    try:
        env.regional_settings['normalization_distance'] = 50000.
        env.reset(options={'variation': 0})
        env.step(3)
        assert env._potential_fraction() == pytest.approx(100 / 50000)
        assert env.potential.value == pytest.approx(.002)
    finally:
        env.close()


def test_incremental_environment_pays_real_new_progress_once_and_keeps_failure_credit(tmp_path):
    env, backend = regional_environment(tmp_path)
    env.close()
    config = env.config
    config['world_model']['gamma'] = 1.
    config['rewards'].update(step=0., death=-2., timeout=-2., completion=20., damage=-.25)
    config['regional_reward'].update(mode='advance', gamma_per_frame=1., scale=1.)
    config['regional_reward'].pop('normalization_distance', None)
    backend.closed = False
    env = GalaxyEnv(config, backend, tmp_path / 'simple', memory=backend, intrinsic=False)
    try:
        env.reset(options={'variation': 0})
        assert env.step(0)[1] == 0  # Stationary animation cannot produce credit.
        advance = env.step(3)[1]
        assert advance == pytest.approx(.1)
        assert env.step(0)[1] == 0
        backend.dead = True
        _, reward, terminated, truncated, info = env.step(0)
        assert terminated and not truncated and reward == -2.
        assert info['regional_shaping_reward'] == 0
        assert env.total_reward == pytest.approx(-1.9)
    finally:
        env.close()


def test_explicit_distance_advance_retains_full_route_and_old_density():
    from smg_rl.rewards import (
        RegionalAdvance,
        RegionalDistanceAdvance,
        validate_distance_objective,
        validate_simple_objective,
    )

    scale = 37872.5332017393 / 2186.1422975980586
    reward = RegionalDistanceAdvance(scale)
    reward.reset(0.)
    assert reward.update(2186.1422975980586 / 37872.5332017393, 12) == pytest.approx(1.)
    assert reward.update(1., 12) == pytest.approx(scale - 1.)
    assert reward.update(.5, 12, terminal=True) == 0.
    settings = {'scale': scale, 'gamma_per_frame': 1.}
    weights = {'step': 0., 'death': -2., 'timeout': -2., 'completion': 20., 'damage': -.25}
    validate_distance_objective(settings, weights, 1.)
    with pytest.raises(ValueError):
        validate_simple_objective(settings, weights, 1.)
    with pytest.raises(ValueError):
        RegionalAdvance(scale)
    with pytest.raises(ValueError):
        RegionalDistanceAdvance(21.)


def test_iqn_has_no_categorical_value_support():
    from smg_rl.distributional import DistributionalQ

    cfg = load_config('config/human-route-smg1-density-v1.yaml')
    net = DistributionalQ(replay_signature(cfg), cfg['dqn'])
    assert net.algorithm == 'iqn'
    assert not hasattr(net, 'support')


def test_density_profile_does_not_saturate_after_old_opening(tmp_path):
    cfg = load_config('config/human-route-smg1-density-v1.yaml')
    env = GalaxyEnv(cfg, backend=None, run_dir=tmp_path, memory=object(), intrinsic=False)
    env.route.best = 2186.1422975980586 / env.route.total_length
    env.potential.reset(0.)
    first = env.potential.update(env._potential_fraction(), 12)
    env.route.best = .5
    later = env.potential.update(env._potential_fraction(), 12)
    assert first == pytest.approx(1.)
    assert later > 7.
    assert cfg['regional_reward']['mode'] == 'distance_advance'
    assert replay_signature(cfg) != replay_signature(load_config('config/human-route-smg1-landing-v2.yaml'))
