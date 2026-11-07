import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from smg_rl.actions import action_set, advance_action, execution_signature
from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.curriculum import BackwardCurriculum, CurriculumBank
from smg_rl.dqn_replay import PrioritizedFrames
from smg_rl.replay import replay_signature
from smg_rl.rewards import RegionalDistanceAdvance, validate_discounted_objective
from smg_rl.route import RouteProgress


def test_carried_jump_changes_only_direction_with_the_same_12_frame_pulse():
    config = load_config('config/iqn-day-smg1-20261029.yaml')
    carry = deepcopy(config)
    carry['actions']['profile'] = 'motor8-carry-v1'
    assert action_set(config) == action_set(carry) and len(action_set(carry)) == 8
    assert execution_signature(config) != execution_signature(carry)
    calls = []
    backend = SimpleNamespace(config=carry, bridge=object(),
        advance=lambda names: calls.append((carry['parallel']['frame_skip'], names)))
    advance_action(backend, ('w',), carry)
    advance_action(backend, ('jump',), carry)
    assert calls[-2:] == [(8, ('w', 'jump')), (4, ('w',))]
    advance_action(backend, (), carry)
    advance_action(backend, ('jump',), carry)
    assert calls[-2:] == [(8, ('jump',)), (4, ())]


def test_human_anchor_pays_no_loading_or_stationary_prefix_credit():
    specification = {'schema': 5, 'waypoints': [[0, 0, 0], [100, 0, 0], [200, 0, 0]],
        'waypoint_radius': 10, 'corridor_radius': 10, 'height_tolerance': 10,
        'landing_radius': 20, 'max_sample_distance': 150, 'requires_grounded': True,
        'exit_confirmations': 1}
    route = RouteProgress(specification)
    telemetry = SimpleNamespace(position=(100, 0, 0), health=3, grounded=True, dead=False, complete=False)
    with pytest.raises(ValueError, match='assisted'):
        route.reset(telemetry, segment=1)
    route.reset_human_anchor(telemetry, 1)
    assert route.best == .5
    reward = RegionalDistanceAdvance(17)
    reward.reset(route.best)
    assert route.update(telemetry) == 0
    assert reward.update(route.best, 12) == 0
    telemetry.position = (200, 0, 0)
    assert route.update(telemetry) == .5
    assert reward.update(route.best, 12) == 8.5
    telemetry.position = (101, 0, 0)
    with pytest.raises(ValueError, match='exactly'):
        route.reset_human_anchor(telemetry, 1)


def test_eight_label_longjump_profile_preserves_pulses_and_requires_preparation():
    config = load_config('config/iqn-day-smg1-20261029.yaml')
    config['actions']['profile'] = 'motor8-longjump-v1'
    calls = []
    backend = SimpleNamespace(config=config, bridge=object(),
        advance=lambda names: calls.append((config['parallel']['frame_skip'], names)))
    advance_action(backend, ('w',), config)
    advance_action(backend, ('crouch',), config)
    assert calls[-1] == (12, ('w', 'crouch'))
    advance_action(backend, ('jump',), config)
    assert calls[-2:] == [(8, ('w', 'crouch', 'jump')), (4, ('w', 'crouch'))]
    advance_action(backend, (), config)
    advance_action(backend, ('jump',), config)
    assert calls[-2:] == [(8, ('jump',)), (4, ())]
    assert len(action_set(config)) == 8


def test_cut_bootstrap_is_distinct_from_real_death_and_signed():
    settings = {'timeout_mode': 'truncation_bootstrap', 'scale': 17., 'gamma_per_frame': .997 ** (1 / 12)}
    weights = {'step': 0., 'death': -2., 'timeout': 0., 'completion': 20., 'damage': -.25}
    validate_discounted_objective(settings, weights, .997)
    with pytest.raises(ValueError):
        validate_discounted_objective(settings, {**weights, 'timeout': -2}, .997)
    config = load_config('config/iqn-day-smg1-20261029.yaml')
    revised = deepcopy(config)
    revised['regional_reward']['timeout_mode'] = 'truncation_bootstrap'
    assert replay_signature(config)['termination_revision'] == 'absorbing-frame-budget-v1'
    assert replay_signature(revised)['termination_revision'] == 'budget-truncation-bootstrap-v2'
    for terminal, expected in [(False, .997), (True, 0.)]:
        replay = PrioritizedFrames(4, 2, gamma=.997, n_step=3)
        first = replay.add_frame(np.zeros((2, 2), np.uint8))
        last = replay.add_frame(np.ones((2, 2), np.uint8), first)
        replay.append(0, first, 0, 0., last, 0, terminal, True)
        assert replay.arrays['discount'][0] == pytest.approx(expected)


def test_backward_frontier_requires_eight_of_ten_current_anchor_ram_stars(tmp_path):
    bank = tmp_path / 'bank.json'
    bank.write_text(json.dumps({'schema': 2, 'source_full_start_ram_star': True,
        'entries': [{'file': f'anchor-{i}.state'} for i in range(3)]}))
    progress = tmp_path / 'progress.json'
    scheduler = BackwardCurriculum(bank, digest(bank), progress)
    for i in range(9):
        scheduler.record('anchor-2.state', True, i)
    assert scheduler.state['frontier'] == 2
    scheduler.record('anchor-2.state', False, 10)
    assert scheduler.state['frontier'] == 1
    # Late completion by a worker from the old frontier cannot unlock an earlier one.
    for i in range(12):
        scheduler.record('anchor-2.state', True, i + 11)
    assert scheduler.state['frontier'] == 1 and scheduler.state['recent'] == []
    restored = BackwardCurriculum(bank, digest(bank), progress)
    assert restored.state == scheduler.state
    with pytest.raises(ValueError, match='verified anchor'):
        scheduler.record('unverified.state', True, 99)


def test_backward_bank_keeps_original_idle_and_fixture_rejections(tmp_path):
    state = tmp_path / 'anchor.state'
    state.write_bytes(b'engineering unit fixture, not a game state')
    entry = {'file': state.name, 'sha256': digest(state), 'scenario': 1,
        'origin_slot': 1, 'near_star_fixture': False, 'accepted': True,
        'route_waypoint': 1, 'validation': {'passed': True, 'idle_frames': 60}}
    bank = tmp_path / 'bank.json'
    data = {'schema': 2, 'game_id': 'RMGE01', 'emulator_sha256': 'fixture',
        'source_full_start_ram_star': True, 'entries': [entry]}
    bank.write_text(json.dumps(data))
    curriculum = CurriculumBank(bank, digest(bank), 'RMGE01', 'fixture', .5, tmp_path / 'progress.json')
    assert curriculum.choose(np.random.default_rng(2), 1) == entry
    entry['near_star_fixture'] = True
    bank.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='near-star'):
        CurriculumBank(bank, digest(bank), 'RMGE01', 'fixture', .5, tmp_path / 'progress.json')


def test_only_a_labeled_anchor_diagnostic_may_start_every_attempt_assisted(tmp_path):
    state = tmp_path / 'anchor.state'
    state.write_bytes(b'engineering unit fixture, not a game state')
    entry = {'file': state.name, 'sha256': digest(state), 'scenario': 1,
        'origin_slot': 1, 'near_star_fixture': False, 'accepted': True,
        'route_waypoint': 1, 'validation': {'passed': True, 'idle_frames': 60}}
    bank = tmp_path / 'bank.json'
    bank.write_text(json.dumps({'schema': 2, 'game_id': 'RMGE01', 'emulator_sha256': 'fixture',
        'source_full_start_ram_star': True, 'entries': [entry]}))
    with pytest.raises(ValueError, match='probability'):
        CurriculumBank(bank, digest(bank), 'RMGE01', 'fixture', 1., tmp_path / 'progress.json')
    diagnostic = CurriculumBank(bank, digest(bank), 'RMGE01', 'fixture', 1., tmp_path / 'progress.json',
                                anchor_diagnostic=True)
    rng = np.random.default_rng(3)
    assert all(diagnostic.choose(rng, 1) == entry for _ in range(50))


def test_hybrid_frontier_needs_greedy_probe_streak_and_robust_training_rate(tmp_path):
    bank = tmp_path / 'bank.json'
    bank.write_text(json.dumps({'schema': 2, 'source_full_start_ram_star': True,
        'entries': [{'file': f'anchor-{i}.state'} for i in range(3)]}))
    rule = {'mode': 'greedy_hybrid', 'probe_every': 10, 'probe_streak': 2,
            'robust_window': 10, 'robust_min': .2}
    scheduler = BackwardCurriculum(bank, digest(bank), tmp_path / 'progress.json', rule)
    for i in range(10):
        scheduler.record('anchor-2.state', i < 2, i)
    assert scheduler.state['frontier'] == 2  # Robust rate alone is not enough.
    scheduler.record_probe('anchor-2.state', True, 10)
    scheduler.record_probe('anchor-1.state', True, 11)  # Probes from another anchor never count.
    assert scheduler.state['frontier'] == 2
    scheduler.record_probe('anchor-2.state', False, 12)
    scheduler.record_probe('anchor-2.state', True, 13)
    assert scheduler.state['frontier'] == 2  # The streak must be consecutive.
    scheduler.record_probe('anchor-2.state', True, 14)
    assert scheduler.state['frontier'] == 1
    assert scheduler.state['probes'] == [] and scheduler.state['recent'] == []
    assert scheduler.state['events'][-1]['rule'] == 'greedy_hybrid'
    # Greedy competence alone cannot replace the robust training rate.
    for i in range(10):
        scheduler.record('anchor-1.state', i == 0, 20 + i)
    scheduler.record_probe('anchor-1.state', True, 30)
    scheduler.record_probe('anchor-1.state', True, 31)
    assert scheduler.state['frontier'] == 1
    restored = BackwardCurriculum(bank, digest(bank), tmp_path / 'progress.json', rule)
    assert restored.state == scheduler.state
    with pytest.raises(ValueError, match='hybrid frontier rule'):
        BackwardCurriculum(bank, digest(bank), tmp_path / 'progress.json').record_probe('anchor-1.state', True, 32)
    with pytest.raises(ValueError, match='Invalid hybrid'):
        BackwardCurriculum(bank, digest(bank), tmp_path / 'progress.json', {**rule, 'robust_min': 1})


def test_canonical_start_mix_is_off_by_default_and_draws_no_random_number():
    from smg_rl.curriculum import canonical_start
    rng = np.random.default_rng(5)
    state = rng.bit_generator.state
    assert canonical_start({}, {}, rng) is False
    assert canonical_start({'start_mix': {'full_start_fraction': 0.}}, {}, rng) is False
    assert rng.bit_generator.state == state
    assert canonical_start({}, {'full_start': True}, rng) is True


def test_canonical_start_mix_follows_its_fraction_and_rejects_bad_values():
    from smg_rl.curriculum import canonical_start
    rng = np.random.default_rng(7)
    mix = {'start_mix': {'full_start_fraction': .25}}
    share = np.mean([canonical_start(mix, {}, rng) for _ in range(4000)])
    assert .22 < share < .28
    assert all(canonical_start({'start_mix': {'full_start_fraction': 1.}}, {}, rng) for _ in range(20))
    for bad in (-.1, 1.5, True, 'half', None):
        with pytest.raises(ValueError, match='canonical-start fraction'):
            canonical_start({'start_mix': {'full_start_fraction': bad}}, {}, rng)
