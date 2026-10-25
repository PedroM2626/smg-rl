import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smg_rl.actions import ACTIONS, action_set, advance_action
from smg_rl.config import load_config
from smg_rl.memory import Telemetry
from smg_rl.motor_transfer import project_motor
from smg_rl.replay import replay_signature
from smg_rl.route import RouteProgress
from smg_rl.world_model import ImaginedAgent


def test_motor_profile_is_versioned_and_releases_emulated_buttons():
    config = load_config('config/staged-smg1-control.yaml')
    original = replay_signature(config)
    assert len(action_set(config)) == 54
    config['actions'] = {'profile': 'motor16-v1'}
    signature = replay_signature(config)
    assert len(signature['action_names']) == 16
    assert signature != original and 'action_execution' in signature
    calls = []
    backend = SimpleNamespace(config=config, bridge=object(), advance=lambda names: calls.append((config['parallel']['frame_skip'], names)))
    advance_action(backend, ('w', 'jump'), config)
    assert calls == [(8, ('w', 'jump')), (4, ('w',))]
    assert backend.last_frame_count == config['parallel']['frame_skip'] == 12
    calls.clear()
    advance_action(backend, ('w',), config)
    assert calls == [(12, ('w',))]


def test_route_requires_order_contact_and_health_without_repeat_credit():
    spec = json.loads(Path('config/routes/smg1-initial-exit.json').read_text())
    route = RouteProgress(spec)
    start = Telemetry(tuple(spec['waypoints'][0]), 1, 'BeltConveyerExGalaxy', 1, 2, 3, 0,
                      False, False, health=3, grounded=True)
    route.reset(start)
    last = replace(start, position=tuple(spec['waypoints'][-1]))
    assert route.update(last) == 0 and not route.exited  # Cannot skip the corner.
    first = replace(start, position=tuple(spec['waypoints'][1]))
    assert route.update(replace(first, grounded=False)) == 0
    assert route.update(replace(first, health=2)) == 0
    assert route.update(first) > 0
    second = replace(start, position=tuple(spec['waypoints'][2]))
    assert route.update(second) > 0
    assert route.update(last) > 0 and not route.exited
    assert route.update(last) == 0 and route.exited
    route.reset(last)  # An assisted teleport never grants a route reward.
    assert route.update(last) == 0 and not route.exited
    broken = {**spec, 'requires_grounded': False}
    with pytest.raises(ValueError):
        RouteProgress(broken)


def test_critic_only_adaptation_preserves_actor_world_and_real_count():
    torch.set_num_threads(1)
    settings = {'deterministic_size':8, 'stochastic_size':2, 'embedding_size':4,
                'hidden_size':8, 'channels':2, 'imagination_horizon':2, 'imagination_batch':4}
    agent = ImaginedAgent({'image_size':16, 'scenario_names':['test'],
                          'action_names':[str(i) for i in ACTIONS]}, settings)
    actor = {k:v.clone() for k,v in agent.actor.state_dict().items()}
    world = {k:v.clone() for k,v in agent.world.state_dict().items()}
    critic = {k:v.clone() for k,v in agent.critic.state_dict().items()}
    agent.update_imagination(agent.world.initial(4), torch.ones(4,1), train_actor=False)
    assert agent.actor_updates == agent.real_decisions == agent.model_updates == 0
    assert agent.imagined_transitions == 8
    assert all(torch.equal(v, actor[k]) for k,v in agent.actor.state_dict().items())
    assert all(torch.equal(v, world[k]) for k,v in agent.world.state_dict().items())
    assert any(not torch.equal(v, critic[k]) for k,v in agent.critic.state_dict().items())


def test_motor_transfer_projects_action_columns_but_resets_policy_value_and_counts():
    torch.set_num_threads(1)
    config = load_config('config/staged-smg1-control.yaml')
    config['environment']['image_size'] = 16
    settings = {'deterministic_size':8, 'stochastic_size':2, 'embedding_size':4,
                'hidden_size':8, 'channels':2}
    source = ImaginedAgent(replay_signature(config), settings)
    source.real_decisions = 123
    original = source.world.gru.weight_ih.clone()
    config['actions'] = {'profile':'motor16-v1'}
    signature = replay_signature(config)
    result = project_motor(source, signature, settings)
    indexes = [source.signature['action_names'].index(n) for n in signature['action_names']]
    columns = [0,1] + [2+i for i in indexes] + [56]
    torch.testing.assert_close(result.world.gru.weight_ih, original[:,columns])
    torch.testing.assert_close(source.world.gru.weight_ih, original)
    probabilities = result.actor(torch.randn(4,result.world.features)).softmax(-1)
    torch.testing.assert_close(probabilities[:,:9].sum(-1), torch.full((4,),0.8))
    assert result.real_decisions == result.model_updates == result.actor_updates == 0
    assert result.world.reward_low == result.world.reward_high == 0
