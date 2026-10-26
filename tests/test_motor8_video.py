from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from smg_rl.actions import action_name, action_set, advance_action
from smg_rl.config import load_config
from smg_rl.motor_transfer import project_motor
from smg_rl.policy_video import VideoBackend
from smg_rl.replay import replay_signature
from smg_rl.world_model import ImaginedAgent


def test_eight_actions_are_exclusive_and_checkpoint_projection_is_explicit():
    torch.set_num_threads(1)
    config = load_config('config/staged-smg1-control.yaml')
    settings = {'deterministic_size':8,'stochastic_size':2,'embedding_size':4,'hidden_size':8,'channels':2}
    config['environment']['image_size'] = 16
    source = ImaginedAgent(replay_signature(config),settings)
    config['actions'] = {'profile':'motor8-v1'}
    actions = action_set(config)
    assert len(actions) == 8 and all(len(a) <= 1 for a in actions)
    assert [action_name(i,config) for i in range(8)] == ['none','d','a','w','s','jump','spin','crouch']
    result = project_motor(source,replay_signature(config),settings)
    probabilities = result.actor(torch.zeros(1,result.world.features)).softmax(-1)[0]
    torch.testing.assert_close(probabilities[:5].sum(),torch.tensor(0.8))
    assert result.world.actions == 8 and source.world.actions == 54
    assert 'motor8-v1' in replay_signature(config)['action_execution']
    calls = []
    backend = SimpleNamespace(config=config,bridge=object(),advance=lambda names:calls.append((config['parallel']['frame_skip'],names)))
    advance_action(backend,('jump',),config)
    assert calls == [(8,('jump',)),(4,())]


def test_video_preserves_actual_policy_tile_and_displays_applied_release():
    backend = SimpleNamespace(bridge=SimpleNamespace(image=lambda:np.zeros((100,100,4),np.uint8)))
    video = VideoBackend(backend,Path('unused.mp4'),'test')
    video.policy_image = np.arange(84*84,dtype=np.uint8).reshape(84,84)
    video.policy_probabilities = np.full(8,0.125)
    video.last_action, video.last_controls = 'jump', ('jump',)
    pressed = video.compose_frame()
    expected = cv2.resize(video.policy_image,(252,252),interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(pressed[36:288,114:366,0],expected)
    video.last_controls = ()  # Final four emulated frames release the jump button.
    released = video.compose_frame()
    assert np.array_equal(pressed[36:288,114:366],released[36:288,114:366])
    assert not np.array_equal(pressed[329:351,114:424],released[329:351,114:424])
    assert video.last_action == 'jump'  # Selected action and physical input are distinct.


def test_four_frame_video_keeps_logical_jump_and_actual_carried_controls():
    config = load_config('config/regional-smg1-4.yaml')
    calls = []
    backend = SimpleNamespace(bridge=object(), config=config,
        advance=lambda controls: calls.append(controls), memory=SimpleNamespace(sample=lambda: None))
    video = VideoBackend(backend, Path('unused.mp4'), 'timing fixture', decision_frames=4)
    video.frame = lambda: None
    advance_action(video, ('w',), config)
    calls.clear()
    advance_action(video, ('jump',), config)
    assert calls == [('w', 'jump'), ('w', 'jump')]
    assert video.last_frame_count == 4
    assert video.last_action == 'jump' and video.last_controls == ('w', 'jump')
