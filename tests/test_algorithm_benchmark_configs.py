from pathlib import Path

from smg_rl.config import load_config
from smg_rl.dqn import DistributionalAgent

ROOT = Path(__file__).resolve().parents[1] / 'config'
BTR_ONLY = {'spectral_norm', 'munchausen', 'alpha', 'quantiles'}


def arms():
    return (load_config(ROOT / 'algorithm-benchmark-f-iqn-20261106.yaml'),
            load_config(ROOT / 'algorithm-benchmark-btr-20261106.yaml'))


def test_benchmark_arms_differ_only_in_the_algorithm_package_and_run_paths():
    control, btr = arms()
    skip = {'paths', 'study', 'dqn'}
    assert {k: v for k, v in control.items() if k not in skip} == {k: v for k, v in btr.items() if k not in skip}
    assert control['training']['seed'] == btr['training']['seed']
    changed = {k for k in control['dqn'].keys() | btr['dqn'].keys() if control['dqn'].get(k) != btr['dqn'].get(k)}
    assert changed == BTR_ONLY | {'learning_rate', 'epsilon', 'epsilon_start', 'epsilon_decay_decisions'}
    assert control['dqn']['gamma'] == btr['dqn']['gamma'] and control['dqn']['n_step'] == btr['dqn']['n_step']
    assert control['paths']['runs'] != btr['paths']['runs'] and control['paths']['replay'] != btr['paths']['replay']
    assert not control['curriculum']['enabled'] and 'demonstration' not in control and 'demonstration' not in btr
    assert 'source_agent_sha256' not in control['study'] and 'source_agent_sha256' not in btr['study']


def test_btr_arm_builds_an_agent_with_the_new_learning_options():
    import torch
    _, btr = arms()
    signature = {'image_size': 84, 'scenario_names': ['sweet_sweet'], 'action_names': list('01234567')}
    agent = DistributionalAgent(signature, btr['dqn'])
    assert type(agent.optimizer) is torch.optim.Adam and agent.settings['target_update_interval'] == 500
    assert any('parametrizations' in name for name, _ in agent.online.conv.named_parameters())
    assert agent.epsilon() == 1.0 and btr['dqn']['epsilon'] == 0.01


def test_bbf_lite_arm_differs_from_arm_f_only_in_the_learner_settings():
    control = load_config(ROOT / 'algorithm-benchmark-f-iqn-20261106.yaml')
    bbf = load_config(ROOT / 'algorithm-benchmark-bbf-lite-20261108.yaml')
    skip = {'paths', 'study', 'dqn'}
    assert {k: v for k, v in control.items() if k not in skip} == {k: v for k, v in bbf.items() if k not in skip}
    changed = {k for k in control['dqn'].keys() | bbf['dqn'].keys() if control['dqn'].get(k) != bbf['dqn'].get(k)}
    assert changed == {'channels', 'noisy', 'weight_decay', 'target_soft_tau', 'reset'}
    assert bbf['dqn']['train_every'] == control['dqn']['train_every'] and bbf['dqn']['gamma'] == control['dqn']['gamma']
    agent = DistributionalAgent({'image_size': 84, 'scenario_names': ['sweet_sweet'], 'action_names': list('01234567')},
                                bbf['dqn'])
    assert type(agent.optimizer).__name__ == 'AdamW' and agent.online.conv[0].out_channels == 64


def test_idle_penalty_arm_differs_from_arm_f_only_in_idle_handling_and_builds(tmp_path):
    from smg_rl.env import GalaxyEnv
    control = load_config(ROOT / 'algorithm-benchmark-f-iqn-20261106.yaml')
    idle = load_config(ROOT / 'algorithm-benchmark-f-idle-20261108.yaml')
    assert idle['dqn'] == control['dqn'] and idle['training'] == control['training'] and idle['parallel'] == control['parallel']
    keys = {k for k in control['regional_reward'].keys() | idle['regional_reward'].keys()
            if control['regional_reward'].get(k) != idle['regional_reward'].get(k)}
    assert keys == {'waypoint_lookahead', 'max_idle_frames', 'timeout_mode'}
    assert {k for k in control['rewards'] if control['rewards'][k] != idle['rewards'][k]} == {'revision', 'timeout'}
    assert idle['rewards']['timeout'] == idle['rewards']['death'] == -2.0
    assert control['rewards']['timeout'] == 0.0 and control['regional_reward']['timeout_mode'] == 'truncation_bootstrap'
    GalaxyEnv(idle, backend=None, run_dir=tmp_path, memory=object(), intrinsic=False)
