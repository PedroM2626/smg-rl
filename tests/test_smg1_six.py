import pytest

from scripts import run_smg1_six
from smg_rl.config import load_config
from smg_rl.replay import replay_signature


def test_six_smg1_collectors_preserve_the_original_policy_task_identity():
    original = load_config('config/impala-iqn-smg1.yaml')
    focused = load_config('config/smg1-six-two-hour.yaml')
    assert focused['parallel']['workers'] == 6
    assert focused['game']['id'] == 'RMGE01' and len(focused['variations']) == 1
    assert focused['dqn'] == original['dqn']
    assert replay_signature(focused) == replay_signature(original)
    assert focused['paths']['runs'] != original['paths']['runs']


def test_existing_six_worker_phase_cannot_dispatch_again(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'
    path.write_text('{}')
    monkeypatch.setattr(run_smg1_six, 'STATE', path)
    monkeypatch.setattr(run_smg1_six, 'STOP', tmp_path / 'STOP')
    with pytest.raises(ValueError, match='duplicate'):
        run_smg1_six.supervise()
