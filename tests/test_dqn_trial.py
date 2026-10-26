import json

import pytest

from smg_rl import dqn_trial
from smg_rl.diagnostics import atomic_json


def test_review_is_idempotent_and_never_recharges_an_already_reviewed_budget():
    job = {'phase': 'completed_reviewed', 'review': {'passed': True}}
    state = {'games': {'smg1': {'elapsed_training_seconds': 123.}}}
    assert dqn_trial.apply_review(state, 'smg1', job) == {'passed': True}
    assert state['games']['smg1']['elapsed_training_seconds'] == 123.


def test_atomic_dispatch_state_failure_prevents_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(dqn_trial, 'STATE', tmp_path / 'state.json')
    monkeypatch.setattr(dqn_trial, 'atomic_json', lambda *a, **k: False)
    with pytest.raises(OSError, match='successor'):
        dqn_trial.save({'phase': 'pending'})
    assert not (tmp_path / 'state.json').exists()


def test_existing_trial_cannot_start_a_duplicate_health_job(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'
    atomic_json(path, {'phase': 'health_running'})
    monkeypatch.setattr(dqn_trial, 'STATE', path)
    with pytest.raises(ValueError, match='duplicating'):
        dqn_trial.start_health()


def test_run_discovery_rejects_stdout_outside_owned_root(tmp_path, monkeypatch):
    log = tmp_path / 'stdout.log'
    log.write_text(json.dumps({'run': str(tmp_path / 'unrelated' / 'dqn-test')}))
    monkeypatch.setattr(dqn_trial, 'load_config', lambda path: {'paths': {'runs': str(tmp_path / 'owned')}})
    with pytest.raises(ValueError, match='outside'):
        dqn_trial.locate_run({'stdout': str(log), 'config': 'fixture'})


def test_only_an_active_unreviewed_health_phase_can_dispatch_main(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'
    atomic_json(path, {'phase': 'main_running'})
    monkeypatch.setattr(dqn_trial, 'STATE', path)
    with pytest.raises(ValueError, match='once-reviewed'):
        dqn_trial.supervise()
