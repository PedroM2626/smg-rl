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


def test_explicit_partial_recovery_preserves_source_and_charges_failed_time_once(tmp_path, monkeypatch):
    state_path = tmp_path / 'state.json'
    state = {'phase': 'needs_review', 'supervisor': {'pid': 1, 'created_time': 0}, 'games': {'smg1': {
        'real_decisions': 100, 'completed_attempts': 10, 'elapsed_training_seconds': 1000.,
        'error': 'original failure', 'jobs': [{'phase': 'needs_review', 'source_real': 100,
            'source_attempts': 10, 'config': 'fixture'}]}}}
    atomic_json(state_path, state)
    monkeypatch.setattr(dqn_trial, 'STATE', state_path)
    monkeypatch.setattr(dqn_trial, 'STOP', tmp_path / 'STOP')
    monkeypatch.setattr(dqn_trial, 'alive', lambda _: False)
    monkeypatch.setattr(dqn_trial, 'locate_run', lambda _: tmp_path / 'run')
    monkeypatch.setattr(dqn_trial, 'load_config', lambda _: {})
    def review(*args, failed=False):
        assert failed
        return {'real_decisions': 150, 'completed_attempts': 15, 'elapsed_seconds': 30.,
                'checkpoint': 'retained-checkpoint', 'csv_decision_gap': 1}
    monkeypatch.setattr(dqn_trial, 'audit_block', review)
    dqn_trial.recover()
    result = json.loads(state_path.read_text())
    assert result['phase'] == 'recovery_ready'
    branch = result['games']['smg1']
    assert branch['real_decisions'] == 150 and branch['completed_attempts'] == 15
    assert branch['elapsed_training_seconds'] == 1030.
    assert branch['jobs'][-1]['phase'] == 'failed_accounted'
    with pytest.raises(ValueError):
        dqn_trial.recover()
