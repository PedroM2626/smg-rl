import json

import pytest

from smg_rl.diagnostics import atomic_json
from smg_rl.exploration_trial import exclusive_state, online, review


def test_trial_cannot_skip_a_budget_or_review_and_never_auto_launches_successors(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'immutable.npz').write_bytes(b'fixture')
    case = {'name':'control', 'seed':11, 'state':'fit_ready', 'new_online_decisions':0,
            'checkpoint':'source.pt', 'runs':[], 'reviews':[],
            'config':{'root':str(tmp_path), 'paths':{'replay':str(tmp_path/'replay')},
                      'training':{'enabled':True}}}
    state = tmp_path / 'state.json'
    atomic_json(state, {'state':'awaiting_health_review', 'source_replay':str(source), 'cases':[case]})
    calls = []

    def train(config, steps, checkpoint):
        calls.append(steps)
        root = tmp_path / 'run'
        root.mkdir()
        atomic_json(root/'status.json', {'state':'completed','real_decisions':256,
                    'model_updates':0,'actor_updates':0,'imagined_transitions':0,'decisions_per_second':10})
        return root

    monkeypatch.setattr('smg_rl.exploration_trial.train_world_model', train)
    with pytest.raises(ValueError, match='previous'):
        online(state, 'control', 11, 2000)
    online(state, 'control', 11, 256)
    assert calls == [256]
    with pytest.raises(ValueError, match='needs review'):
        online(state, 'control', 11, 2000)
    review(state, 'control', 11, 'continue', 'Finite fixture completed; review supports only the next declared block.')
    result = json.loads(state.read_text())
    assert result['cases'][0]['state'] == 'reviewed'
    assert len(result['cases'][0]['reviews']) == 1
    assert next((tmp_path/'replay/source').glob('*.npz')).read_bytes() == b'fixture'
    assert (tmp_path/'replay/source/snapshot-complete.json').exists()
    assert calls == [256]


def test_experiment_lock_rejects_concurrent_dispatch(tmp_path):
    path = tmp_path/'state.json'

    @exclusive_state(0)
    def inner(state_path):
        return None

    @exclusive_state(0)
    def outer(state_path):
        with pytest.raises(RuntimeError, match='owns'):
            inner(state_path)

    outer(path)
    inner(path)  # Lock is released at the end of the bounded operation.
