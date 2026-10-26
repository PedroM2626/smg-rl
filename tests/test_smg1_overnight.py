from datetime import datetime

import pytest

from scripts import run_smg1_overnight as overnight
from smg_rl.dqn_training import deadline_timestamp


def test_deadline_matches_sao_paulo_and_requires_timezone():
    assert deadline_timestamp('2026-10-27T07:00:00-03:00') == deadline_timestamp(overnight.DEADLINE)
    assert deadline_timestamp(None) is None
    with pytest.raises(ValueError, match='timezone'):
        deadline_timestamp('2026-10-27T07:00:00')


def test_remaining_time_clamps_at_deadline(monkeypatch):
    end = datetime.fromisoformat(overnight.DEADLINE).timestamp()
    monkeypatch.setattr(overnight.time, 'time', lambda: end - 10)
    assert overnight.remaining_seconds() == 10
    monkeypatch.setattr(overnight.time, 'time', lambda: end + 10)
    assert overnight.remaining_seconds() == 0


def test_supervisor_refuses_existing_dispatch_state(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'
    path.write_text('{}')
    monkeypatch.setattr(overnight, 'STATE', path)
    with pytest.raises(ValueError, match='never duplicate'):
        overnight.supervise()
