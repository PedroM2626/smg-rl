import importlib

import pytest


def test_recovery_budget_leaves_shutdown_reserve(monkeypatch):
    monkeypatch.syspath_prepend('scripts')
    module = importlib.import_module('run_reward_hour_recovery2')
    assert module.segment_limit(1800) == 600
    assert module.segment_limit(3400) == pytest.approx(80)
    assert module.segment_limit(3471) == 0
    assert module.segment_limit(3602.7) == 0
