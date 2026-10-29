"""Boundaries and immutable-source checks, without emulators or learning."""

import importlib.util
import json
from pathlib import Path

import pytest

from smg_rl.checkpoints import digest

spec = importlib.util.spec_from_file_location('iqn_day', Path(__file__).parents[1] / 'scripts/run_iqn_day.py')
day = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day)


def test_day_budget_preserves_cleanup_and_never_extends_deadline():
    assert day.block_budget(0, 86400) == 300
    assert day.block_budget(1, 86000) == 900
    assert day.block_budget(2, 85000) == 1800
    assert day.block_budget(48, 250) == 70
    assert day.block_budget(48, 180) == 0
    assert day.block_budget(48, -10) == 0
    for remaining in (181, 239, 300, 1200, 2500):
        assert day.block_budget(8, remaining) + 180 <= remaining


def test_day_source_checks_both_original_files(tmp_path):
    (tmp_path / 'agent.pt').write_bytes(b'original agent')
    (tmp_path / 'replay.npz').write_bytes(b'original replay')
    hashes = {'agent_sha256': digest(tmp_path / 'agent.pt'),
              'replay_sha256': digest(tmp_path / 'replay.npz')}
    (tmp_path / 'manifest.json').write_text(json.dumps(hashes))
    assert day.assert_source(tmp_path, hashes) == hashes
    (tmp_path / 'replay.npz').write_bytes(b'changed replay')
    with pytest.raises(ValueError, match='Retained source changed'):
        day.assert_source(tmp_path, hashes)


def test_day_pool_growth_uses_bytes_and_allows_reclamation():
    before = {'tags': [{'paged_bytes': 1024, 'nonpaged_bytes': 2048}]}
    after = {'tags': [{'paged_bytes': 512, 'nonpaged_bytes': 1536}]}
    assert day.pool_growth(before, after) == -1024
