import json
from datetime import datetime
from types import SimpleNamespace

from smg_rl import train


def test_same_clock_timestamp_never_reuses_or_overwrites_an_existing_run(tmp_path, monkeypatch):
    frozen = datetime(2026, 10, 26, 8, 54, 5, 776756)
    monkeypatch.setattr(train, 'datetime', SimpleNamespace(now=lambda: frozen))
    config = {'paths': {'runs': str(tmp_path)}, 'purpose': 'timestamp collision fixture'}
    paths = [train.new_run(config, 'eval-dqn-video') for _ in range(3)]
    assert len(set(paths)) == 3
    for path in paths:
        assert path.name.startswith('eval-dqn-video-20261026-085405-776756')
        assert json.loads((path / 'config.json').read_text()) == config
    assert paths[0].name == 'eval-dqn-video-20261026-085405-776756'
