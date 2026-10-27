from types import SimpleNamespace

import pytest

from smg_rl import resource_disk


def test_checkpoint_disk_reserve_applies_to_every_output_and_profile_volume():
    assert resource_disk.disk_safe({'runs': {'free_bytes': 8 * 2**30}, 'user': {'free_bytes': 4 * 2**30}})
    assert not resource_disk.disk_safe({'runs': {'free_bytes': 8 * 2**30}, 'user': {'free_bytes': 2 * 2**30}})


def test_preflight_checks_existing_ancestor_without_creating_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_disk.shutil, 'disk_usage', lambda _: SimpleNamespace(free=2**30, total=16 * 2**30))
    config = {'paths': {'runs': str(tmp_path / 'new/runs'), 'user': str(tmp_path / 'new/profile')}}
    with pytest.raises(OSError, match='disk headroom'):
        resource_disk.require_disk_headroom(config)
    assert not (tmp_path / 'new').exists()
