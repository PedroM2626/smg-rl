import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'manage_states', Path(__file__).resolve().parents[1] / 'scripts' / 'manage_states.py')
manage_states = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manage_states)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setitem(manage_states.BUNDLES, 'test', {
        'scope': 'test', 'files': ['data/a.s01', '.local/bank/bank.json']})
    source = tmp_path / 'source'
    (source / 'data').mkdir(parents=True)
    (source / '.local/bank').mkdir(parents=True)
    (source / 'data/a.s01').write_bytes(b'state' * 1000)
    (source / '.local/bank/bank.json').write_text('{}')
    out = tmp_path / 'test.tar'
    manage_states.pack('test', source, out)
    return out


def test_round_trip_verifies_and_skips_identical_files(tmp_path, bundle):
    target = tmp_path / 'target'
    target.mkdir()
    manifest, installed = manage_states.unpack(bundle, target)
    assert manifest['bundle'] == 'test' and sorted(installed) == ['.local/bank/bank.json', 'data/a.s01']
    assert (target / 'data/a.s01').read_bytes() == b'state' * 1000
    assert manage_states.unpack(bundle, target)[1] == []
    (target / 'data/a.s01').write_bytes(b'local')
    with pytest.raises(FileExistsError):
        manage_states.unpack(bundle, target)
    assert (target / 'data/a.s01').read_bytes() == b'local'
    assert manage_states.unpack(bundle, target, force=True)[1] == ['data/a.s01']


def test_verify_only_installs_nothing(tmp_path, bundle):
    target = tmp_path / 'target'
    target.mkdir()
    assert manage_states.unpack(bundle, target, check_only=True)[1] == []
    assert not (target / 'data').exists()


def _forged(tmp_path, manifest, members):
    path = tmp_path / 'forged.tar'
    with tarfile.open(path, 'w') as archive:
        for name, data in [('manifest.json', json.dumps(manifest).encode())] + members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.parametrize('name', ['../escape', '/abs/x', 'C:/x', 'docs/x', 'data\\..\\x'])
def test_unsafe_destinations_are_rejected(tmp_path, name):
    forged = _forged(tmp_path, {'schema': 1, 'bundle': 'x', 'files': [
        {'path': name, 'size': 1, 'sha256': '0'}]}, [(name, b'x')])
    with pytest.raises(ValueError, match='Unsafe'):
        manage_states.unpack(forged, tmp_path)


def test_checksum_mismatch_and_extra_members_install_nothing(tmp_path):
    good = manage_states.hashlib.sha256(b'x').hexdigest()
    entry = {'path': 'data/a.s01', 'size': 1, 'sha256': good}
    target = tmp_path / 'target'
    target.mkdir()
    bad = _forged(tmp_path, {'schema': 1, 'bundle': 'x', 'files': [entry]}, [('data/a.s01', b'y')])
    with pytest.raises(ValueError, match='Checksum'):
        manage_states.unpack(bad, target)
    extra = _forged(tmp_path, {'schema': 1, 'bundle': 'x', 'files': [entry]},
                    [('data/a.s01', b'x'), ('data/b.s01', b'x')])
    with pytest.raises(ValueError, match='Unexpected'):
        manage_states.unpack(extra, target)
    assert not (target / 'data').exists()


def test_published_bundles_cover_configured_human_bank():
    files = manage_states.BUNDLES['smg-rl-states-smg1-human-bank']['files']
    bank = Path(manage_states.HUMAN_BANK)
    assert str(bank / 'bank.json').replace('\\', '/') in files
    assert len([f for f in files if f.endswith('.state')]) == 11
    for name in manage_states.BUNDLES:
        for path in manage_states.BUNDLES[name]['files']:
            manage_states.safe_destination(path)
