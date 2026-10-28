import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smg_rl import human_recorder as recorder
from smg_rl.profiles import PROFILES


def telemetry(complete=False, dead=False, scenario=1, nerve=None):
    data = {'position': (1., 2., 3.), 'scenario': scenario, 'stage': PROFILES['RMGE01'].stage,
            'dead': dead, 'complete': complete, 'scene_nerve': nerve or PROFILES['RMGE01'].playing,
            'scene_frame': 5, 'health': 3, 'grounded': True}
    return SimpleNamespace(**data, to_dict=lambda: data.copy())


def test_only_full_start_real_terminal_can_be_accepted():
    success = telemetry(complete=True, nerve=PROFILES['RMGE01'].complete[0])
    assert recorder.confirmed_star(success, True, 1)
    assert not recorder.confirmed_star(success, False, 1)
    assert not recorder.confirmed_star(success, True, 0)
    assert not recorder.confirmed_star(telemetry(complete=True), True, 1)
    assert not recorder.confirmed_star(telemetry(complete=True, dead=True, nerve=PROFILES['RMGE01'].complete[0]), True, 1)
    assert not recorder.confirmed_star(telemetry(complete=True, scenario=2, nerve=PROFILES['RMGE01'].complete[0]), True, 1)


def test_simultaneous_controls_are_recorded_without_forcing_exclusive_actions():
    assert recorder.controls_mask(('w', 'jump')) == (1 << recorder.CONTROLS.index('w')) | (1 << recorder.CONTROLS.index('jump'))
    with pytest.raises(ValueError):
        recorder.controls_mask(('secret',))


def test_input_is_sampled_from_documented_physical_buttons_and_focus_is_required():
    pressed = {recorder.KEYS['w'], recorder.KEYS['jump'], recorder.KEYS['spin']}
    api = SimpleNamespace(GetAsyncKeyState=lambda key: 0x8000 if key in pressed else 0)
    assert recorder.pressed_controls(api) == ('w', 'jump', 'spin')
    assert recorder.window_active(SimpleNamespace(FindWindow=lambda *_: 12, GetForegroundWindow=lambda: 12))
    assert not recorder.window_active(SimpleNamespace(FindWindow=lambda *_: 12, GetForegroundWindow=lambda: 13))


def test_discard_only_owned_attempts_inside_pending_root(tmp_path):
    root = tmp_path / 'pending'
    target = root / 'attempt-000001'
    target.mkdir(parents=True)
    (target / '.human-recorder-owned').write_text('owned')
    (target / 'partial').write_text('failed')
    recorder.discard_attempt(target, root)
    assert not target.exists()
    other = tmp_path / 'important'
    other.mkdir()
    (other / 'keep').write_text('data')
    with pytest.raises(ValueError):
        recorder.discard_attempt(other, root)
    assert (other / 'keep').read_text() == 'data'


def test_failed_attempt_aborts_encoder_and_deletes_only_owned_files(tmp_path, monkeypatch):
    calls = []

    class FakeVideo:
        def __init__(self, path, rgba):
            Path(path).write_bytes(b'partial video')

        def abort(self):
            calls.append('aborted')

    monkeypatch.setattr(recorder, 'VideoWriter', FakeVideo)
    backend = SimpleNamespace(image=lambda: np.zeros((84, 84), dtype=np.uint8),
                              bridge=SimpleNamespace(image=lambda: np.zeros((10, 10, 4), dtype=np.uint8)))
    pending = tmp_path / 'pending'
    attempt = recorder.Attempt(pending, 1, backend, telemetry())
    attempt.discard()
    assert calls == ['aborted'] and not list(pending.iterdir())


def test_synthetic_promotion_requires_star_and_records_complete_files(tmp_path, monkeypatch):
    class FakeVideo:
        def __init__(self, path, rgba):
            self.path = Path(path)
            self.frames = 0

        def write(self, rgba):
            self.frames += 1

        def close(self):
            self.path.write_bytes(b'fixture video')

    monkeypatch.setattr(recorder, 'VideoWriter', FakeVideo)
    monkeypatch.setattr(recorder.cv2, 'VideoCapture', lambda _: SimpleNamespace(
        read=lambda: (True, np.zeros((10, 10, 3), dtype=np.uint8)), isOpened=lambda: True, release=lambda: None))
    backend = SimpleNamespace(image=lambda: np.zeros((84, 84), dtype=np.uint8),
                              bridge=SimpleNamespace(image=lambda: np.zeros((10, 10, 4), dtype=np.uint8)))
    pending = tmp_path / 'pending'
    attempt = recorder.Attempt(pending, 1, backend, telemetry())
    attempt.record(('w', 'jump'), backend, telemetry(complete=True, nerve=PROFILES['RMGE01'].complete[0]))
    output = attempt.promote(tmp_path / 'completed', True, 'a' * 64)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['full_start_ram_star'] and manifest['samples'] == 1
    assert not manifest['training_compatible_without_conversion']
    assert (output / 'observations.gray8').stat().st_size == 2 * 84 * 84
    assert json.loads((output / 'inputs.jsonl').read_text())['controls'] == ['w', 'jump']


def test_incomplete_attempt_cannot_be_promoted(tmp_path, monkeypatch):
    class FakeVideo:
        def __init__(self, path, rgba):
            self.path, self.frames = Path(path), 1

        def close(self):
            self.path.write_bytes(b'fixture video')

    monkeypatch.setattr(recorder, 'VideoWriter', FakeVideo)
    backend = SimpleNamespace(image=lambda: np.zeros((84, 84), dtype=np.uint8),
                              bridge=SimpleNamespace(image=lambda: np.zeros((10, 10, 4), dtype=np.uint8)))
    attempt = recorder.Attempt(tmp_path / 'pending', 1, backend, telemetry())
    with pytest.raises(ValueError, match='valid full-start'):
        attempt.promote(tmp_path / 'completed', True, 'fixture-hash')
    assert not (tmp_path / 'completed').exists()
