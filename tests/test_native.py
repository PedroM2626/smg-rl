import mmap
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from smg_rl.native import HEADER_BYTES, PIXEL_BYTES, PROTOCOL_VERSION, NativeBridge, NativeDolphin


def bridge():
    result = object.__new__(NativeBridge)
    result.mapping = mmap.mmap(-1, HEADER_BYTES + PIXEL_BYTES)
    result.timeout = 0.001
    return result


def test_shared_frame_rejects_invalid_shape_and_returns_owned_pixels():
    source = bridge()
    try:
        with pytest.raises(RuntimeError, match="valid RGBA"):
            source.image()
        for field, value in [(10, 2), (11, 1), (12, 8)]:
            source.set(field, value)
        source.mapping[HEADER_BYTES : HEADER_BYTES + 8] = bytes(range(8))
        image = source.image()
        source.mapping[HEADER_BYTES] = 99
        assert image.shape == (1, 2, 4)
        assert image[0, 0, 0] == 0
        source.set(12, 200)
        with pytest.raises(RuntimeError, match="valid RGBA"):
            source.image()
    finally:
        source.close()


def test_native_timeout_and_pending_request_are_explicit():
    source = bridge()
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            source.command(2, 12, ["w", "jump"])
        assert source.get(7) == 17
        with pytest.raises(RuntimeError, match="still pending"):
            source.command(2, 12)
    finally:
        source.close()


def test_compact_gray_frame_is_owned_and_does_not_accept_wrong_stride():
    source = bridge()
    try:
        for field, value in [(10, 2), (11, 2), (12, 2), (30, 1)]:
            source.set(field, value)
        source.mapping[HEADER_BYTES:HEADER_BYTES + 4] = bytes([0, 30, 60, 90])
        pixels = source.image()
        source.mapping[HEADER_BYTES] = 255
        assert pixels.shape == (2, 2) and pixels.tolist() == [[0, 30], [60, 90]]
        source.set(12, 8)
        with pytest.raises(RuntimeError, match="valid RGBA"):
            source.image()
    finally:
        source.close()


def test_legitimate_uniform_native_transition_frame_is_accepted():
    backend = object.__new__(NativeDolphin)
    backend.config = {"environment": {"image_size": 84}}
    backend.bridge = SimpleNamespace(image=lambda: np.zeros((84, 84), np.uint8))
    backend.capture()
    assert backend.last_image.shape == (84, 84)


def test_stale_reply_event_cannot_acknowledge_an_unfinished_command():
    source = bridge()
    source.timeout = 0.01
    source.events = SimpleNamespace(publish=lambda: None, wait=lambda ms: None, close=lambda: None)
    source.wakeup = "events"
    try:
        with pytest.raises(TimeoutError):
            source.command(2, 12)
        assert source.get(3) != source.get(4)
    finally:
        source.close()


def test_native_action_rejects_inexact_frame_acknowledgment():
    backend = object.__new__(NativeDolphin)
    backend.config = {"parallel": {"frame_skip": 12}}
    backend.process = SimpleNamespace(pid=100)
    backend.bridge = SimpleNamespace(get=lambda field: 11, command=lambda *args: None)
    with pytest.raises(RuntimeError, match="Expected 12 native frames, got 11"):
        backend.advance(["jump"])


def restore_backend(root):
    backend = object.__new__(NativeDolphin)
    backend.root = root
    backend.bridge = SimpleNamespace(get=lambda _: 1)
    backend.memory = SimpleNamespace(profile=SimpleNamespace(game_id='RMGE01'))
    backend.load_slot = lambda slot: None
    return backend


def test_restore_staging_preserves_initial_slot_once_without_accumulating_copies(tmp_path):
    backend = restore_backend(tmp_path / 'owned-profile')
    folder = backend.root / 'StateSaves'
    folder.mkdir(parents=True)
    slot = folder / 'RMGE01.s09'
    slot.write_bytes(b'original inherited slot')
    source = tmp_path / 'immutable-source.sav'
    source.write_bytes(b'immutable archive state')
    for _ in range(5):
        backend.load_state(source)
    assert slot.read_bytes() == source.read_bytes() == b'immutable archive state'
    retained = list(folder.glob('*.retained-*'))
    assert len(retained) == 1 and retained[0].read_bytes() == b'original inherited slot'
    assert not list(folder.glob('*.staging-*'))


def test_failed_restore_copy_leaves_original_slot_and_source_intact(tmp_path, monkeypatch):
    from smg_rl import native

    backend = restore_backend(tmp_path / 'owned-profile')
    folder = backend.root / 'StateSaves'
    folder.mkdir(parents=True)
    slot = folder / 'RMGE01.s09'
    slot.write_bytes(b'original')
    source = tmp_path / 'immutable-source.sav'
    source.write_bytes(b'source')

    def full_disk(src, dst):
        dst.write_bytes(b'partial')
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(native.shutil, 'copy2', full_disk)
    with pytest.raises(OSError, match='No space'):
        backend.load_state(source)
    assert slot.read_bytes() == b'original' and source.read_bytes() == b'source'
    assert not list(folder.glob('*.staging-*')) and not list(folder.glob('*.retained-*'))


def test_frame_reply_requires_matching_cpu_completion_even_when_paused():
    source = bridge()
    source.timeout = 1
    source.set(0, 0x534D4752)
    source.set(1, PROTOCOL_VERSION)

    def premature_reply():
        while source.get(3) == 0:
            time.sleep(0.0005)
        source.set(14, 1)
        source.set(8, 12)
        source.set(16, source.get(3) - 1)
        source.set(4, source.get(3))

    thread = threading.Thread(target=premature_reply)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="CPU frame completion"):
            source.command(2, 12)
    finally:
        thread.join()
        source.close()


@pytest.mark.parametrize(
    "pid,paused,error,expected",
    [
        (999, 1, 0, "different Dolphin PID"),
        (100, 0, 0, "without a paused"),
        (100, 1, 2, "rejected command"),
    ],
)
def test_native_rejects_ack_from_wrong_pid_unpaused_or_failed_emulator(
    pid, paused, error, expected
):
    source = bridge()
    source.timeout = 1
    source.set(0, 0x534D4752)
    source.set(1, PROTOCOL_VERSION)
    process = SimpleNamespace(pid=100, poll=lambda: None)

    def reply():
        while source.get(3) == 0:
            time.sleep(0.0005)
        source.set(2, pid)
        source.set(13, error)
        source.set(14, paused)
        source.set(4, source.get(3))

    thread = threading.Thread(target=reply)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match=expected):
            source.command(4, process=process)
    finally:
        thread.join()
        source.close()
