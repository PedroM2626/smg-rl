import mmap
import threading
import time
from types import SimpleNamespace

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


def test_native_action_rejects_inexact_frame_acknowledgment():
    backend = object.__new__(NativeDolphin)
    backend.config = {"parallel": {"frame_skip": 12}}
    backend.process = SimpleNamespace(pid=100)
    backend.bridge = SimpleNamespace(get=lambda field: 11, command=lambda *args: None)
    with pytest.raises(RuntimeError, match="Expected 12 native frames, got 11"):
        backend.advance(["jump"])


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
