"""Versioned, PID-verified Dolphin shared-memory capture and synchronous stepping."""

import ctypes
import mmap
import os
import struct
import time
import uuid

import cv2
import numpy as np

from smg_rl.background import BackgroundDolphin
from smg_rl.diagnostics import measured

HEADER_BYTES = 256
PIXEL_BYTES = 1920 * 1080 * 4
MAGIC = 0x534D4752
PROTOCOL_VERSION = 3
CONTROLS = ("w", "a", "s", "d", "jump", "crouch", "spin")


class BridgeEvents:
    """Private auto-reset Windows events; ownership/liveness checks stay in the bridge."""

    def __init__(self, name):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
        self.kernel.CreateEventW.restype = ctypes.c_void_p
        self.kernel.SetEvent.argtypes = [ctypes.c_void_p]
        self.kernel.SetEvent.restype = ctypes.c_int
        self.kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.kernel.WaitForSingleObject.restype = ctypes.c_uint32
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        self.handles = []
        try:
            for suffix in ("request", "reply", "cpu"):
                handle = self.kernel.CreateEventW(None, False, False, name + "-" + suffix)
                if not handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                self.handles.append(handle)
        except BaseException:
            self.close()
            raise

    def publish(self):
        if not self.kernel.SetEvent(self.handles[0]):
            raise ctypes.WinError(ctypes.get_last_error())

    def wait(self, milliseconds):
        result = self.kernel.WaitForSingleObject(self.handles[1], milliseconds)
        if result not in (0, 258):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        for handle in self.handles:
            self.kernel.CloseHandle(handle)
        self.handles.clear()


class NativeBridge:
    def __init__(self, timeout=30.0, skip_present=False, wakeup="events", capture="cpu", size=84,
                 crop=(0.12, 0.17, 0.88, 0.91), quiet_ui=True):
        if os.name != "nt":
            raise RuntimeError("The Dolphin native bridge requires Windows.")
        self.name = "Local\\SMGRL-" + uuid.uuid4().hex
        if wakeup not in ("events", "poll") or capture not in ("full", "cpu", "gpu"):
            raise ValueError("Invalid native wakeup or capture mode.")
        if not 1 <= size <= 512 or not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1):
            raise ValueError("Invalid native crop or observation size.")
        self.events = BridgeEvents(self.name)
        try:
            self.mapping = mmap.mmap(-1, HEADER_BYTES + PIXEL_BYTES, tagname=self.name)
        except BaseException:
            self.events.close()
            raise
        self.mapping[:HEADER_BYTES] = bytes(HEADER_BYTES)
        self.set(0, MAGIC)
        self.set(1, PROTOCOL_VERSION)
        self.set(19, int(skip_present) | (4 if wakeup == "events" else 0) | (8 if quiet_ui else 0))
        self.set(25, size)
        for field, fraction in enumerate(crop, start=26):
            self.set(field, round(fraction * 1_000_000))
        self.set(43, {"full": 0, "cpu": 1, "gpu": 2}[capture])
        self.wakeup = wakeup
        self.timeout = timeout

    def get(self, field):
        return struct.unpack_from("<I", self.mapping, field * 4)[0]

    def set(self, field, value):
        struct.pack_into("<I", self.mapping, field * 4, value)

    def command(self, command, argument=0, names=(), process=None):
        if self.get(3) != self.get(4):
            raise RuntimeError("A native command is still pending; restart the owned worker.")
        sequence = (self.get(3) + 1) & 0x7FFFFFFF or 1
        self.set(5, command)
        self.set(6, argument)
        self.set(7, sum(1 << i for i, name in enumerate(CONTROLS) if name in names))
        # perf_counter_ns uses QPC but not the raw counter's frequency. Publish
        # the raw QPC count so native dispatch latency uses exactly one clock.
        if os.name == "nt":
            ticks = ctypes.c_longlong()
            ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(ticks))
            struct.pack_into("<Q", self.mapping, 33 * 4, ticks.value)
        # Windows x64 stores are ordered; the aligned request publishes prior fields.
        self.set(3, sequence)
        events = getattr(self, "events", None) if getattr(self, "wakeup", "poll") == "events" else None
        if events:
            events.publish()
        deadline = time.monotonic() + self.timeout
        while self.get(4) != sequence:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"Native Dolphin exited with code {process.returncode}.")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Native command {command} timed out (sequence {sequence}).")
            if events:
                events.wait(max(1, min(50, int((deadline - time.monotonic()) * 1000))))
            else:
                time.sleep(0.0005)
        if self.get(0) != MAGIC or self.get(1) != PROTOCOL_VERSION:
            raise RuntimeError("Native bridge protocol mismatch.")
        if process is not None and self.get(2) != process.pid:
            raise RuntimeError("Native bridge belongs to a different Dolphin PID.")
        if self.get(13):
            raise RuntimeError(
                f"Native bridge rejected command {command}: error {self.get(13)}; "
                f"diagnostics={self.diagnostics()}."
            )
        if not self.get(14):
            raise RuntimeError("Native command returned without a paused emulator.")
        if command == 2 and (self.get(16) != sequence or self.get(8) != argument):
            raise RuntimeError("Native response precedes the matching CPU frame completion.")

    def diagnostics(self):
        return {name: self.get(field) for name, field in {
            "request": 3, "response": 4, "completed_frames": 8,
            "step_complete_sequence": 16, "field_count": 17,
            "acknowledged_field": 18, "readback_us": 20, "frame_wait_us": 21,
            "skipped_presents": 22, "reconciliations": 23, "rejected_state": 24,
            "pixel_format": 30, "cpu_batch_us": 31, "cpu_idle_us": 32, "dispatch_us": 35,
            "present_us": 36, "scale_us": 37, "image_bytes": 38, "source_width": 39,
            "source_height": 40, "queue_drain_calls": 41, "preprocess_us": 42, "capture_mode": 43,
        }.items()}

    def image(self):
        width, height, stride = (self.get(i) for i in (10, 11, 12))
        channels = 1 if self.get(30) == 1 else 4
        if self.get(30) not in (0, 1) or not (0 < width <= 1920 and 0 < height <= 1080 and stride == width * channels):
            raise RuntimeError("Native renderer has not published a valid RGBA frame.")
        size = height * stride
        shape = (height, width) if channels == 1 else (height, width, 4)
        return np.frombuffer(self.mapping[HEADER_BYTES : HEADER_BYTES + size], np.uint8).reshape(shape)

    def close(self):
        self.mapping.close()
        if getattr(self, "events", None):
            self.events.close()


class NativeDolphin(BackgroundDolphin):
    def __init__(self, config, index=0):
        self.bridge = NativeBridge(
            config["parallel"]["timeout"], config["parallel"].get("skip_present", False),
            config["parallel"].get("bridge_wakeup", "events"),
            config["parallel"].get("native_capture", "cpu"),
            config["environment"]["image_size"], config["environment"]["crop"],
            config["parallel"].get("quiet_ui", True),
        )
        self.last_frame_count = 0
        try:
            super().__init__(config, index)
        except BaseException:
            self.bridge.close()
            raise

    def detect_pause(self):
        return bool(self.bridge.get(14))

    @measured("pause")
    def pause(self, value=True):
        if not value:
            raise RuntimeError("Native mode advances bounded frames; free-running is disabled.")
        self.bridge.command(4, process=self.process)
        self.paused = True

    def release(self):
        self.names = ()
        self.bridge.set(7, 0)
        self.server.set()

    @measured("reset")
    def load_slot(self, slot):
        self.release()
        previous = self.bridge.get(9)
        self.bridge.command(3, slot, process=self.process)
        self._fresh_image(previous)
        self.paused = True

    @measured("advance")
    def advance(self, names):
        previous = self.bridge.get(9)
        frames = self.config["parallel"]["frame_skip"]
        with self.profiler.measure("native_round_trip"):
            self.bridge.command(2, frames, names, self.process)
        self.last_frame_count = self.bridge.get(8)
        if self.last_frame_count != frames:
            raise RuntimeError(f"Expected {frames} native frames, got {self.last_frame_count}.")
        self._fresh_image(previous)
        self.release()

    def _fresh_image(self, previous):
        if self.bridge.get(9) == previous:
            raise RuntimeError("Native command did not publish a fresh rendered frame.")
        self.capture()

    @measured("capture")
    def capture(self):
        rgba = self.bridge.image()
        if rgba.ndim == 2:
            size = self.config["environment"]["image_size"]
            if rgba.shape != (size, size) or rgba.std() <= 1:
                raise RuntimeError("Native compact observation has an invalid shape or is uniform.")
            self.last_image = rgba
            return
        height, width = rgba.shape[:2]
        left, top, right, bottom = self.config["environment"]["crop"]
        crop = rgba[
            int(top * height) : int(bottom * height), int(left * width) : int(right * width)
        ]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGBA2GRAY)
        if gray.std() <= 1:
            raise RuntimeError("Native renderer published an empty or uniform observation.")
        size = self.config["environment"]["image_size"]
        self.last_image = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    def close(self):
        if self.closed:
            return
        super().close()
        self.bridge.close()


def background_backend(config, index=0):
    backend = config.get("parallel", {}).get("backend", "legacy")
    if backend == "native":
        return NativeDolphin(config, index)
    if backend == "legacy":
        return BackgroundDolphin(config, index)
    raise ValueError(f"Unknown background backend: {backend}")
