"""Versioned, PID-verified Dolphin shared-memory capture and synchronous stepping."""

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
CONTROLS = ("w", "a", "s", "d", "jump", "crouch", "spin")


class NativeBridge:
    def __init__(self, timeout=30.0):
        if os.name != "nt":
            raise RuntimeError("The Dolphin native bridge requires Windows.")
        self.name = "Local\\SMGRL-" + uuid.uuid4().hex
        self.mapping = mmap.mmap(-1, HEADER_BYTES + PIXEL_BYTES, tagname=self.name)
        self.mapping[:HEADER_BYTES] = bytes(HEADER_BYTES)
        self.set(0, MAGIC)
        self.set(1, 1)
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
        # Windows x64 stores are ordered; the aligned request publishes prior fields.
        self.set(3, sequence)
        deadline = time.monotonic() + self.timeout
        while self.get(4) != sequence:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"Native Dolphin exited with code {process.returncode}.")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Native command {command} timed out (sequence {sequence}).")
            time.sleep(0.0005)
        if self.get(0) != MAGIC or self.get(1) != 1:
            raise RuntimeError("Native bridge protocol mismatch.")
        if process is not None and self.get(2) != process.pid:
            raise RuntimeError("Native bridge belongs to a different Dolphin PID.")
        if self.get(13):
            raise RuntimeError(f"Native bridge rejected command {command}: error {self.get(13)}.")
        if not self.get(14):
            raise RuntimeError("Native command returned without a paused emulator.")

    def image(self):
        width, height, stride = (self.get(i) for i in (10, 11, 12))
        if not (0 < width <= 1920 and 0 < height <= 1080 and stride == width * 4):
            raise RuntimeError("Native renderer has not published a valid RGBA frame.")
        size = height * stride
        return np.frombuffer(self.mapping[HEADER_BYTES : HEADER_BYTES + size], np.uint8).reshape(
            height, width, 4
        )

    def close(self):
        self.mapping.close()


class NativeDolphin(BackgroundDolphin):
    def __init__(self, config, index=0):
        self.bridge = NativeBridge(config["parallel"]["timeout"])
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
