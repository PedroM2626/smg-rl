"""Isolated, hidden Dolphin workers using DSU input and internal PNG capture.

The existing emulator and savestates are retained. Rendering still uses a GPU
and a hidden Win32 surface; this is not a renderer-free NoGUI Dolphin build.
"""

import configparser
import json
import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from smg_rl.diagnostics import measured
from smg_rl.dolphin import prepare_profile
from smg_rl.dsu import DSUServer
from smg_rl.memory import GameMemory, MemoryFault
from smg_rl.process_memory import ProcessMemory
from smg_rl.profiles import game_id
from smg_rl.scheduling import apply_affinity


def prepare_worker(config, index, port):
    config = deepcopy(config)
    config["paths"]["user"] = str(Path(config["paths"]["user"]).parent / f"worker-{index:02d}")
    root = prepare_profile(config)
    ini = configparser.ConfigParser(interpolation=None, strict=False)
    ini.optionxform = str
    ini.read(root / "Config" / "Dolphin.ini", encoding="utf-8")
    for section, entries in {
        "Input": {"BackgroundInput": "True"},
        "General": {"HotkeysRequireFocus": "False"},
        "Interface": {
            "PauseOnFocusLost": "False",
            "OnScreenDisplayMessages": "False",
            "AbortOnPanicAlert": "True",
        },
        "Core": {"EmulationSpeed": str(config["parallel"]["emulation_speed"])},
        "Analytics": {"Enabled": "False"},
        # Dolphin falls back to Cubeb for unknown backend names. Muting that
        # stream still starts/stops Windows audio at every frame-step boundary.
        "DSP": {"Backend": "No Audio Output", "Volume": "0"},
    }.items():
        if section not in ini:
            ini[section] = {}
        ini[section].update(entries)
    if "cpu_thread" in config["parallel"]:
        ini["Core"]["CPUThread"] = str(bool(config["parallel"]["cpu_thread"]))
    if "graphics_backend" in config["parallel"]:
        graphics_backend = config["parallel"]["graphics_backend"]
        if graphics_backend not in ("D3D", "D3D12", "Vulkan", "OGL"):
            raise ValueError("Unsupported Dolphin graphics backend.")
        ini["Core"]["GFXBackend"] = graphics_backend
    with (root / "Config" / "Dolphin.ini").open("w", encoding="utf-8") as stream:
        ini.write(stream)
    device = "DSUClient/0/SMG-RL"
    (root / "Config" / "DSUClient.ini").write_text(
        f"[Server]\nEnabled = {bool(port)}\nEntries = SMG-RL:127.0.0.1:{port};\n", encoding="utf-8"
    )
    (root / "Config" / "Hotkeys.ini").write_text(
        f"[Hotkeys]\nDevice = {device}\nGeneral/Toggle Pause = Share\n"
        "General/Take Screenshot = L3\nFrame Advance/Frame Advance = R3\n"
        "Load State/Load State Slot 1 = Square\nLoad State/Load State Slot 2 = Cross\n"
        "Load State/Load State Slot 3 = Circle\nLoad State/Load State Slot 4 = Triangle\n",
        encoding="utf-8",
    )
    (root / "Config" / "WiimoteNew.ini").write_text(
        f"[Wiimote1]\nDevice = {device}\nExtension = Nunchuk\nButtons/A = PS\n"
        "Nunchuk/Buttons/Z = `Touch Button`\nNunchuk/Stick/Up = `Left Y-`\n"
        "Nunchuk/Stick/Down = `Left Y+`\nNunchuk/Stick/Left = `Left X-`\n"
        "Nunchuk/Stick/Right = `Left X+`\nShake/X = R1\nShake/Y = R1\nShake/Z = R1\n",
        encoding="utf-8",
    )
    graphics = configparser.ConfigParser(interpolation=None)
    graphics.optionxform = str
    graphics.read(root / "Config" / "GFX.ini", encoding="utf-8")
    for section, entries in {
        "Settings": {"InternalResolution": "1"},
        "Hardware": {"VSync": "False"},
    }.items():
        if section not in graphics:
            graphics[section] = {}
        graphics[section].update(entries)
    with (root / "Config" / "GFX.ini").open("w", encoding="utf-8") as stream:
        graphics.write(stream)
    logger = configparser.ConfigParser(interpolation=None)
    logger.optionxform = str
    logger["Options"] = {
        "WriteToFile": "True",
        "WriteToConsole": "False",
        "WriteToWindow": "False",
        "Verbosity": "3",
    }
    logger["Logs"] = {key: "True" for key in ("BOOT", "CORE", "COMMON", "Video", "CI")}
    with (root / "Config" / "Logger.ini").open("w", encoding="utf-8") as stream:
        logger.write(stream)
    return root


class BackgroundDolphin:
    @measured("startup")
    def __init__(self, config, index=0):
        self.config, self.index = config, index
        # Native masks bypass UDP input updates, but the current native build
        # still needs DSU discovery to instantiate its controller device.
        self.server = DSUServer()
        self.closed = False
        self.reader = None
        self.process = None
        self.names = ()
        self.last_image = None
        self.last_signature = None
        try:
            self.root = prepare_worker(config, index, self.server.port)
            self.screenshots = self.root / "ScreenShots"
            self.screenshots.mkdir(exist_ok=True)
            startup = subprocess.STARTUPINFO()
            startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
            self.process = subprocess.Popen(
                [
                    config["paths"]["dolphin"],
                    "--batch",
                    "--user",
                    str(self.root),
                    "--exec",
                    config["paths"]["rom"],
                    "--save_state",
                    str(self.root / "StateSaves" / f"{game_id(config)}.s{config['variations'][0]['start_slot']:02d}"),
                ],
                startupinfo=startup,
                env={**os.environ, "SMG_RL_BRIDGE": self.bridge.name}
                if hasattr(self, "bridge")
                else {k: v for k, v in os.environ.items() if k != "SMG_RL_BRIDGE"},
            )
            self.affinity = apply_affinity(self.process.pid, config["parallel"].get("cpu_affinity", "auto"))
            (self.root / "process.json").write_text(
                json.dumps(
                    {
                        "pid": self.process.pid,
                        "owner": os.getpid(),
                        "port": self.server.port,
                        "user": str(self.root),
                        "cpu_affinity": self.affinity,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            deadline = time.monotonic() + config["parallel"].get("startup_timeout", 90.0)
            while True:
                self.hide_windows()
                try:
                    if self.reader is None:
                        self.reader = ProcessMemory(self.process.pid, game_id(config))
                    self.memory = GameMemory(self.reader, game_id(config), config.get('game', {}).get('stage'),
                                         config.get('game', {}).get('signals'))
                    self.memory.sample()
                    if not self.server.clients and not hasattr(self, "bridge"):
                        raise MemoryFault("Waiting for the private DSU connection.")
                    break
                except MemoryFault as exc:
                    if self.process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Dolphin worker {index} did not become ready (exit={self.process.poll()}): {exc}"
                        ) from exc
                    time.sleep(0.2)
            self.hide_windows()
            self.paused = self.detect_pause()
            self.pause(True)
        except BaseException:
            self.close()
            raise

    def hide_windows(self):
        if not self.config["parallel"]["hidden"]:
            return
        import win32gui
        import win32process

        def visit(window, _):
            if win32process.GetWindowThreadProcessId(window)[1] == self.process.pid:
                win32gui.ShowWindow(window, 0)

        win32gui.EnumWindows(visit, None)

    def _pulse(self, command):
        self.server.set(self.names, [command])
        time.sleep(0.12 if command == "pause" else 0.045)
        self.server.set(self.names)
        time.sleep(0.035 if command == "pause" else 0.012)

    def detect_pause(self):
        first = self.memory.sample()
        # A 45 ms rendering stall was previously mistaken for a paused emulator.
        # Return early on movement; require a longer stable window for a pause.
        changes = 0
        previous = (first.scene_frame, first.scene_nerve)
        for _ in range(6):
            time.sleep(0.03)
            second = self.memory.sample()
            current = (second.scene_frame, second.scene_nerve)
            if previous != current:
                changes += 1
                if changes >= 2:
                    return False
            previous = current
        return True

    @measured("pause")
    def pause(self, value=True):
        # Reconcile against RAM before toggling, even if the cached state differs.
        self.paused = self.detect_pause()
        if self.paused == value:
            return
        for _ in range(6):
            self._pulse("pause")
            self.paused = self.detect_pause()
            if self.paused == value:
                return
        self.release()
        raise RuntimeError(f"Worker {self.index} did not acknowledge pause={value}.")

    def release(self):
        self.names = ()
        self.server.set()

    @measured("reset")
    def load_slot(self, slot):
        self.release()
        self.pause(True)
        self._pulse(f"load{slot}")
        time.sleep(0.15)
        self.paused = self.detect_pause()
        self.pause(True)
        self.last_image = None
        self.capture()

    @measured("advance")
    def advance(self, names):
        self.names = tuple(names)
        self.server.set(self.names)
        first = self.memory.sample()
        self.pause(False)
        deadline = time.monotonic() + self.config["parallel"]["timeout"]
        while True:
            current = self.memory.sample()
            if (
                current.scene_nerve != first.scene_nerve
                or current.dead
                or current.complete
                or current.scene_frame - first.scene_frame >= self.config["parallel"]["frame_skip"]
            ):
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"Worker {self.index} stopped advancing game frames.")
            time.sleep(0.001)
        self.pause(True)
        self.capture()
        self.release()

    @measured("capture")
    def capture(self):
        # The screenshot comes from Dolphin's renderer, not the desktop surface.
        started = time.time_ns()
        self._pulse("capture")
        self._pulse("frame")
        deadline = time.monotonic() + self.config["parallel"]["timeout"]
        retry_frame = time.monotonic() + 0.12
        retry_capture = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            candidates = list(self.screenshots.rglob("*.png"))
            if candidates:
                newest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
                stat = newest.stat()
                signature = (str(newest), stat.st_mtime_ns, stat.st_size)
                if signature != self.last_signature and stat.st_mtime_ns >= started:
                    try:
                        encoded = newest.read_bytes()
                    except OSError:
                        encoded = b""
                    raw = (
                        cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                        if len(encoded) > 12 and encoded[-8:-4] == b"IEND"
                        else None
                    )
                    if raw is not None and raw.std() > 1:
                        height, width = raw.shape
                        # Internal screenshots already exclude window chrome and bars.
                        left, top, right, bottom = self.config["environment"]["crop"]
                        raw = raw[
                            int(top * height) : int(bottom * height),
                            int(left * width) : int(right * width),
                        ]
                        size = self.config["environment"]["image_size"]
                        self.last_image = cv2.resize(
                            raw, (size, size), interpolation=cv2.INTER_AREA
                        )
                        self.last_signature = signature
                        # All candidates are generated in this owned worker profile.
                        for old in candidates:
                            if old != newest:
                                old.unlink(missing_ok=True)
                        return
            if time.monotonic() >= retry_frame:
                self._pulse("frame")
                retry_frame = time.monotonic() + 0.12
            if time.monotonic() >= retry_capture:
                self._pulse("capture")
                retry_capture = time.monotonic() + 1.0
            time.sleep(0.003)
        raise RuntimeError(f"Worker {self.index} did not produce a fresh rendered screenshot.")

    def image(self):
        if self.last_image is None:
            self.capture()
        return self.last_image.copy()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.release()
            if self.reader:
                self.reader.close()
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            self.server.close()
