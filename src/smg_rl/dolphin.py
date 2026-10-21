"""Windows Dolphin adapter for held inputs, viewport capture, and paused decisions.

This is a real-time desktop backend, not a frame-synchronous emulator API.
Only a dedicated Dolphin render window may receive input. Focus loss aborts.
"""

import configparser
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


def prepare_profile(config: dict) -> Path:
    paths = config["paths"]
    target = Path(paths["user"])
    marker = target / ".smg-rl-profile"
    if target.exists() and not marker.exists():
        raise RuntimeError(f"Refusing to modify an unowned profile: {target}")
    if not marker.exists() or marker.read_text(encoding="utf-8").strip() != "ready":
        source = Path(paths["source_user"])
        target.mkdir(parents=True, exist_ok=True)
        marker.write_text("preparing", encoding="utf-8")
        for folder in ("Config", "Wii", "GC", "GameSettings"):
            if (source / folder).is_dir():
                shutil.copytree(
                    source / folder,
                    target / folder,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("play_rec.dat"),
                )
        ini = configparser.ConfigParser(interpolation=None, strict=False)
        ini.optionxform = str
        dolphin_ini = target / "Config" / "Dolphin.ini"
        ini.read(dolphin_ini, encoding="utf-8")
        overrides = {
            "Interface": {"ConfirmStop": "False", "PauseOnFocusLost": "False"},
            "Display": {
                "Fullscreen": "False",
                "RenderToMain": "False",
                "RenderWindowWidth": "640",
                "RenderWindowHeight": "480",
            },
            "Core": {"EmulationSpeed": "1.0", "EnableCheats": "False"},
            "General": {"UseDiscordPresence": "False"},
        }
        for section, fields in overrides.items():
            if not ini.has_section(section):
                ini.add_section(section)
            for key, value in fields.items():
                ini[section][key] = value
        with dolphin_ini.open("w", encoding="utf-8") as stream:
            ini.write(stream)
        hotkeys = "[Hotkeys]\nDevice = DInput/0/Keyboard Mouse\nGeneral/Toggle Pause = F10\n"
        for slot in range(1, 5):
            hotkeys += f"Load State/Load State Slot {slot} = F{slot}\n"
        (target / "Config" / "Hotkeys.ini").write_text(hotkeys, encoding="utf-8")
        marker.write_text("ready\n", encoding="utf-8")
    state_dir = target / "StateSaves"
    state_dir.mkdir(exist_ok=True)
    for slot in range(1, 5):
        source_state = Path(paths["assets"]) / f"SB4E01.s{slot:02d}"
        if not source_state.is_file():
            raise FileNotFoundError(source_state)
        dest = state_dir / source_state.name
        if not dest.exists():
            shutil.copy2(source_state, dest)
    return target


class DesktopDolphin:
    def __init__(self, config: dict):
        if os.name != "nt":
            raise RuntimeError("The desktop adapter requires Windows.")
        import mss
        import win32api
        import win32con
        import win32gui
        import win32process

        self.api, self.con, self.gui, self.process = win32api, win32con, win32gui, win32process
        self.config = config
        self.capture = mss.mss()
        self.held: set[str] = set()
        self.closed = False
        self.paused = False
        self.hwnd = self._find_window()
        self.focus()
        from smg_rl.memory import GameMemory

        self.memory = GameMemory()
        self.paused = self.detect_pause()

    def detect_pause(self):
        first = self.memory.sample()
        time.sleep(0.12)
        second = self.memory.sample()
        return (first.scene_frame, first.scene_nerve) == (second.scene_frame, second.scene_nerve)

    def _find_window(self):
        windows = []

        def visit(hwnd, _):
            title = self.gui.GetWindowText(hwnd)
            if self.gui.IsWindowVisible(hwnd) and "Dolphin" in title and "(SB4E01)" in title:
                windows.append(hwnd)

        self.gui.EnumWindows(visit, None)
        if len(windows) != 1:
            raise RuntimeError(f"Expected one SB4E01 render window, found {len(windows)}.")
        return windows[0]

    def focus(self):
        self.gui.ShowWindow(self.hwnd, self.con.SW_RESTORE)
        if self.gui.GetForegroundWindow() != self.hwnd:
            self.gui.SetForegroundWindow(self.hwnd)
        time.sleep(0.1)
        self.check_focus()

    def check_focus(self):
        if self.gui.GetForegroundWindow() != self.hwnd:
            self.release()
            raise RuntimeError(
                "Dolphin lost focus. Inputs were released; resume with Dolphin in front."
            )

    def _event(self, name, down):
        if name in ("jump", "spin"):
            flags = {
                ("jump", True): self.con.MOUSEEVENTF_LEFTDOWN,
                ("jump", False): self.con.MOUSEEVENTF_LEFTUP,
                ("spin", True): self.con.MOUSEEVENTF_MIDDLEDOWN,
                ("spin", False): self.con.MOUSEEVENTF_MIDDLEUP,
            }
            self.api.mouse_event(flags[name, down], 0, 0)
        else:
            key = self.con.VK_LSHIFT if name == "crouch" else ord(name.upper())
            self.api.keybd_event(
                key, self.api.MapVirtualKey(key, 0), 0 if down else self.con.KEYEVENTF_KEYUP, 0
            )

    def release(self):
        for name in tuple(self.held):
            self._event(name, False)
        self.held.clear()

    def set_action(self, names):
        self.check_focus()
        names = set(names)
        for name in self.held - names:
            self._event(name, False)
        priority = {"crouch": 0, "jump": 2, "spin": 3}
        for name in sorted(names - self.held, key=lambda key: priority.get(key, 1)):
            self._event(name, True)
        self.held = names

    def hotkey(self, virtual_key):
        self.check_focus()
        scan = self.api.MapVirtualKey(virtual_key, 0)
        self.api.keybd_event(virtual_key, scan, 0, 0)
        try:
            time.sleep(self.config["environment"]["key_hold_seconds"])
        finally:
            self.api.keybd_event(virtual_key, scan, self.con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.04)

    def pause(self, value=True):
        # DInput is polled; a key event alone is not evidence that Dolphin paused.
        for _ in range(3):
            self.paused = self.detect_pause()
            if self.paused == value:
                return
            self.hotkey(self.con.VK_F10)
        self.paused = self.detect_pause()
        if self.paused != value:
            self.release()
            raise RuntimeError("Dolphin did not acknowledge the F10 pause state.")

    def load_slot(self, slot):
        self.release()
        # Dolphin accepts load shortcuts while paused. Preserve the saved scene
        # before moving platforms can carry Mario away from a near-star state.
        self.pause(True)
        self.hotkey(self.con.VK_F1 + slot - 1)
        time.sleep(0.6)
        self.pause(True)

    def advance(self, names):
        # Use a neutral controller while Dolphin polls emulator hotkeys. A held
        # controller key can otherwise suppress a hotkey through input priority.
        self.release()
        try:
            self.pause(False)
            self.set_action(names)
            time.sleep(self.config["environment"]["action_seconds"])
            self.release()
            time.sleep(0.04)
            self.pause(True)
        finally:
            self.release()

    def image(self):
        self.check_focus()
        left, top, right, bottom = self.gui.GetClientRect(self.hwnd)
        x, y = self.gui.ClientToScreen(self.hwnd, (left, top))
        width, height = right - left, bottom - top
        # Fit a 16:9 viewport inside the client area, excluding letterbox bars.
        vw, vh = min(width, int(height * 16 / 9)), min(height, int(width * 9 / 16))
        x, y = x + (width - vw) // 2, y + (height - vh) // 2
        rgb = np.asarray(self.capture.grab({"left": x, "top": y, "width": vw, "height": vh}))
        crop_left, crop_top, crop_right, crop_bottom = self.config["environment"]["crop"]
        rgb = rgb[
            int(crop_top * vh) : int(crop_bottom * vh),
            int(crop_left * vw) : int(crop_right * vw),
            :3,
        ]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        n = self.config["environment"]["image_size"]
        return cv2.resize(gray, (n, n), interpolation=cv2.INTER_AREA)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.release()
        if self.gui.GetForegroundWindow() == self.hwnd:
            self.pause(True)
        self.capture.close()


def launch(config, slot=1):
    if os.name != "nt":
        raise RuntimeError("The desktop adapter requires Windows.")
    processes = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq Dolphin.exe", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if '"dolphin.exe"' in processes.lower():
        raise RuntimeError(
            "Close the existing Dolphin process, or use --attach with a running SB4E01 stage."
        )
    user = prepare_profile(config)
    paths = config["paths"]
    if not Path(paths["rom"]).is_file():
        raise FileNotFoundError(paths["rom"])
    return subprocess.Popen(
        [
            paths["dolphin"],
            "--user",
            str(user),
            "--exec",
            paths["rom"],
            "--save_state",
            str(user / "StateSaves" / f"SB4E01.s{slot:02d}"),
        ]
    )
