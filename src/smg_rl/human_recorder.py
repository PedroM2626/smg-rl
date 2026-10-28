"""Interactive, success-only human capture using a private frame-stepped Dolphin.

Human input is sampled every two emulated frames. Simultaneous buttons are kept
as a bitmask; they are not silently converted into the agent's eight actions.
"""

import csv
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.native import CONTROLS, NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.profiles import PROFILES
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.train import new_run

WINDOW = 'SMG RL - Human recorder'
GAME_ID = 'RMGE01'
FRAMES_PER_SAMPLE = 2
SAMPLE_FPS = 30
KEYS = {'w': 0x57, 'a': 0x41, 's': 0x53, 'd': 0x44,
        'jump': 0x01, 'crouch': 0x10, 'spin': 0x04}


def next_frame_deadline(previous, now):
    """Do not run a burst of old frames when interactive rendering falls behind."""
    return max(previous + 1 / SAMPLE_FPS, now)


def confirmed_star(telemetry, start_verified, steps):
    """A near-star fixture or stale/invalid terminal can never become a success."""
    return bool(start_verified and steps > 0 and telemetry.scenario == 1
                and telemetry.stage == PROFILES[GAME_ID].stage and telemetry.complete
                and not telemetry.dead and telemetry.scene_nerve in PROFILES[GAME_ID].complete)


def controls_mask(names):
    if any(name not in CONTROLS for name in names):
        raise ValueError('Unknown recorded game control.')
    return sum(1 << index for index, name in enumerate(CONTROLS) if name in names)


def discard_attempt(path, pending_root):
    """Delete only a recorder-owned pending attempt, never a caller-selected path."""
    path, pending_root = Path(path), Path(pending_root)
    if (not re.fullmatch(r'attempt-[0-9]{6}', path.name) or path.is_symlink()
            or path.resolve().parent != pending_root.resolve()
            or not (path / '.human-recorder-owned').is_file()):
        raise ValueError('Refusing to discard a non-owned or unexpected path.')
    shutil.rmtree(path)


def exact_observation(backend):
    image = np.asarray(backend.image())
    if image.shape != (84, 84) or image.dtype != np.uint8:
        raise ValueError('The recorded visual observation must be an exact 84x84 uint8 image.')
    return image.copy()


class VideoWriter:
    def __init__(self, path, rgba):
        executable = shutil.which('ffmpeg')
        if not executable:
            raise FileNotFoundError('ffmpeg is required on PATH for video recording.')
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError('Video requires full RGBA capture.')
        self.height, self.width = rgba.shape[:2]
        self.path = Path(path)
        self.frames = 0
        command = [executable, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{self.width}x{self.height}', '-r', str(SAMPLE_FPS),
            '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(self.path)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.pending = queue.Queue(maxsize=4)
        self.error = None
        self.worker = threading.Thread(target=self._encode, daemon=True)
        self.worker.start()

    def _encode(self):
        try:
            while True:
                rgba = self.pending.get()
                if rgba is None:
                    break
                self.process.stdin.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB).tobytes())
        except Exception as exc:
            self.error = exc
        finally:
            try:
                self.process.stdin.close()
            except OSError:
                pass

    def ready(self):
        if self.error is not None:
            raise RuntimeError('Video encoder failed.') from self.error
        if self.process.poll() is not None:
            raise RuntimeError('Video encoder exited unexpectedly.')
        return not self.pending.full()

    def write(self, rgba):
        if rgba.shape != (self.height, self.width, 4):
            raise ValueError('The game viewport changed size during recording.')
        if not self.ready():
            raise RuntimeError('Video encoder queue is full; game must remain paused.')
        self.pending.put_nowait(rgba.copy())
        self.frames += 1

    def close(self):
        deadline = time.monotonic() + 60
        while True:
            if self.error is not None or self.process.poll() is not None:
                self.abort()
                raise RuntimeError('Video encoder failed while finishing.') from self.error
            try:
                self.pending.put(None, timeout=.1)
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    self.abort()
                    raise RuntimeError('Video encoder did not finish in 60 seconds.') from None
        self.worker.join(max(0, deadline - time.monotonic()))
        if self.worker.is_alive() or self.error is not None:
            self.abort()
            raise RuntimeError('Video encoder could not finish its queued frames.') from self.error
        try:
            code = self.process.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError('Video encoder did not finish in 60 seconds.') from None
        error = self.process.stderr.read().decode('utf-8', errors='replace')
        self.process.stderr.close()
        if code or not self.path.is_file() or not self.path.stat().st_size or not self.frames:
            raise RuntimeError(f'Video encoding failed: {error}')

    def abort(self):
        """Discarded attempts need not wait for MP4 finalization."""
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        # The worker may be waiting for input or blocked in the pipe.
        if self.worker.is_alive():
            try:
                self.pending.put_nowait(None)
            except queue.Full:
                pass
            self.worker.join(timeout=5)
        self.process.stderr.close()


class Attempt:
    def __init__(self, pending_root, number, backend, initial):
        self.pending_root = Path(pending_root)
        self.path = self.pending_root / f'attempt-{number:06d}'
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / '.human-recorder-owned').write_text('owned\n', encoding='utf-8')
        self.frames = (self.path / 'observations.gray8').open('xb')
        self.actions = (self.path / 'inputs.jsonl').open('x', encoding='utf-8')
        self.trajectory = (self.path / 'trajectory.csv').open('x', encoding='utf-8', newline='')
        self.csv = csv.DictWriter(self.trajectory, fieldnames=['sample', 'emulated_frames', 'x', 'y', 'z',
            'health', 'grounded', 'dead', 'complete', 'scene_nerve', 'scene_frame'])
        self.csv.writeheader()
        self.steps = 0
        self.last = initial
        self.closed = False
        self.frames.write(exact_observation(backend).tobytes())
        self.write_position(initial)
        rgba = backend.bridge.image().copy()
        self.video = VideoWriter(self.path / 'gameplay.mp4', rgba)

    def write_position(self, telemetry):
        x, y, z = telemetry.position
        self.csv.writerow({'sample': self.steps, 'emulated_frames': self.steps * FRAMES_PER_SAMPLE,
            'x': x, 'y': y, 'z': z, 'health': telemetry.health, 'grounded': telemetry.grounded,
            'dead': telemetry.dead, 'complete': telemetry.complete,
            'scene_nerve': f'0x{telemetry.scene_nerve:08X}', 'scene_frame': telemetry.scene_frame})

    def record(self, names, backend, telemetry):
        self.actions.write(json.dumps({'sample': self.steps, 'emulated_frame_start': self.steps * FRAMES_PER_SAMPLE,
            'duration_frames': FRAMES_PER_SAMPLE, 'controls': list(names), 'mask': controls_mask(names)}) + '\n')
        self.steps += 1
        self.frames.write(exact_observation(backend).tobytes())
        self.video.write(backend.bridge.image().copy())
        self.last = telemetry
        self.write_position(telemetry)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.video.close()
        finally:
            self.frames.close()
            self.actions.close()
            self.trajectory.close()

    def discard(self):
        if not self.closed:
            self.closed = True
            try:
                self.video.abort()
            finally:
                self.frames.close()
                self.actions.close()
                self.trajectory.close()
        discard_attempt(self.path, self.pending_root)

    def promote(self, completed_root, start_verified, source_state_sha256):
        self.close()
        if not confirmed_star(self.last, start_verified, self.steps):
            raise ValueError('Attempt does not have a valid full-start RAM-confirmed star.')
        if not re.fullmatch(r'[0-9a-f]{64}', source_state_sha256):
            raise ValueError('A verified source savestate SHA-256 is required.')
        if self.video.frames != self.steps:
            raise ValueError('Encoded frame count differs from input count.')
        if (self.path / 'observations.gray8').stat().st_size != (self.steps + 1) * 84 * 84:
            raise ValueError('Observation frame count differs from input count.')
        with (self.path / 'inputs.jsonl').open(encoding='utf-8') as stream:
            inputs = [json.loads(line) for line in stream]
            if len(inputs) != self.steps:
                raise ValueError('Recorded input count differs from real game samples.')
        for index, item in enumerate(inputs):
            if (item['sample'] != index or item['emulated_frame_start'] != index * FRAMES_PER_SAMPLE
                    or item['duration_frames'] != FRAMES_PER_SAMPLE or item['mask'] != controls_mask(item['controls'])):
                raise ValueError('Input timeline or controls are inconsistent.')
        with (self.path / 'trajectory.csv').open(encoding='utf-8', newline='') as stream:
            positions = list(csv.DictReader(stream))
        if (len(positions) != self.steps + 1 or positions[-1]['complete'] != 'True'
                or positions[-1]['dead'] != 'False'
                or int(positions[-1]['scene_nerve'], 16) != self.last.scene_nerve):
            raise ValueError('Trajectory does not end at the validated RAM event.')
        probe = cv2.VideoCapture(str(self.path / 'gameplay.mp4'))
        try:
            opened, frame = probe.read()
            if not probe.isOpened() or not opened or frame is None:
                raise ValueError('Encoded video could not be decoded.')
        finally:
            probe.release()
        plot_human_trajectory(self.path / 'trajectory.csv', self.path / 'trajectory.png')
        files = ('gameplay.mp4', 'observations.gray8', 'inputs.jsonl', 'trajectory.csv', 'trajectory.png')
        manifest = {'state': 'completed', 'game_id': GAME_ID, 'scenario': 1, 'start_slot': 1,
            'source_state_sha256': source_state_sha256, 'full_start_ram_star': True,
            'terminal_telemetry': self.last.to_dict(), 'samples': self.steps,
            'emulated_frames_per_sample': FRAMES_PER_SAMPLE, 'observation_shape': [84, 84],
            'observation_dtype': 'uint8', 'video_frames': self.video.frames,
            'control_mask_order': list(CONTROLS),
            'control_semantics': 'Raw simultaneous human buttons sampled at two-frame intervals; not motor8-v1 action labels.',
            'training_compatible_without_conversion': False,
            'files_sha256': {name: digest(self.path / name) for name in files}}
        if not atomic_json(self.path / 'manifest.json', manifest):
            raise OSError('Could not write success manifest.')
        completed_root = Path(completed_root)
        completed_root.mkdir(parents=True, exist_ok=True)
        target = completed_root / self.path.name
        if target.exists():
            raise FileExistsError('A successful attempt cannot overwrite another.')
        self.path.rename(target)
        return target


def plot_human_trajectory(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with Path(source).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    x = [float(row['x']) for row in rows]
    z = [float(row['z']) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(x, z, linewidth=1.5, color='#3155a5')
    ax.scatter([x[0], x[-1]], [z[0], z[-1]], c=['#27a46b', '#df9e25'], s=70, zorder=3)
    ax.annotate('Start', (x[0], z[0]))
    ax.annotate('Star event', (x[-1], z[-1]))
    ax.set(xlabel='World X', ylabel='World Z', title='Human full-start trajectory (top-down)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def window_active(win32gui):
    handle = win32gui.FindWindow(None, WINDOW)
    return bool(handle and win32gui.GetForegroundWindow() == handle)


def pressed_controls(win32api):
    return tuple(name for name in CONTROLS if win32api.GetAsyncKeyState(KEYS[name]) & 0x8000)


def display(backend, message, attempt, names=()):
    image = cv2.cvtColor(backend.bridge.image(), cv2.COLOR_RGBA2BGR)
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 55), (0, 0, 0), -1)
    cv2.putText(image, message, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
    text = f'Attempt {attempt} | {" + ".join(names) if names else "none"} | F9: stop'
    cv2.putText(image, text, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, image)
    cv2.waitKey(1)


def start_is_canonical(telemetry, config):
    expected = json.loads(Path(config['paths']['calibration']).read_text(encoding='utf-8'))['variations']['sweet_sweet']['start']
    return bool(telemetry.scenario == 1 and telemetry.stage == PROFILES[GAME_ID].stage
                and not telemetry.dead and not telemetry.complete and telemetry.health == 3
                and np.linalg.norm(np.subtract(telemetry.position, expected)) <= .1)


def record_human(max_minutes=120, max_attempts=100):
    if os.name != 'nt':
        raise RuntimeError('The human recorder requires Windows and an interactive desktop.')
    if not 1 <= max_minutes <= 360 or not 1 <= max_attempts <= 500:
        raise ValueError('Use finite session limits: 1–360 minutes and 1–500 attempts.')
    import psutil
    import win32api
    import win32gui

    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Close existing Dolphin instances before launching this isolated recorder.')
    config = load_config('config/smg1-six-two-hour.yaml')
    config['paths']['runs'] = str(Path(config['root']) / 'runs/human-demonstrations')
    config['parallel'].update(frame_skip=FRAMES_PER_SAMPLE, native_capture='full', workers=1)
    config['replay']['enabled'] = False
    require_disk_headroom(config)
    source_state = Path(config['paths']['assets']) / f'{GAME_ID}.s01'
    source_hash = digest(source_state)
    run = new_run(config, 'human')
    pending = run / 'pending'
    completed = run / 'completed'
    pending.mkdir()
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    status = {'state': 'starting', 'run': str(run), 'attempts_started': 0, 'deaths_discarded': 0,
              'completed': 0, 'source_state_sha256': source_hash, 'limits_minutes': max_minutes,
              'limits_attempts': max_attempts, 'note': 'Human input sampled only while the recorder window has focus.'}
    atomic_json(run / 'status.json', status)
    print(json.dumps({'run': str(run), 'window': WINDOW}), flush=True)
    backend = NativeDolphin(config)
    attempt = None
    created_window = False
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        created_window = True
        backend.load_slot(1)
        initial = backend.memory.sample()
        if not start_is_canonical(initial, config):
            raise ValueError('Slot 1 does not match the verified living Sweet Sweet initial state.')
        status['state'] = 'waiting_for_enter'
        atomic_json(run / 'status.json', status)
        deadline = time.monotonic() + max_minutes * 60
        started = False
        release_required = True
        next_frame = time.monotonic()
        while time.monotonic() < deadline and (attempt is not None or status['attempts_started'] < max_attempts):
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            active = window_active(win32gui)
            if active and win32api.GetAsyncKeyState(0x78) & 0x8000:  # F9
                break
            if not active:
                display(backend, 'Click this window to play. Game is paused.', status['attempts_started'])
                time.sleep(.05)
                next_frame = time.monotonic()
                continue
            if not started:
                display(backend, 'Press Enter to start. F9 stops the recorder.', status['attempts_started'])
                if win32api.GetAsyncKeyState(0x0D) & 0x8000:
                    started = True
                time.sleep(.03)
                next_frame = time.monotonic()
                continue
            names = pressed_controls(win32api)
            if release_required and names:
                display(backend, 'Release controls to begin the attempt.', status['attempts_started'])
                time.sleep(.03)
                next_frame = time.monotonic()
                continue
            if attempt is None:
                release_required = False
                status['attempts_started'] += 1
                attempt = Attempt(pending, status['attempts_started'], backend, initial)
                status['state'] = 'recording'
                atomic_json(run / 'status.json', status)
            now = time.monotonic()
            if now < next_frame:
                time.sleep(next_frame - now)
            if not window_active(win32gui):
                next_frame = time.monotonic()
                continue
            if not attempt.video.ready():
                display(backend, 'Encoder busy - game paused.', status['attempts_started'])
                time.sleep(.01)
                next_frame = time.monotonic()
                continue
            names = pressed_controls(win32api)
            backend.advance(names)
            telemetry = backend.memory.sample()
            display(backend, 'Recording. A death resets this attempt.', status['attempts_started'], names)
            attempt.record(names, backend, telemetry)
            next_frame = next_frame_deadline(next_frame, time.monotonic())
            if attempt.steps % (SAMPLE_FPS * 10) == 0:
                require_disk_headroom(config)
            if telemetry.dead:
                attempt.discard()
                attempt = None
                status['deaths_discarded'] += 1
                status['state'] = 'resetting'
                atomic_json(run / 'status.json', status)
                backend.load_slot(1)
                initial = backend.memory.sample()
                if not start_is_canonical(initial, config):
                    raise ValueError('Death reset did not restore the verified initial state.')
                release_required = True
                next_frame = time.monotonic()
            elif confirmed_star(telemetry, True, attempt.steps):
                if digest(source_state) != source_hash:
                    raise ValueError('The original savestate changed during recording.')
                target = attempt.promote(completed, True, source_hash)
                attempt = None
                status.update(state='completed', completed=1, successful_attempt=str(target))
                atomic_json(run / 'status.json', status)
                print(json.dumps({'validated_recording': str(target), 'state': 'completed'}), flush=True)
                display(backend, 'Star confirmed! Recording saved.', status['attempts_started'], names)
                time.sleep(3)
                break
        else:
            status['state'] = 'session_limit_reached'
        if status['state'] not in ('completed', 'session_limit_reached'):
            status['state'] = 'stopped_by_user'
        if attempt is not None:
            attempt.discard()
            attempt = None
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        if attempt is not None:
            attempt.close()  # Quarantined pending data are never treated as a success.
        raise
    finally:
        backend.close()
        if created_window:
            cv2.destroyWindow(WINDOW)
        atomic_json(run / 'status.json', status)
    return run
