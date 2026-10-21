# SMG RL

Vision-based recurrent reinforcement learning for **Super Mario Galaxy 2**, running in the Windows desktop version of Dolphin. One scenario-conditioned policy trains on both Twisty Trials variations:

1. **Spinning and Spinning and Spinning** — start state 1, near-star state 2.
2. **Turning Turning Double Time** — start state 3, near-star state 4.

The policy sees four grayscale gameplay frames and a scenario one-hot vector. World coordinates, death, and completion are read from game RAM **only for rewards, validation, logging, and trajectory plots**. They are never policy observations.

Exploration combines an inverse-dynamics visual embedding, episodic k-nearest-neighbor novelty, and a separate Random Network Distillation model for each variation. Recurrent PPO supplies temporal memory for rotating platforms and jumps. This is an **NGU-inspired exploration system**, not a reproduction of NGU's distributed Q-learning algorithm.

## Current validation status

- Live-tested with **Dolphin 2609a**, the USA **SB4E01** executable, and all four supplied savestates.
- Both near-star states begin **incomplete**. Jumping and collecting each star produces the same completion nerve: **`0x807D5318`**.
- Walking backward off the starting platform in each variation produces death nerve **`0x807D4E3C`** and scene nerve **`0x807D5320`**.
- Starting position, forward movement, pause acknowledgment, memory endianness, and dynamic actor pointers are checked.
- Automated tests run without the game. Short live training runs validate the pipeline; **the supplied smoke checkpoints are not policies that have solved either course**.

See [the address profile](docs/memory-profile.md), [the experiment guide](docs/training.md), and [the local validation report](docs/validation.md).

## Install

Use 64-bit **Python 3.11 or 3.12**. The default `python` command on this machine points to Python 3.8, so use the launcher explicitly:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The tested Windows dependency versions are recorded in `requirements-windows.lock.txt`. To reproduce them in a fresh environment, install that file before installing the project with `--no-deps -e .`.

The local assets belong in `data/`:

```text
data/
  Super Mario Galaxy 2 (USA) (En,Fr,Es).nkit.iso
  SB4E01.s01
  SB4E01.s02
  SB4E01.s03
  SB4E01.s04
```

ROMs, savestates, copied Dolphin profiles, checkpoints, and full experiment logs are ignored by Git. Bring your own game image and states. Savestates require a compatible Dolphin build.

## Configure Dolphin

The defaults use `D:\emuladores\Dolphin-x64\Dolphin.exe` and `%APPDATA%\Dolphin Emulator` as the source profile. Change paths in a copy of `config/default.yaml` named `config/local.yaml` if needed. That local file is ignored by Git. Pass `--config config/local.yaml` **before** the subcommand.

The project copies the source profile into `.local/dolphin-user`, preserving the original profile. The disposable Wii play log is omitted because Dolphin may lock it. The copy receives explicit F1–F4 load bindings and an F10 pause binding. State files are copied only if the destination does not already exist.

```powershell
.\.venv\Scripts\smg-rl.exe prepare
```

The existing controller bindings are used:

| Input | Action |
| --- | --- |
| W / A / S / D | Nunchuk movement |
| Left Shift | Crouch / ground pound |
| Left mouse button | Jump |
| Middle mouse button | Shake / spin |

Close other Dolphin processes before starting the isolated profile. Keep the render window visible and in front while the agent runs. Do not use the keyboard or mouse during training. Losing focus releases the held inputs and aborts the run, saving the available checkpoint.

## Calibrate and train

The checked-in calibration matches the supplied states. Re-run calibration whenever the game, savestates, emulator build, or address profile changes:

```powershell
.\.venv\Scripts\smg-rl.exe calibrate
```

Calibration opens the game, checks all four states, collects both stars, tests both deaths, and records evidence. It returns to state 1 and leaves Dolphin paused. It uses no memory writes.

With the game still open, start training:

```powershell
.\.venv\Scripts\smg-rl.exe train --attach --steps 1000000
```

If Dolphin is closed, omit `--attach` to launch the isolated profile automatically. To attach to an existing session, it must already be in one of these two Twisty Trials scenarios, with working F1–F4 and F10 bindings.

Press **Ctrl+C in the terminal** to stop and save. Each run has its own timestamped directory:

```text
runs/train-.../
  config.json
  episodes.jsonl
  model-10000.zip
  model-10000.intrinsic.pt
  model-final.zip
  model-final.intrinsic.pt
  spinning/episode-000001.csv
  spinning/episode-000001.png
  double_time/episode-000002.csv
  double_time/episode-000002.png
```

CSV files include XYZ coordinates, rewards, progress, actual scene-frame counters, and state-machine values. PNGs show the measured **world X/Z trajectory from above** and progress over decision steps. They contain real sampled positions, not estimated camera motion or a fabricated map.

Resume with both matching checkpoint files present:

```powershell
.\.venv\Scripts\smg-rl.exe train --attach --resume runs/train-EXAMPLE/model-final.zip --steps 1000000
```

The policy, optimizer, RND target/predictor, inverse model, curiosity optimizer, and lifelong statistics resume together. Episodic memory and LSTM state restart at the next savestate reset. Reproducibility is statistical: desktop input timing is asynchronous.

## Evaluate and inspect

Evaluation disables intrinsic rewards and curiosity updates, uses deterministic policy actions, and reports success separately for each variation:

```powershell
.\.venv\Scripts\smg-rl.exe evaluate --attach runs/train-EXAMPLE/model-final.zip --episodes 10
.\.venv\Scripts\smg-rl.exe telemetry
.\.venv\Scripts\smg-rl.exe plot runs/train-EXAMPLE/spinning/episode-000001.csv
```

A successful near-star calibration validates the detector; it does not count as an agent completing a full course. Training and evaluation reset only to states **1 and 3**.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

`ci/windows-tests.yml` is a GitHub Actions template. Move it to `.github/workflows/tests.yml` if your publishing credential has the `workflow` scope. The current repository uses locally executed checks.

## Practical limits

This backend uses a visible desktop window and polled Windows inputs. It acknowledges pause using the game's frame counter, pauses during learning updates, and releases controller inputs around emulator shortcuts. Decision duration is variable; `action_seconds` is the additional held-action interval, not a promise of an exact frame count. Neutral game frames can occur during the shortcut handshake. A headless or frame-synchronous Dolphin integration would improve throughput and action precision.

The address profile deliberately fails on unsupported game IDs, executable signatures, scenes, scenarios, invalid pointers, and non-finite coordinates. The route reward projects XYZ positions onto the calibrated start-to-star axis in X/Z. This is useful for these courses, but is a geometric progress proxy rather than a hidden native "percent complete" field. Completion is exclusively the star-collection state transition.

Long training and reward tuning are still required to learn both courses. Compare success rates per variation rather than treating intrinsic return as evidence of winning. At desktop speed, one million decisions can take several days.

## References

- [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038)
- [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
- [Recurrent PPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
- [Garigari SB4E01 symbols](https://github.com/SMGCommunity/Garigari/blob/main/config/SB4E01/symbols.txt)
- [Python Dolphin Memory Engine](https://github.com/randovania/py-dolphin-memory-engine)
- [Dolphin source](https://github.com/dolphin-emu/dolphin)
