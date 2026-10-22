# SMG RL

Visual recurrent reinforcement learning for **Super Mario Galaxy 1 and 2** in
Dolphin on Windows. The canonical repository is
[PedroM2626/smg-rl](https://github.com/PedroM2626/smg-rl), private at the owner's
request. Original project code and documentation use the [MIT License](LICENSE).
All repository text and generated labels are in English.

One scenario-conditioned recurrent PPO policy learns the two Galaxy 2 missions.
A separate policy learns Sweet Sweet Galaxy in Galaxy 1. Exploration combines
visual inverse-dynamics features, episodic k-nearest-neighbor novelty, and Random
Network Distillation (RND). This is **NGU-inspired exploration**, rather than a
reproduction of NGU's distributed Q-learning algorithm.

The implementation has live detector, health, capture, training, checkpoint, and
recovery evidence. **A learned full-course solution has not been established.**
Near-star calibration success is a detector test, not evidence of learning.
The [documentation index](docs/README.md) separates operating instructions,
mathematical specification, measured evidence, and planned experiments.

## Supported tasks

| Game / executable | Mission | Training start | Detector fixture | Recommended configuration |
| --- | --- | --- | --- | --- |
| Galaxy 2 USA / `SB4E01` | Twisty Trials: Spinning and Spinning and Spinning | Slot 1 | Slot 2, near star | `config/parallel.yaml` |
| Galaxy 2 USA / `SB4E01` | Twisty Trials: Turning Turning Double Time | Slot 3 | Slot 4, near star | `config/parallel.yaml` |
| Galaxy 1 USA / `RMGE01` | Sweet Sweet Galaxy: Rocky Road | Slot 1 | Slot 2, near star | `config/smg1-training.yaml` |

The checked profiles and supplied savestates were tested with **Dolphin 2609a**.
Other regions, revisions, or state files require new validation. Galaxy 1's
`config/smg1.yaml` is a validation-only configuration with training disabled.

## Observation, memory, and rewards

The policy receives four cropped 84 x 84 grayscale gameplay frames and a scenario
one-hot vector: two entries for Galaxy 2, one for Galaxy 1. There are 54 discrete
actions: nine movement directions, including idle, combined with six button
modes. RAM values are used for rewards, checks, and logs, and are excluded from
the policy observation.

| Signal | Source and interpretation | Default contribution |
| --- | --- | --- |
| Exploration | Per-worker, per-scenario episodic visual novelty multiplied by normalized RND novelty | `0.02 * intrinsic_bonus` |
| Progress | New valid episode high-water projection of Mario's world X/Z position onto the calibrated start-to-star axis | `5 * increase` |
| Completion | Game-specific star-collection scene nerve | `+20` |
| Death | Game-specific actor/scene predicates; Galaxy 1 also checks zero HP | `-2` |
| Nonfatal damage, Galaxy 1 | Decrease in RAM health since the preceding sample | `-0.25` per HP lost |
| Time | Each decision | `-0.0001` |

**Progress is a geometric proxy, not a native game completion percentage.** Height,
displacement, and death gates reject invalid progress. Sweet Sweet Galaxy's
curved route is not mapped to ordered waypoints, so its projection can misstate
distance along the actual path. Only the RAM completion event proves success.

Galaxy 2 completion is `0x807D5318` for both missions; Galaxy 1 completion is
`0x806A1BD8`. Galaxy 1 health and maximum health are read through the dynamic
Mario actor pointer at offsets `0x380` and `0x3E0`. A live 3-to-2 HP transition
was measured. Healing earns no bonus; an unchanged HP value incurs no repeated
damage penalty. Fatal transitions use the death penalty without an extra damage
charge. Damage frames receive no novelty bonus. Net HP sampling can miss damage
followed by healing between samples.

See the [Galaxy 2 pointer profile](docs/memory-profile.md),
[Galaxy 1 profile](docs/smg1.md), [damage evidence](docs/damage-and-recovery.md),
and [reward equations and limitations](docs/methodology.md).

## Install and supply local assets

Use 64-bit **Python 3.11 or 3.12** on Windows. The tested GPU environment uses
Python 3.11, Torch 2.6.0+cu124, and SB3/sb3-contrib 2.6.0. Run these commands from
the repository root in a fresh environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

The GPU lock includes the CUDA wheel index and development tools. The separate
`requirements-windows.lock.txt` records the older desktop CPU environment.
Default `python` on the development machine points to Python 3.8, so commands
explicitly use the virtual environment. CUDA configurations require a compatible
NVIDIA GPU/driver. For a CPU experiment, copy the relevant complete YAML to an
ignored `config/local.yaml` and change `training.device` to `cpu`.

```text
data/
  Super Mario Galaxy 2 (USA) (En,Fr,Es).nkit.iso
  SB4E01.s01
  SB4E01.s02
  SB4E01.s03
  SB4E01.s04
  Super Mario Galaxy (USA) (En,Fr,Es).nkit.iso
  RMGE01.s01
  RMGE01.s02
```

See [asset roles](data/README.md). ROMs, savestates, copied profiles, virtual
environments, checkpoints, and full run logs are local-only and ignored by Git.
Savestates must match the emulator build.

Defaults point to `D:\emuladores\Dolphin-x64\Dolphin.exe` and use
`%APPDATA%\Dolphin Emulator` as the source user profile. Edit a copy of the
appropriate YAML if paths differ. `--config` must appear **before** the
subcommand. Paths resolve relative to the project root. Profiles are copied
into owned `.local/` directories; the source profile is preserved. Existing
destination state copies are not overwritten; use a new owned profile after
replacing source savestates.

## Validate before training

The checked-in calibration matches the supplied local states. Re-run validation
after changing the game executable, savestates, Dolphin build, memory profile,
or control timing:

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml calibrate-background
.\.venv\Scripts\smg-rl.exe --config config/smg1.yaml calibrate-background
```

Calibration uses a temporary hidden worker to check start/goal states, movement,
star collection, death, and pause acknowledgment. It records evidence and closes
its owned instance. No game memory is written. Training and evaluation reset
only to full-start states: Galaxy 2 slots 1/3 and Galaxy 1 slot 1.

## Train with hidden instances and automatic recovery

The recommended local arrangement is **two independent learners and five Dolphin
processes**: four Galaxy 2 workers alternating its missions and one Galaxy 1
worker. Run each command in a separate terminal when starting fresh experiments:

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml train-supervised --workers 4 --steps 1000000
```

```powershell
.\.venv\Scripts\smg-rl.exe --config config/smg1-training.yaml train-supervised --workers 1 --steps 1000000
```

Check `status` before starting additional learners. These commands do not attach
to an existing training run. Each worker owns its profile, DSU loopback input,
RAM reader, and renderer capture. Keyboard/mouse focus is unnecessary. The GPU
still renders to a hidden Win32 surface: this is **hidden rendering, not
renderer-free headless emulation**. CUDA runs policy learning; curiosity models
run on CPU with independent worker/scenario statistics.

`train-supervised` wraps `train-parallel`, retries infrastructure failures using
completed policy/curiosity pairs, and preserves its original timestep target.
It backs off between attempts and detects 240 seconds without learner status.
A failing worker aborts that attempt; recovery may lose work since the last
complete checkpoint. It does not repair a bad configuration or guarantee
convergence. See [recovery semantics](docs/damage-and-recovery.md).

`--steps` is the number of **additional aggregate decisions**, including when
resuming; a PPO rollout can round the final count upward. To resume a particular
experiment, replace the example path with a real matching checkpoint pair:

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml train-supervised --workers 4 --steps 100000 --resume runs/parallel-EXAMPLE/model-final.zip
```

Both `model-final.zip` and `model-final.intrinsic.pt` must exist. Policy,
optimizer, curiosity models, and lifelong statistics resume. Episodes, emulator
state, episodic memory, LSTM state, RNG state, and partial rollouts restart.
This is statistical continuation, not exact replay.

## Inspect, report, and stop

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml status
.\.venv\Scripts\smg-rl.exe --config config/smg1-training.yaml status
.\.venv\Scripts\smg-rl.exe report --output runs/training-report.json
```

`status` selects the newest learner attempt under the configured runs directory
and includes its owning supervisor state. An explicit attempt directory may
follow `status` or `stop`. The report audits both games, including live episode
snapshots, and exports a fixed best-trajectory CSV and top-down figure for each
available task. It excludes validation/evaluation fixtures and near-star resets.
A success requires a full-start trace and the correct RAM completion evidence.
Missing evidence is reported as a warning or unknown value.

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml stop
.\.venv\Scripts\smg-rl.exe --config config/smg1-training.yaml stop
```

Each stop request also stops the owning supervisor. Allow the current vector
step and final checkpoint save to finish. Ctrl+C in the supervisor's foreground
terminal also requests shutdown. A worker failure may prevent a final save;
the most recent complete periodic pair remains the recovery point.

The CLI generates reports when invoked. Scheduled notifications are configured
separately in the local desktop app; cloning this repository creates no schedule.
See [concurrent operation and report interpretation](docs/concurrent-training.md).

## Outputs and trajectory plots

```text
runs/                                      # Galaxy 2
  supervised-TIMESTAMP-PID/                 # Manager status and attempt logs
  parallel-TIMESTAMP/                       # One learner attempt
    config.json, controller.json, status.json
    model-N.zip, model-N.intrinsic.pt        # Matching resumable pair
    worker-00/spinning/episode-000001.csv
    worker-00/spinning/episode-000001.png
    worker-00/live.csv
    worker-00/episodes.jsonl
runs/smg1/                                 # Independent Galaxy 1 namespace
  supervised-TIMESTAMP-PID/
  parallel-TIMESTAMP/worker-00/sweet_sweet/
```

CSV logs contain measured XYZ positions, valid progress, action, reward,
actor/scene nerves, scene counter, and terminal outcomes. Galaxy 1 also records
health, maximum health, HP lost, and damage penalty. Per-worker episode summaries
are saved as JSONL. Live snapshots update every 16 decisions and at reset/terminal
transitions. Periodic pairs are saved every 1024 Galaxy 2 or 256 Galaxy 1
aggregate decisions; each attempt retains the newest three periodic pairs.

Plots show actual world **X/Z coordinates from above**, with decision order and
progress history. They are trajectories, not reconstructed floor maps. Every
episode gets a CSV; PNGs are generated every 20 episodes, on success, and for
interrupted episodes. Generate an individual figure later with:

```powershell
.\.venv\Scripts\smg-rl.exe plot runs/parallel-EXAMPLE/worker-00/spinning/episode-000001.csv --output runs/trajectory.png
```

## Evaluation and desktop fallback

`evaluate` currently uses the visible desktop adapter. Stop parallel learners
and close their Dolphin instances before starting a desktop evaluation. It uses
deterministic actions with intrinsic rewards and curiosity learning disabled:

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml evaluate runs/parallel-EXAMPLE/model-final.zip --episodes 10
.\.venv\Scripts\smg-rl.exe --config config/smg1-training.yaml evaluate runs/smg1/parallel-EXAMPLE/model-final.zip --episodes 10
```

Galaxy 2 evaluates both missions; Galaxy 1 evaluates its single mission. Match
the checkpoint's game and observation/action spaces. The visible adapter needs
foreground focus and W/A/S/D movement, Shift crouch, left mouse jump, and middle
mouse spin. The owned profile sets F1-F4 load and F10 pause. `prepare`, `launch`,
`calibrate`, `train`, and `--attach` support this single-instance fallback;
`telemetry` also uses the single-process hook. See the [complete CLI reference](docs/cli.md).

## Timing, performance, and research limits

Input, pause, and PNG capture are asynchronous. Scene counters reset on state
transitions; screenshot flushing can add frames. Approximate frame skipping is
not a frame-synchronous simulator API. Current pause checks use longer pulses,
reconciliation, and two observed counter advances to avoid queued-frame errors.
Galaxy 2 startup is staggered by five seconds per worker index, with a separate
90-second boot timeout and 30-second ordinary action/capture timeout.

Short historical throughput benchmarks used earlier timing revisions and are
documented in [background training](docs/background-training.md). Re-measure the
current revision, include startup and failures separately, and avoid inferring
linear scaling from worker count. Higher emulation speed or frame skip can
coarsen control and miss brief events. Pause and startup failures have occurred;
supervision bounds interruption, rather than removing the underlying timing
limitations. Long runs, multiple seeds, reward ablations, and independent success
evaluation remain necessary to establish learning quality.

## Development and license

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

Portable tests cover telemetry, health/reward semantics, curiosity persistence,
observations, capture/pause behavior, DSU, supervision, and reporting. They do not
boot a game. Live evidence is documented separately in the
[validation register](docs/validation.md). `ci/windows-tests.yml` is a template;
hosted GitHub Actions is not configured by its presence in that directory.

The [MIT License](LICENSE) applies to original project material. Game artwork,
the retained gameplay screenshot, Dolphin, and dependencies retain their own
rights and licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
The [technical report](docs/methodology.md),
[experimental protocol](docs/experiments.md),
[reproducibility guide](docs/reproducibility.md), and
[BibTeX bibliography](docs/references.bib) provide the academic and operational
detail behind this overview.
