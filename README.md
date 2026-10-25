# SMG RL

Visual recurrent reinforcement learning for **Super Mario Galaxy 1 and 2** in
Dolphin on Windows.

**Latest reward review:** [hourly reviews and the motor8 reward audit](docs/motor8-reward-review.md)
identify stationary waypoint credit and an overweighted SMG1 initial-prefix
reward. An isolated schema-2 correction is tested but not enabled in training.
SMG1 continuation stopped at 4096 new real decisions after prediction gates
still failed; SMG2 reached its 6096-decision cap and awaits bounded evaluation.

**Current owner-selected controls:** [exactly eight exclusive actions in both games](docs/motor8-and-algorithms.md),
with a video inset of the actual policy image and live applied-button indicators.
The new bounded motor8 pilot supersedes the remaining legacy54/motor16 stages;
old checkpoints and their historical profiles are retained. The CLI defaults to
`config/motor8-smg2.yaml`; Galaxy 1 uses `config/motor8-smg1.yaml`. Rainbow DQN is
assessed as a separate future baseline, not an unverified replacement of the
active world-model learner.

**Earlier October 25 intervention (superseded):** [reviewed motor and exploration experiments](docs/exploration-interventions.md)
implement a versioned 16-action profile, exact button releases, RAM ground contact
for SMG1 and a validated initial-route prefix. A four-arm, three-seed curiosity/
entropy comparison and separate motor/reward pilots use finite reviewed budgets.
The scripted platform exit is a control fixture, not learned mission completion.
Remaining legacy54/motor16 stages were superseded by the owner's eight-action
request; their data and checkpoints remain retained.

**October 25 failure analysis:** [the intrinsic-reward and control audit](docs/exploration-failure-analysis.md)
confirms positive calculated novelty without horizontal movement, but finds
small immediate gradient effects at coefficient 0.01 and substantial action,
entropy and progress-target concerns. A trained zero-intrinsic comparison is
still needed. Read-only diagnostics and 731 actual-game fixture decisions added
no training steps or learned completions. The owner subsequently authorized the
bounded intervention linked above; older stop statements are dated snapshots.

**October 25 follow-up:** [actual policy videos and frontier savestate exploration](docs/frontier-exploration.md)
are implemented and checked in bounded live pilots. A paired
[8-versus-16 imagination trial](docs/horizon-trial.md) passed prediction gates but
showed no clear navigation benefit; horizon 8 remains the default. Experiments
are stopped after their budgets, with local videos and checkpoints retained.

**Earlier staged result (October 25, 2026, before the current motor8 pilot):** all bounded experiments finished
and training is stopped. Galaxy 2 terminal reward weight 5 improved prediction
gates and enabled more actor updates, without observed navigation improvement
over 25000 new real decisions per arm. Galaxy 1 curiosity coefficient 0.1 showed
no clear benefit over 0.01 in 10000 decisions per arm. No full-start mission
completion was observed. See [the staged results, plots and limitations](docs/staged-training.md).

**The owner authorized Dreamer-inspired collection and training on October 22,
2026.** At the dated snapshot of **October 23, 2026, 17:48 UTC**, both
world-model jobs were running. Galaxy 2 had 732,288 real decisions, 22,748 model
updates, 1,182 imagined actor updates and 1,192,992 imagined transitions;
Galaxy 1 had 229,349 real decisions, 30,508 model updates, 421 imagined actor
updates and 431,104 imagined transitions. The maximum recorded geometric projection
was 19.53% (`spinning`) and 13.76% (`double_time`) for Galaxy 2, and 5.06% for
Galaxy 1; no full-start mission has been completed. At the owner's request,
both jobs were stopped later that day before the PC was shut down; their final
checkpoints are retained at 870,272 Galaxy 2 and 278,773 Galaxy 1 real
decisions. On October 24 the owner authorized resuming with
[bounded stages and reviews](docs/staged-training.md), starting with a longer
Galaxy 2 offline comparison and a Galaxy 1 online health check.
The preserved PPO pairs remain at 824,416 Galaxy 2 and
220,891 Galaxy 1 decisions. Live counters remain in each run's `status.json`;
this paragraph is a dated snapshot. See
[the world-model implementation](docs/world-model.md) and the
[dated live record](docs/world-model-live.md).

The baseline uses one scenario-conditioned recurrent PPO policy for the two Galaxy 2 missions.
A separate policy targets Sweet Sweet Galaxy in Galaxy 1. Exploration combines
visual inverse-dynamics features, episodic k-nearest-neighbor novelty, and Random
Network Distillation (RND). This is **NGU-inspired exploration**, rather than a
reproduction of NGU's distributed Q-learning algorithm.

The native backend provides shared-memory renderer capture, bounded frame
stepping, stage profiling, and supervised checkpoint recovery. It has live
detector, health, capture, training, checkpoint, and recovery evidence. **A learned full-course solution has not been established.**
Near-star calibration success is a detector test, not evidence of learning.
The [documentation index](docs/README.md) separates operating instructions,
mathematical specification, measured evidence, and planned experiments.

The current **protocol-v3** backend uses Windows event wakeups and sends
already cropped 84 x 84 grayscale observations from Dolphin. The measured
collector profiles each worker's barrier wait and can transfer frame stacks
through shared memory while preserving synchronous recurrent PPO semantics.
See [image equivalence, configuration and timing evidence](docs/optimization-v3.md).

The recommended profiles now select Vulkan after a matched renderer comparison.
The latest revision also updates Torch/SB3, verifies checkpoint pair integrity,
and bounds failed-worker recovery. See the [renderer and security audit](docs/optimization-v4.md)
and [security policy](SECURITY.md). Full-course learning remains unproven.

An experimental PyTorch visual world model now provides episode-separated pixel
replay, recurrent dynamics, reward/continuation prediction, disagreement exploration,
and quality-gated actor/critic updates in latent imagination. It is Dreamer-inspired,
not a DreamerV3 reproduction. Live collection now records new image sequences;
old trajectory CSVs lack the required images. Experimental profiles are
`config/world-model.yaml` and `config/smg1-world-model.yaml`. Model optimizer updates
start after 4096 replay transitions; policy updates additionally require every
held-out scenario gate to pass. A failing cycle pauses new imagined actor
updates without stopping collection or discarding the model. No learning-speed
advantage or learned full-course completion has been established.
`inspect-replay` is read-only.
The Galaxy 1 objective repair balances pixel likelihood and terminal sampling
and directly supervises prior predictions; see the
[repair procedure and evidence](docs/world-model-smg1-repair.md).

## Supported tasks

| Game / executable | Mission | Training start | Detector fixture | Recommended configuration |
| --- | --- | --- | --- | --- |
| Galaxy 2 USA / `SB4E01` | Twisty Trials: Spinning and Spinning and Spinning | Slot 1 | Slot 2, near star | `config/native.yaml` |
| Galaxy 2 USA / `SB4E01` | Twisty Trials: Turning Turning Double Time | Slot 3 | Slot 4, near star | `config/native.yaml` |
| Galaxy 1 USA / `RMGE01` | Sweet Sweet Galaxy: Rocky Road | Slot 1 | Slot 2, near star | `config/smg1-native.yaml` |

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
| Exploration | Per-worker, per-scenario visual novelty; fades after 30 stalled decisions and stops at 120 | `0.005 * intrinsic_bonus`, at most `+0.5` per episode |
| Progress | New valid episode high-water projection of Mario's world X/Z position onto the calibrated start-to-star axis | `5 * increase` |
| Completion | Game-specific star-collection scene nerve | `+20` |
| Death | Game-specific actor/scene predicates; Galaxy 1 also checks zero HP | `-2` |
| Nonfatal damage, Galaxy 1 | Decrease in RAM health since the preceding sample | `-0.25` per HP lost |
| Time | Each decision | `-0.001` |
| Timeout | Decision limit or 180 decisions without valid progress | `-2`, separate from RAM death |

These are the recommended native configurations' `bounded-exploration-v2`
settings. Legacy configurations retain their historical rewards. Curiosity is
scored every eligible decision and optimized in batches of eight; PPO still
collects 128 decisions per worker per rollout. See the
[optimization and stagnation audit](docs/optimization-v2.md) for measurements,
reward equations, validation, and checkpoint migration.

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
Python 3.11, Torch 2.14.1+cu126, and SB3/sb3-contrib 2.9.0. Run these commands from
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

Legacy configurations point to `D:\emuladores\Dolphin-x64\Dolphin.exe` and use
`%APPDATA%\Dolphin Emulator` as the source user profile. Edit a copy of the
appropriate YAML if paths differ. `--config` must appear **before** the
subcommand. Paths resolve relative to the project root. Profiles are copied
into owned `.local/` directories; the source profile is preserved. Existing
destination state copies are not overwritten; use a new owned profile after
replacing source savestates.

## Build the native backend

The recommended native configurations use `.local/native-dolphin/Dolphin.exe`.
They require a separate patched **Dolphin 2609a** build, compatible MSVC/Windows
SDK, CMake, and Ninja. The original installed emulator remains available for the
legacy fallback. From a compatible developer PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_native.ps1
```

The build script pins upstream source, applies the local bridge, and deploys into
an owned ignored directory. See [build prerequisites, protocol, measured evidence,
and licensing](docs/native-backend.md). For the stock emulator instead, use
`config/parallel.yaml` (Galaxy 2) or `config/smg1-training.yaml` (Galaxy 1) in the
commands below; those retain hotkey/PNG control and their original calibration.
Native configurations have separate calibration files.

## Validate before training

The checked-in calibration matches the supplied local states. Re-run validation
after changing the game executable, savestates, Dolphin build, memory profile,
or control timing:

```powershell
.\.venv\Scripts\smg-rl.exe --config config/native.yaml calibrate-background
.\.venv\Scripts\smg-rl.exe --config config/smg1-native.yaml calibrate-background
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
.\.venv\Scripts\smg-rl.exe --config config/native.yaml train-supervised --workers 4 --steps 1000000
```

```powershell
.\.venv\Scripts\smg-rl.exe --config config/smg1-native.yaml train-supervised --workers 1 --steps 1000000
```

Check `status` before starting additional learners. These commands do not attach
to an existing training run. Each worker owns its profile, loopback device discovery, shared-memory inputs
and renderer capture, and read-only RAM reader. Keyboard/mouse focus is unnecessary. The GPU
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
.\.venv\Scripts\smg-rl.exe --config config/native.yaml train-supervised --workers 4 --steps 100000 --resume runs/parallel-EXAMPLE/model-final.zip
```

Both `model-final.zip` and `model-final.intrinsic.pt` must exist. Policy,
optimizer, curiosity models, and lifelong statistics resume. Episodes, emulator
state, episodic memory, LSTM state, RNG state, and partial rollouts restart.
This is statistical continuation, not exact replay.

## Inspect, report, and stop

```powershell
.\.venv\Scripts\smg-rl.exe --config config/native.yaml status
.\.venv\Scripts\smg-rl.exe --config config/smg1-native.yaml status
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
.\.venv\Scripts\smg-rl.exe --config config/native.yaml stop
.\.venv\Scripts\smg-rl.exe --config config/smg1-native.yaml stop
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
    worker-00/timings.json                    # Bounded stage timing statistics
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
.\.venv\Scripts\smg-rl.exe --config config/native.yaml evaluate runs/parallel-EXAMPLE/model-final.zip --episodes 10
.\.venv\Scripts\smg-rl.exe --config config/smg1-native.yaml evaluate runs/smg1/parallel-EXAMPLE/model-final.zip --episodes 10
```

Galaxy 2 evaluates both missions; Galaxy 1 evaluates its single mission. Match
the checkpoint's game and observation/action spaces. The visible adapter needs
foreground focus and W/A/S/D movement, Shift crouch, left mouse jump, and middle
mouse spin. The owned profile sets F1-F4 load and F10 pause. `prepare`, `launch`,
`calibrate`, `train`, and `--attach` support this single-instance fallback;
`telemetry` also uses the single-process hook. See the [complete CLI reference](docs/cli.md).

## Profiling and measured acceleration

```powershell
.\.venv\Scripts\smg-rl.exe --config config/native.yaml benchmark --steps 128
.\.venv\Scripts\smg-rl.exe --config config/smg1-native.yaml benchmark --steps 128
```

The benchmark measures a fixed action cycle with visual curiosity and no PPO
optimization; it is excluded from training success reports. Workers publish
`timings.json`; learner status includes control/capture, telemetry, curiosity,
policy-forward, rollout, PPO-update, and checkpoint timings. Nested stages overlap
and must not be summed as exclusive costs.

Current protocol-v3 repeated environment tests measured **16.34 decisions/s in
Galaxy 1** and **20.07 in Galaxy 2**, excluding resets and PPO. Matched
poll/full-image conditions on the same build measured 15.34 and 17.97,
respectively: observed differences of about 6.6% and 11.6%. Native CPU image
preparation matched the tested OpenCV images exactly while reducing bridge
payload from 1521216 to 7056 bytes. Four-worker collection tests favored ordered
pipes over the shared-vector alternatives; all modes retain the synchronous
PPO barrier. These are short engineering tests, not five-worker training rates
or proof of faster learning. See [current measurements](docs/optimization-v3.md),
[curiosity/reward revision](docs/optimization-v2.md), and
[historical backend comparison](docs/native-backend.md).

## Timing, performance, and research limits

Native commands advance exactly the requested number of Dolphin frame-step
boundaries and return while paused, with fresh RGBA pixels in shared memory.
The final-frame readback avoids PNG files and screenshot-flush frames. Native
configurations use unlimited emulation speed and 12 frames per decision. All
128 decisions in each final environment benchmark acknowledged 12 frames;
unchanged-nerve scene-counter deltas also matched 12. Scene counters can reset
on transitions and are not the frame-control clock.

The fallback retains asynchronous hotkeys, bounded pause probes, and PNG capture.
Its historical throughput and failures are documented in
[background training](docs/background-training.md). Native control removes those
hotkey/capture failure paths; file-lock retries, panic exits, PID checks, command
timeouts, and supervision handle other failures. This is not a guarantee against
persistent file locks, invalid states, or driver/process crashes.

A 12-frame native action is shorter than the old approximate action, whose
handshakes added many frames. Migrating old policy/curiosity checkpoints changes
the control-duration distribution. It preserves weights, not an identical
experimental condition. More decisions per second do not prove better learning.
Long runs, independent seeds, reward ablations, and full-start success evaluation
remain necessary.

### Decision budgets and measured wall-clock

A decision is one environment step (12 emulated frame boundaries), not one
rendered frame or one imagined transition. The measured cumulative rates at the
October 23, 2026, 17:48 UTC snapshot imply the following wall-clock budgets for
a fresh 1,000,000-decision run. These are projections for consuming the budget, not for
learning or completing a mission:

| Mode | Galaxy 2 | Galaxy 1 | Basis |
| --- | ---: | ---: | --- |
| World model (collection + fitting + validation) | about 25.6 h | about 75 h (range 63-86 h) | 10.83 and 3.71 decisions/s |
| PPO, recent concurrent measurement | 17-18 h | 55-61 h | 15.4-16.4 and 4.5-5.1 decisions/s |
| PPO, earlier native production snapshot | about 11 h | about 38 h | 25.0 and 7.3 decisions/s |

The recent PPO row comes from the [v4 runtime snapshot](docs/optimization-v4.md);
the earlier row comes from the [native production snapshot](docs/native-backend.md).

The world-model rate includes fitting and held-out validation, so it is lower
per real decision than the PPO collection rate. The tradeoff is favorable only
if fewer real decisions are needed; that has not been established. The Galaxy 2
world-model run started from zero, while the Galaxy 1 estimate is extrapolated
from a resumed run and earlier measurements. GPU load, emulator speed, resets,
recoveries and failures change the actual result. The `decisions_per_second`
field in a stopped PPO status file can describe only the last resume interval,
not the whole run.

## Development and license

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
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
