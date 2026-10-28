# SMG RL

Visual reinforcement-learning experiments for **Super Mario Galaxy** and
**Super Mario Galaxy 2** on Windows, using Dolphin. This is a research project:
no full-start, RAM-confirmed mission completion has been demonstrated in the
experiments reported here.

The later [frozen SMG1 IQN evaluation](docs/frozen-iqn-evaluation-20261027.md)
completed six canonical attempts without a star. Its goal-distance candidate was
analyzed offline. A separately signed [frozen-vision genetic proximity experiment](docs/genetic-goal-distance-20261027.md)
selects full XYZ distance without a mapped route; it does not change the IQN reward.
Its five-minute pilot ended with no star and no improvement over its imported
champion; [the audit](docs/genetic-goal-distance-20261027.md#completed-pilot-and-audit)
separates completed attempts, partial attempts and geometric metrics.
The later owner-requested [three-hour run](docs/genetic-goal-distance-20261027.md#owner-requested-three-hour-run)
also produced no star or improvement over its reevaluated initial champion.
The subsequent [robust-selection calibration](docs/robust-genetic-plan-20261027.md#calibration-result-blocked)
failed its preregistered viability gate; neither proposed comparison arm ran.
The latest [expanding-route IQN trial](docs/dynamic-route-night.md#completed-result)
stopped after two main blocks without a validated route extension or a star.
The [experiment history](docs/experiment-history.md) summarizes findings,
negative results and comparison limits across the project.
The subsequent [matched reward-scale comparison](docs/reward-scale-study-20261028.md#completed-matched-comparison)
failed its preregistered gate in both training seeds; no larger block was started.
The later [chocolate-frontier exploration](docs/chocolate-frontier-search-20261028.md)
reproduced partial scripted paths from the full start, but found no stable
archive candidate or learned mission completion; its demonstration comparison
completed with zero stars and did not pass its continuation gate.
The subsequent [scripted Sweet Sweet search](docs/sweet-manual-search-20261028.md)
replayed a longer partial path from the initial slot; it did not reach a star.

## Current evidence

In the October 27, 2026 audit, an earlier SMG1 IQN/IMPALA allocation stopped
after approximately 2 hours 18 minutes because the disk filled. The intended
four hours were **not** completed. Its last valid checkpoint contains 749,006
real decisions and 12,125 completed attempts. A further 97,668 acknowledged
decisions in the failed segment have no usable final policy/replay checkpoint.
The failed checkpoint is incomplete and must not be resumed. See the
[failure audit and recovery boundaries](docs/iqn-four-hours-20261027.md).

The best full-start IQN collection trajectory in that failed segment reached a
**6.586% start-to-goal geometric projection**, then died. This is not a native
mission percentage, a fraction of the winding route, or a frozen-policy test;
the weights at that point were not saved. No star was observed. Earlier frozen
IQN/IMPALA evaluations also obtained no stars. The two SMG2 evaluations reached
100% of their *mapped openings*, not 100% of their missions, before dying.
[Frozen results](docs/impala-final-results.md) distinguish these outcomes.

A separate full-visual genetic trial obtained no stars and did not improve its
unchanged 8.0665% geometric baseline. The strongest mutated-vision candidate
reached 7.9463%; these are collection maxima from different experiments, not a
controlled IQN-versus-GA ranking. See [genetic results](docs/genetic-visual.md).

The expanding-route trial subsequently collected 100,884 decisions and reached
66.25% of the existing opening during exploratory collection, with no accepted
route extensions or stars. Its best grounded start-to-goal projection was 4.016%;
neither number is native mission progress or a frozen-policy success. Training
was stopped at the October 28 review; check actual process and controller state
before starting anything.
The [current evidence and next-experiment guide](docs/current-status.md) is the
entry point for status, limitations, and a bounded research plan.

## Tasks and inputs

| Game | Mission | Full-start slot | Near-star detector fixture |
| --- | --- | --- | --- |
| Galaxy 2, `SB4E01` | Twisty Trials: Spinning and Spinning and Spinning | 1 | 2 |
| Galaxy 2, `SB4E01` | Twisty Trials: Turning Turning Double Time | 3 | 4 |
| Galaxy 1, `RMGE01` | Sweet Sweet Galaxy: Rocky Road | 1 | 2 |

The near-star states validate the completion detector and do not establish
learned success. Curated intermediate states have separate assisted-start labels.
ROMs, savestates, worker profiles, run data, replay and checkpoints are local and
ignored by Git. See [asset names and roles](data/README.md). The checked profiles
were validated against the documented USA game builds and Dolphin 2609a; other
versions require recalibration.

## IQN method and reward profiles

The source SMG1 run used `config/smg1-six-two-hour.yaml`, six isolated Dolphin
collectors and one model-free IQN learner with an IMPALA-style residual visual
encoder. The online encoder is trainable; it was not frozen in IQN. It uses
prioritized replay, double and dueling heads, three-step returns and noisy
layers. The learner does not use a world model, imagined experience or visual
curiosity. Rainbow, recurrent PPO, world models and genetic search are separate
implemented experimental alternatives, not components automatically active in
this run. The older Rainbow/IQN comparison used different signed settings; see
[model-free implementation](docs/model-free.md).

The policy receives a stack of four cropped 84 × 84 grayscale images and the
scenario identity. RAM coordinates, health and terminal signals are used for
reward and evaluation, not fed to the policy. The selected `motor8-v1` profile
provides eight exclusive logical actions: none, right, left, forward, backward,
jump, spin and crouch. Each decision advances 12 emulated frames; jump and spin
use the documented 8-frame press / 4-frame release pulse. The separate frame-4
carry-direction experiment has different control semantics and must not be
silently mixed with this replay. See [controls and signatures](docs/motor8-and-algorithms.md).

The source SMG1 signed reward uses `regional-advance-undiscounted-v2`: new
grounded high-water advance along a **mapped opening only**, star +20, death or
timeout -2, and SMG1 nonfatal HP loss -0.25 per HP; no per-step or intrinsic
bonus, with gamma 1. It includes a mix of canonical and verified assisted
starts. Beyond the mapped route there is no incremental route reward until the
star; additional training time alone is not evidence this limitation is fixed.
The later expanding-route trial used canonical starts only, a fresh signed
replay/optimizer and a different fixed normalization; see its
[protocol and results](docs/dynamic-route-night.md). The old
`regional-frame-potential-v1` profiles have different reward semantics.
See [reward specification and attempt metrics](docs/simple-reward-impala.md).

The game-specific completion detector reads a scene-state address
(`0x807D5318` for the tested SMG2 missions, `0x806A1BD8` for tested SMG1);
SMG1 HP uses actor-relative offsets. Address interpretation depends on the
validated game/emulator state. Full-start star evidence must be distinguished
from detector fixtures, assisted resets, imagined transitions and projections.
See [SMG2 memory](docs/memory-profile.md), [SMG1 memory](docs/smg1.md) and
[damage limitations](docs/damage-and-recovery.md).

## Installation and local operation

The validation-only [Center of the Universe setup](docs/center-of-universe.md)
includes a separate human trajectory recorder and a pre-boss reference location.
No training is enabled for that mission.

To record a human Sweet Sweet demonstration with automatic death reset and
RAM-confirmed success, see the [human recording guide](docs/human-recording.md).
The guide also covers the two SMG2 recording launchers and the independently
signed experimental SMG1 profile derived from a completed human route. Route
coverage is not a native mission-completion signal or evidence of agent success.
See the [experiment reuse register](docs/experiment-reuse-20261028.md) for prior
results and the [one-hour-per-reward study](docs/reward-hour-pair-20261028.md)
for its bounded protocol and evaluation limits.
The [reward-density comparison](docs/human-route-density-pair-20261028.md)
tests the full human route with two separate reward scales.
The [progress interpretation](docs/progress-metric-interpretation-20261028.md)
withdraws the faulty initial-segment score: it could report progress while Mario
remained on the starting platform. Ordered human-observed landmark proximity
now provides conservative candidates for visual review, not a level percentage.

Use Windows, 64-bit Python 3.11 or 3.12, and user-provided game images and
compatible savestates. The tested GPU environment used Python 3.11 and Torch
2.14.1+cu126. The lock records exact packages and an extra CUDA index; availability
of the pinned wheels on those indexes is required when recreating the environment.
The previously cached installation wheels were removed during storage cleanup.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Native profiles require a separately patched Dolphin build and local assets;
see [native build and prerequisites](docs/native-backend.md). Hidden operation
still renders through a hidden surface; it is not renderer-free headless
emulation. Keep the original emulator and user profile untouched. If paths
differ, copy a complete YAML directly under `config/`, adjust its paths and
revalidate. `--config` precedes the CLI subcommand.

Read-only inspection (the historical failed run can contain a stale `running`
status; check actual PIDs, creation times and failure evidence as well):

```powershell
.\.venv\Scripts\python.exe -m smg_rl.cli --help
.\.venv\Scripts\python.exe scripts/run_smg1_six.py snapshot --state .local/iqn-four-hour-20261027-state.json
```

The generic CLI default is `config/impala-iqn-smg2.yaml`, **not** the latest
SMG1 branch configuration. Configuration files are complete profiles, not
overlays. A checkpoint must match its game, model, action, reward and replay
signatures; preserve the original checkpoint and audit before any continuation.
The [CLI reference](docs/cli.md) documents examples and their side effects.
They do not authorize an unbounded job or a retry of a failed allocation.

TensorBoard event files can contain episode reward, action occupancy, sampled
path distance, grounded route advance and attempts, depending on the run. A
local server is not necessarily running; to start one intentionally:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir runs/model-free/tensorboard --host 127.0.0.1 --port 6006
```

Visit <http://127.0.0.1:6006/> while that process is running. Historical event
coverage varies; absent metrics must not be interpreted as zero.

## Documentation, tests and license

- [Documentation index](docs/README.md) and [current status](docs/current-status.md)
- [Experimental evidence and next steps](docs/current-status.md#next-experiments)
- [Training guide](docs/training.md), [CLI](docs/cli.md) and [reproducibility](docs/reproducibility.md)
- [Historical PPO mathematics](docs/methodology.md), [world model](docs/world-model.md)
  and [experiment protocol](docs/experiments.md)
- [Storage cleanup and failure evidence](docs/iqn-four-hours-20261027.md#subsequent-verified-storage-cleanup)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Portable tests do not boot Dolphin or prove that a policy solved a mission.
Original project material uses the [MIT license](LICENSE); games, Dolphin and
dependencies have separate rights, as described in [third-party notices](THIRD_PARTY_NOTICES.md).
