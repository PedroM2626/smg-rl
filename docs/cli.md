# Command-line reference

The current CLI default is `config/impala-iqn-smg2.yaml`. The finite
[IQN/IMPALA protocol](simple-reward-impala.md) owns the current six collectors;
do not launch these example commands in addition to an active supervisor.

```powershell
# Read-only view of actual owned runs, attempts and bounded budgets.
.venv\Scripts\python.exe scripts/run_impala_trial.py snapshot

# Explicit standalone finite example, only when no current owned collector runs.
.venv\Scripts\python.exe -m smg_rl.cli --config config/impala-iqn-smg1.yaml train-dqn --steps 3072 --max-seconds 900
```

`--max-seconds` stops collection at the earlier wall/decision boundary, then
closes emulators and saves the final immutable checkpoint. Boot time is included;
close/checkpoint overhead may finish afterward. Attempts persist on resume.
Historical profiles retain their original reward/encoder settings.

Run from the repository root using `.\.venv\Scripts\smg-rl.exe` or
`.\.venv\Scripts\python.exe -m smg_rl.cli`. The optional global argument
`--config PATH` goes before the subcommand and defaults to `config/iqn-smg2.yaml`
for the retained IQN profile. Both initial algorithm trials are completed and
stopped; Rainbow remains explicitly selectable for deliberately allocated work.
Current model-free profiles are `rainbow-smg1.yaml`, `rainbow-smg2.yaml` and the
corresponding `iqn-*` alternatives. Use `train-dqn` and `record-dqn`; see the
[implementation, TensorBoard and finite protocol](model-free.md).
The retained `config/regional-smg1-12.yaml` is the earlier RSSM reward/curriculum
profile, not a model-free learner profile.
These require the verified local curriculum banks; old checkpoint/replay signatures
are deliberately incompatible. The separate `regional-*-4.yaml` profiles are
experimental control candidates. See [regional validation and results](regional-intervention.md).
Use explicit `config/motor8-smg1.yaml` or `config/motor8-smg2.yaml` only when reproducing the closed pilot. Historical
`config/default.yaml` and staged profiles retain legacy action semantics for
checkpoint reproduction; they are not the owner-selected eight-action default.
Configuration files should live directly under `config/`: the loader treats the
configuration's grandparent directory as the project root. YAML files are complete
configurations, not overlays.

`--help` and `<command> --help` display parser help without starting an emulator.
Current motor profiles require the native backend for exact button pulses;
desktop `train`, `launch` and visible calibration need an explicitly selected
legacy desktop-compatible profile. Old 54-action models cannot resume as eight
actions without the recorded explicit transfer described in
[the motor8 protocol](motor8-and-algorithms.md).
Example paths containing `EXAMPLE` below must be replaced with real local paths.

## Commands

| Command and arguments | Backend / effect |
| --- | --- |
| `prepare` | Copy the configured source profile and referenced states into an owned desktop profile; no game launch. |
| `launch [--slot {1,2,3,4}]` | Launch the owned visible Dolphin profile, loading the selected slot (default 1). |
| `telemetry` | Read one telemetry snapshot using the single-process memory hook; not a parallel-worker selector. |
| `calibrate [--attach]` | Visible desktop checks of configured state roles, completion, death, movement, and pause; save evidence/calibration. |
| `calibrate-background` | The same detector checks using a temporary hidden, PID-scoped worker; requires a `parallel` section; closes that owned instance. |
| `train [--attach] [--steps N] [--rollout-steps N] [--resume ZIP]` | Single visible desktop recurrent PPO learner. Requires training enabled. |
| `train-parallel [--workers N] [--steps N] [--resume ZIP]` | One hidden vectorized learner attempt; no automatic restart. Requires a `parallel` section and training enabled. |
| `train-supervised [--workers N] [--steps N] [--resume ZIP]` | Manage hidden learner attempts with paired-checkpoint recovery, timeout, retry backoff, and a fixed overall target. |
| `train-dqn [--steps N] [--resume DIRECTORY]` | Finite Rainbow/IQN real-pixel training with prioritized frame replay and TensorBoard. Workers are explicit in the profile. No world model. |
| `record-dqn DIRECTORY [--variation N] [--seeds 11 23 47] [--max-steps 300]` | Frozen canonical-start Q-policy video; actual grayscale input, raw expected Q returns and applied controls. No optimizer calls. |
| `status [RUN]` | Read the latest parallel attempt in the configured runs root, or the explicit attempt; includes owning supervisor state. |
| `stop [RUN]` | Request graceful stop of that parallel attempt and its supervisor, when present. |
| `evaluate [--attach] ZIP [--episodes N]` | Visible desktop deterministic evaluation, intrinsic reward/curiosity updates disabled; default 5 episodes per configured variation. |
| `plot CSV [--output PNG]` | Generate a top-down measured trajectory and progress figure from a CSV; no emulator access. |
| `benchmark [--steps N]` | Fixed-action environment profiling with curiosity, without PPO; default 64 decisions; results excluded from success reports. |
| `report [--output JSON]` | Audit training evidence across both games under project `runs/`; with output, save JSON and available best-trajectory artifacts. |

Desktop launch refuses to run alongside existing Dolphin processes. Desktop
commands need a foreground render window and configured Windows inputs.
`--attach` uses an already open, correctly configured visible game; it does not
attach to a hidden learner. The `telemetry` hook is suitable only when a single
intended Dolphin process is available. Use worker CSVs/status for parallel runs.

## Selecting a game

| Configuration | Use | Run root |
| --- | --- | --- |
| `config/native.yaml` | Galaxy 2 native capture/frame control, CUDA | `runs/` |
| `config/smg1-native.yaml` | Galaxy 1 native capture/frame control, CUDA | `runs/smg1/` |
| `config/default.yaml` | Galaxy 2 visible desktop, CPU | `runs/` |
| `config/parallel.yaml` | Galaxy 2 hidden workers, CUDA | `runs/` |
| `config/smg1.yaml` | Galaxy 1 hidden detector validation; training disabled | `runs/smg1/` |
| `config/smg1-training.yaml` | Separate Galaxy 1 hidden learner, CUDA | `runs/smg1/` |

Native configurations require a protocol-v3 build and use bounded exploration
rewards and CPU curiosity updates in batches of eight. `parallel.skip_present`
is false after the measured comparison; changing it is an experiment, not a
required hidden-mode switch. The [optimization guide](optimization-v2.md) lists
standalone stress, batching, comparison, replay, and plotting script arguments.
These scripts are development tools, not additional CLI subcommands.

The [current native-pipeline guide](optimization-v3.md) documents
`parallel.bridge_wakeup`, `native_capture`, `collector`, `shared_observations`,
`receive_order` and `quiet_ui`, plus live image equivalence and collection
benchmarks. Native YAMLs select event wakeups, CPU area preprocessing, quiet
native Qt notifications, and the measured ordered-pipe collector. Shared vector
slots and ready-order receipt remain optional after their comparisons failed
to improve the selected workload consistently. The ordinary SB3 collector
remains the default in legacy configurations.

`report` scans both namespaces under project `runs/`, independent of the selected
game configuration. In contrast, `status` and `stop` search only the selected
configuration's immediate runs directory. Their explicit `RUN` must be a learner
attempt under that directory, containing `controller.json`; pass a `parallel-*`
directory, not a `supervised-*` manager directory. During retry backoff the latest
attempt may be dead while the manager is alive; inspect `supervisor_status`.

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml status
.\.venv\Scripts\smg-rl.exe --config config/smg1-training.yaml status
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml status runs/parallel-EXAMPLE
.\.venv\Scripts\smg-rl.exe report --output runs/training-report.json
```

## Budgets, checkpoints, and shutdown

Training `--steps` means additional aggregate decisions, including on resume.
Rollout collection may round the final count upward. Omitted values use the
configuration's training budget and worker count. A supervised invocation fixes
its target to initial checkpoint timesteps plus the requested additional budget;
retries consume the remaining budget rather than requesting it afresh.

`--resume ZIP` requires the matching `.intrinsic.pt` file. Checkpoints must match
the game's observation/action spaces. Do not resume Galaxy 1 with a Galaxy 2
policy. The files are sequentially saved, so verify a complete pair after a crash.
The supervisor skips incomplete or structurally invalid candidates and resumes
the highest-timestep available pair. It cannot guarantee that every structurally
readable checkpoint will load successfully in a changed environment.

`stop` writes markers; saving is asynchronous. It also signals the owning
supervisor so that the next retry is not started. Foreground Ctrl+C is supported
for desktop, parallel, and supervised training. A failed worker may prevent a
final curiosity save; a policy-only emergency snapshot is not a resumable pair.
See [checkpoint details](reproducibility.md) and [recovery](damage-and-recovery.md).

## Reporting and evaluation boundaries

`report` reads files and does not control gameplay. It excludes detector fixtures,
evaluation runs, near-star starts, and invalid terminal evidence. Output warnings
identify missing or inconsistent records. `status` progress refers to each
worker's current episode; the report computes historical valid maxima from
available trajectories. Both use geometric projection, not native course percent.

`evaluate` is a separate visible-backend operation. Complete graceful shutdown
of parallel learners and close their owned instances first. For background
training checkpoints, pass the same game configuration to evaluation; the
configuration selects the correct observation shape, states, and memory profile.
The original desktop keyboard/mouse mappings remain required. Evaluation creates
its own evidence and does not count as training success in `report`.

## Checkpoint integrity and bounded recovery

New `train` and `train-parallel` saves include `model-*.pair.json` SHA-256
manifests. Keep that file with its ZIP and `.intrinsic.pt` partner. Recovery
rejects `.incomplete` publications and checksum/archive errors; complete legacy
local pairs remain supported. Only load trusted local models: SB3 metadata can
contain serialized Python objects. See [the security policy](../SECURITY.md).

`parallel.worker_timeout` bounds measured collector replies (default 120 s).
`parallel.max_stalled_attempts` bounds consecutive supervised attempts without a
new checkpoint (default 8). A permanent failure is published as `failed` and
returns an error. Progress resets that consecutive-failure count. These options
do not extend the requested total training budget.

## Experimental visual world model and authorization

`ablate-world-model --checkpoint FILE [--updates 128] [--seeds 11 23 47]
[--validation-batches 8] [--arms continued_baseline terminal_reward_weight]` runs the narrowly authorized offline Galaxy 2 screening
experiment. It requires its own scoped local authorization grant; it does not
release the global hold or create an emulator. It never updates the actor or
replaces the production checkpoint. See the [paired ablation protocol](world-model-ablation.md).

`inspect-replay` reads compatible visual archive metadata/counts without starting
Dolphin or fitting a model. `train-world-model [--steps N] [--resume world.pt]`
collects real experience and performs gated imagined actor/critic updates;
`fit-world-model [--cycles N] [--resume world.pt]` fits existing visual replay
without an emulator. Use the explicit `config/world-model.yaml` or
`config/smg1-world-model.yaml` profile. These models cannot resume PPO ZIP files.

All training commands check `.local/training-hold.json` relative to the project
root before creating an environment. The owner explicitly authorized the current
world-model experiment; its preceding hold was archived locally. A future hold
must still be honored. No training example in these docs overrides an active
hold. `status` and `stop` also accept world-model run directories;
when omitted, the run is selected by modification time across supported types.
The existing `train-supervised` command remains a PPO-pair supervisor. It is not
a supervisor for world-model `.pt` checkpoints. See [the full workflow](world-model.md)
and [the dated live record](world-model-live.md).

## Policy videos, frontier resets and horizon trials

`record-world-policy --checkpoint FILE [--variation 0] [--seeds 11 23 47]
[--max-steps 600]` records every requested attempt as local silent H.264 MP4,
with RAM telemetry overlays and CSV trajectories. It requires ffmpeg, forces
full starts, disables replay/frontier selection and never updates a learner.
The checkpoint manifest supplies the exact world-model settings. See
[the recording and archive guide](frontier-exploration.md).

Frontier resets are opt-in through `config/frontier-smg1.yaml` or
`config/frontier-smg2.yaml`, used with the existing bounded `train-world-model`
command. They require the frontier-capable native build and isolated archive
paths. Real probe decisions are separately accounted, and assisted starts are
excluded from full-start report evidence. Pilot configurations are capped;
they are not instructions to resume an indefinite training job.

`test-world-horizon --checkpoint FILE [--updates 64]` audits horizons 8 and 16
against the original prediction gates, then fits paired offline actor-only copies
at seeds 11, 23, 47 only if every audit passes. It never starts an emulator or
updates the world model. Real-game evaluation is a separate bounded recording
step. See [the protocol and observed results](horizon-trial.md).
