# Command-line reference

Run from the repository root using `.\.venv\Scripts\smg-rl.exe` or
`.\.venv\Scripts\python.exe -m smg_rl.cli`. The optional global argument
`--config PATH` goes before the subcommand and defaults to `config/default.yaml`.
Configuration files should live directly under `config/`: the loader treats the
configuration's grandparent directory as the project root. YAML files are complete
configurations, not overlays.

`--help` and `<command> --help` display parser help without starting an emulator.
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
a supervisor for world-model `.pt` checkpoints. See [the full workflow](world-model.md).
