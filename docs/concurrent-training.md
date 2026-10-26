# Concurrent Galaxy 1 and Galaxy 2 training

**Current explicit authorization:** [six-instance regional training](regional-six-training.md)
uses two Sweet Sweet collectors, two Spinning collectors and two Double Time
collectors, with two independent game policies. Initial health blocks passed;
the new finite main blocks have an 8192-real-decision cap per mission. Older
resource limits and stopped-pilot statements below describe earlier experiments.

**Latest status (October 25, 2026):** the finite motor8 pilot has closed. A
separately requested [SMG1 prediction repair](smg1-boundary-repair.md) passed
fresh gates and one actor smoke; no long online or six-emulator run is active.
The [BTR/code and parallelism review](btr-research-review.md) distinguishes
collectors from independent agents. The older configurations and measurements
below are historical; do not dispatch their million-step commands automatically.

**Status note (2026-10-23):** the supervised PPO jobs described here are stopped;
their final pairs are preserved. The separate world-model experiment recorded in
[the live world-model record](world-model-live.md) was also stopped at the
owner's request on October 23, 2026; its final checkpoints are preserved and
all training remains stopped until the owner explicitly authorizes resuming.
The commands below are retained for reproducibility and intentional replacement
runs.

**Native backend update (2026-10-22):** the local learners now use
`config/native.yaml` and `config/smg1-native.yaml`, with shared-memory RGBA
capture and bounded frame stepping. The stock hotkey/PNG backend remains a
fallback. See [native controls, profiling, benchmarks, and migration](native-backend.md).
The older measurements below retain their original backend/timing context.

## Authorized configuration

On 2026-10-21 the owner authorized Sweet Sweet Galaxy training alongside the
existing Galaxy 2 learner. The conservative configuration uses **two independent
policies and five hidden Dolphin instances**:

| Learner | Tasks | Workers | Seed | Minibatch | Checkpoint interval |
| --- | --- | ---: | ---: | ---: | ---: |
| Galaxy 2 | Both Twisty Trials scenarios | 4, alternating scenarios | 42 | 256 | 1024 aggregate decisions |
| Galaxy 1 | Sweet Sweet Galaxy / Rocky Road | 1 | 142 | 64 | 256 decisions |

Both learners use CUDA for policy optimization, separate CPU curiosity models,
128 decisions per worker per rollout, and independent checkpoint/run directories.
They do not transfer weights or share novelty statistics. Galaxy 1 starts from
`RMGE01.s01`; `RMGE01.s02` remains exclusively a detector fixture. Its policy has
a one-element scenario vector, rather than Galaxy 2's two-element vector.

`config/smg1.yaml` retains `training.enabled: false` for validation. The new
`config/smg1-training.yaml` is explicitly enabled. Training authorization is
enforced by configuration rather than a hardcoded ban on the game ID.

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml train-supervised --workers 4 --steps 1000000
.\.venv\Scripts\smg-rl.exe --config config/smg1-training.yaml train-supervised --workers 1 --steps 1000000
```

These commands start fresh runs. Do not execute them again when the two learners
are already running. To resume, supply a matching checkpoint pair with `--resume`.
Each command should run in its own terminal. `--steps` requests additional
aggregate decisions, including on resume. A supervisor fixes the total target
once and uses the remaining budget after each failure; it does not reset the
requested budget on retry. Policies and curiosity models remain separate across
games. Galaxy 1 charges -0.25 per nonfatal HP lost; Galaxy 2 retains death-only
penalization for these missions. See [damage and recovery](damage-and-recovery.md).

Startup attempts have failed to locate a worker's verified RAM. Workers allow
90 seconds to boot, separate from the normal 30-second action/capture timeout.
The current Galaxy 2 configuration staggers startup by **five seconds per worker
index** (0/5/10/15 seconds); the earlier one-second spacing was insufficient in
some attempts. Galaxy 1 has one worker, so its index-zero startup adds no delay.
Boot failures remain possible and are handled by supervised retry.

## Initial concurrent resource measurement

Ten local samples with both learners active measured mean CPU utilization
13.47%, maximum CPU utilization 17.6%, at least 11.20 GiB available RAM,
maximum VRAM usage 859 MiB of 8188 MiB, and GPU temperature up to 55 C.
GPU utilization briefly reached 80% during this sampling interval. There was
about 600 GiB of free disk space. The raw local record is
`.local/concurrent-hardware.json` and is not published.

These are short operational measurements, not a sustained thermal stress test
or a guarantee of responsiveness under unrelated heavy workloads. Galaxy 2
continued around 4.5 aggregate decisions/second and Galaxy 1 around 1.3 in the
initial concurrent observation. Re-measure after changing worker count, graphics,
frame budget, power mode, or other workloads. Additional workers are not enabled
by default merely because VRAM is available.

## Current state and stopping

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml status
.\.venv\Scripts\smg-rl.exe --config config/smg1-training.yaml status
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml stop
.\.venv\Scripts\smg-rl.exe --config config/smg1-training.yaml stop
```

The two stop commands are independent and also stop their owning supervisors.
Healthy learners save final checkpoint pairs; a failed or hung worker may prevent
a final save, leaving a periodic pair as the recovery point. Periodic retention
keeps the newest three completed pairs per learner attempt. The
machine must remain running for local training; suspending it halts emulation.
The scheduled morning report also needs the local desktop app available.

## Auditable morning report

```powershell
.\.venv\Scripts\smg-rl.exe report --output runs/morning-report-20261022.json
```

The report scans only training run roots (`train-*` and `parallel-*`), including
the nested `runs/smg1/` namespace. Validation, calibration, and evaluation roots
are excluded. A trajectory must start with a live reset within 500 world units
of its calibrated start. Near-star trajectories are rejected. A success requires
the trajectory's success outcome, a completion flag, no death flag, and the
correct game-specific scene nerve. A bright image or 100% projection alone is
insufficient. Recorded-episode counts include interrupted episodes and are
distinct from successful full-course episodes.

Workers atomically publish their current episode to `live.csv` every 16 decisions
and on terminal transitions, including reset. Final episode CSVs remain the
authoritative completed records. The report includes running snapshots without
double-counting terminal live copies. It saves a fixed copy of the best recorded
trajectory and generates its measured X/Z plot. Reporting does not stop either
learner or mutate its checkpoints. Snapshot publication tolerates a temporary
Windows file lock from a viewer. Older worker status-only records are labeled
when a matching trajectory is unavailable. Malformed or missing evidence is
reported as a warning or unknown value rather than converted into a success.

The owner requested a **single report at 07:00 America/Sao_Paulo on 2026-10-22**.
It is configured as a thread automation in the desktop app, not a repository cron
job. The report does not stop training at 07:00. Its results are separated into
Spinning and Spinning and Spinning, Turning Turning Double Time, and Rocky Road.

All percentages refer to the maximum **valid high-water X/Z projection** recorded
by the reward system. This is an approximate geometric progress measure; it is
not a native completion percentage, remaining path length, or probability of
winning. In particular, Sweet Sweet Galaxy's curved route has not been mapped
to ordered waypoints. Full-course completion remains a separate RAM event.

## Subsequent damage and recovery revision

Both initial concurrent learners later stopped on pause acknowledgment failures.
The first supervised recovery used the 37512-step Galaxy 2 and 2048-step Galaxy 1
checkpoint pairs. Subsequent continuation used newer complete pairs. These
counts identify historical recovery evidence, not the latest training status.
Galaxy 1 now includes measured HP supervision and a -0.25 nonfatal damage
coefficient. Longer pause pulses and stricter counter checks trade throughput
for more reliable hotkey delivery. The initial resource/rate measurements above
are historical; query `status` and `report` for current run artifacts.
