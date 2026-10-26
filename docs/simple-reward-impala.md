# Simple regional reward and six-hour IQN/IMPALA trial

**Completed:** both original six-hour branches and all nine frozen evaluations
finished. Zero stars occurred. [Final results and Sweet Sweet diagnosis](impala-final-results.md)
separate the learned local openings from full-course failure; no continuation
is allocated and the hourly follow-up is retired after final review.

Date: October 26, 2026. This owner-authorized intervention follows the completed
[model-free comparison](model-free.md). That comparison remains archived: zero
RAM-confirmed stars and no useful frozen navigation. New configurations, replay,
weights, counters and TensorBoard series start independently. Neither historical
PPO/RSSM checkpoints nor Rainbow/IQN pilot artifacts are modified.

**Crash/recovery update:** Windows commit exhaustion stopped both games at
96000/174504 committed real decisions. The owner explicitly requested diagnosis
and resumption. [Memory-pressure recovery](memory-pressure-recovery.md) records
the Realtek helper cleanup, Torch-free collectors, pre-spawn BLAS limits,
commit-based stop guard, retained SMG2 CSV coverage gap and exact remaining
original budgets. Recovery health passed before the resumed one-hour blocks;
reward, architecture, replay identity and learner state remain unchanged.

## Objective: remove the advantage of postponing a failure

The previous gamma 0.997, step -0.001, failure -2 objective gave a stationary
failure at decision 180 a discounted return of approximately -1.307, versus
-1.937 at decision 14. Its potential shaping closed to zero at failed canonical
starts. Consequently, delaying a failure was preferable even without curiosity.
This is a verified incentive, not proof of the unique cause of the old policies.

The new finite-episode objective has no temporal discount and no living cost:

```text
gamma = 1
reward = newly confirmed opening-route fraction
       + 20 if a RAM-confirmed star is collected
       -  2 if Mario dies OR the episode's declared idle/task budget expires
       - 0.25 per nonfatal HP lost (SMG1 only)
```

The grounded ordered route tracker already maintains a high-water mark. The
credit accumulator pays only positive increases from the episode's actual start.
The entire mapped opening pays at most +1; death/timeout remains -2. An assisted
start does not receive credit for its existing position. Retracing, stationary
animation, spin, crouch and airborne shortcuts cannot manufacture new route
credit. Route geometry, grounded checks and native completion detectors are
unchanged. Missing route geometry is not extrapolated.

For any failed episode with no progress or HP loss, return is exactly -2,
whether it takes 1, 14, 180 or 3000 decisions. A failed half-opening attempt
returns -1.5 irrespective of its duration. This removes the waiting advantage
without adding an incentive for immediate suicide. It does **not** force motion:
idle and immediate failure tie when neither makes progress. No time penalty was
added to reverse one exploit into another. Gamma1 is an episodic experimental
choice, not an assertion that undiscounted Q-learning is universally preferable.

This is intentionally **dense auxiliary progress reward**, not policy-invariant
potential shaping: terminal failure does not claw back confirmed progress. The
logs retain the historical `regional_shaping_reward`/`shaping_potential` column
names for compatibility, but `mode=advance` and reward revision
`regional-advance-undiscounted-v2` explicitly identify the changed semantics.
Signed replay rejects the old reward. A failed full opening still has negative
return; a confirmed star dominates the entire opening reward.

The return is now duration-neutral for a fixed set of progress/damage/outcomes.
All episodes remain finite: 2160 idle physical frames or 36000 total physical
frames, and native death/star conditions. Collector checkpoint cuts bootstrap;
they are not failed episodes and never incur a synthetic death penalty.

Controls, return calculations and negative cases are recorded in
[the objective audit](assets/performance/simple-impala-objective-audit-20261026.json).
Analytic controls and synthetic benchmarks do not establish game learning.

## Visual encoder and optimizer choice

The new `encoder=impala` is an original implementation of an IMPALA-style
residual convolutional encoder: channels 32/64/64, each stage a 3x3 convolution,
3x3 stride2 maxpool and two preactivation residual blocks. Adaptive maxpool
produces 6x6 features, followed by a 512-unit projection. The IQN cosine
embedding, dueling NoisyLinear heads and eight exclusive controls remain.
Observations remain four grayscale 84x84 frames plus scenario identity. RAM does
not enter the neural input. No world model, synthetic transition or novelty
module is constructed.

The [BTR paper, Appendix E](https://arxiv.org/pdf/2411.03820v2#page=21)
motivates residual visual features. This experiment is **not full BTR**: it does
not add its full architecture/loss/normalization recipe or reproduce its Atari
rankings. The Nature encoder remains the default for old profiles/checkpoints;
encoder selection is part of checkpoint settings and incompatible resumes fail.

On the local RTX4070 Laptop GPU, a synchronized random-pixel diagnostic measured
Nature versus IMPALA median forward/backward batch32 costs of approximately
2.37/10.26ms and batch2 inference of 0.64/1.32ms. These exclude replay, double-Q
targets, quantile loss, optimizer and Dolphin; they are not training throughput
or evidence of better features. Parameter counts were 2,779,314/2,664,370.
IMPALA costs more convolution work despite fewer total parameters.

IQN remains the gradient-based candidate because it supports real-transition
reuse, avoids C51 support clipping and had lower measured update overhead in the
small archived trial. No local evidence establishes a genetic alternative as
better. Population evaluation would consume additional expensive Dolphin
rollouts; no genetic experiment is silently substituted or reported as tested.
The [Deep GA paper](https://arxiv.org/abs/1712.06567) establishes feasibility,
not superiority for these tasks. A future equal-emulation-budget comparison
would require its own fitness, population, seeds and frozen evaluations.

Exploration now stays random through 2048 aggregate real decisions, then decays
linearly from epsilon1 to0.1 over100000 SMG1 or200000 SMG2 aggregate decisions.
This equalizes the decay scale per mission across the two games. NoisyLinear
exploration remains enabled. Final frozen evaluations use epsilon0 to distinguish
the learned policy from lucky random movements. This bundle changes reward,
encoder and exploration together: it is a practical intervention, not an ablation
that can attribute improvement to one component.

## Finite schedule and ownership

Profiles: `config/impala-iqn-smg1.yaml` and `config/impala-iqn-smg2.yaml`.
There are two independent game policies and six collector emulators: two Sweet
Sweet, two Spinning and two Double Time. SMG2 counts aggregate both missions.
Existing curated start banks and their 50% mixture remain signed and unchanged.

1. Fresh health blocks:3072 SMG1 and6144 SMG2 real decisions,1536 per worker,
   each with an explicit900-second wall budget. No old parameters/replay transfer.
2. Inspect both actual completed health runs, exact counters, losses, attempts,
   reward components and checkpoint hashes before any successor.
3. A first15-minute main block per game, then blocks of at most60 minutes.
   Every block closes its owned emulators, flushes pending replay, saves a new
   immutable checkpoint and passes a mechanical accounting/reward review before
   its successor. All main blocks resume only their own new branch.
4. Total accounted training wall budget is21600 seconds per game, including
   health/startup and the close/checkpoint time included in final run status.
   Separate inter-block review/resume overhead is outside that counter. Independent absolute
   real-decision safety caps are500000 SMG1 and1000000 SMG2 aggregate. These are
   ceilings, not targets or permission to exceed six hours. The earlier boundary
   wins. Errors, owner stop or integrity failures stop successors, retaining
   partial collection. No blind retry or indefinite restart is permitted.

`scripts/run_impala_trial.py health` performs the once-only dispatch. After both
health blocks finish, `supervise` audits them and runs the finite main schedule.
`snapshot` is read-only. State is atomically recorded at
`.local/impala-six-hour-state.json`; each command is durable before dispatch,
and actual controller PID/creation time/run is retained. The supervisor itself
has PID/creation-time ownership. It never dispatches a completed budget again.

```powershell
.venv\Scripts\python.exe scripts/run_impala_trial.py snapshot
New-Item -ItemType File .local/impala-six-hour.STOP
```

The STOP file prevents successors and requests active owned runs to checkpoint.
It is an owner stop, not an instruction to kill unrelated processes. The hourly
chat review inspects the supervisor; it never launches duplicate training or
extends these allocations. Pause/retire the follow-up when the finite trial ends.

## Attempts, trajectories and evidence

`attempts.jsonl` records every completed attempt with mission, full/assisted start,
worker episode, cumulative real decisions, optimizer age, return, duration,
grounded opening advance and RAM result. Per-mission/start attempt counters and
the latest100 opening advances persist through checkpoints. Interrupted
collector cuts remain CSV evidence but are not counted as completed attempts.

TensorBoard at `http://127.0.0.1:6006/` now provides both:

- `episodes/...` on a cumulative real-decision axis;
- `attempt_axis/...` on that mission/start's completed-attempt axis, including
  return, duration, RAM stars, opening advance and last100 mean advance.

Training status includes cumulative completed attempts, death/timeouts/star
counts, exploration epsilon and recent/best opening advances for each start
group. Active episode route progress remains visible too. Separate full-start
and assisted data; never aggregate an assisted anchor into claimed initial-state
success. Compare rolling means/distributions and frozen-policy evaluations,
not just a single best random rollout. Existing top-down world X/Z trajectory
plots and all episode CSV files remain retained.

The routes cover only short verified openings.100% opening coverage is not100%
mission completion. Six hours is a resource allocation, not a promise of learning
or a star. Hourly reviews must explicitly inspect stalls, action occupancy,
Q/loss scale, deaths/HP, exploration age and active as well as completed attempts.
At completion, review artifacts before any separately recorded canonical frozen
evaluations; no training extension is allocated by lack of progress.

## Initial actual-game review

Both fresh health blocks completed and passed the signed-checkpoint/replay,
physical-frame, episode and reward-component audit before main dispatch:

| Game | Real decisions | Optimizer updates | Completed attempts | Decisions/s |
| --- | ---: | ---: | ---: | ---: |
| SMG1 |3072|256|35|12.19|
| SMG2, aggregate |6144|1023|78|19.65|

There were zero RAM stars and no runtime errors. Best canonical collection
advances were26.15% Sweet Sweet,51.86% Spinning and52.26% Double Time of their
short mapped openings. These include mostly random exploration and are **not
frozen-policy learning results or whole-mission percentages**. Full and assisted
attempt groups, all eight action counts and retained trajectories are in the
[actual health evidence](assets/performance/simple-impala-health-20261026.json).
Portable validation passed199 tests and Ruff before the main schedule started.

At clean completion only, the declared final evaluation allocation is three
canonical-start seeds11/23/47 per mission, at most600 decisions each, epsilon0,
with raw-Q/exact-image/control overlays and unchanged checkpoint weights.
These at most5400 evaluation decisions are separate from training and must be
recorded pending/running/completed before/after dispatch. Owner stop/hold blocks
evaluation too. No automatic training extension or genetic run is allocated.
The separate `.local/impala-six-hour-evaluations.json` ledger owns evaluation
dispatch; do not edit the training supervisor's state concurrently to record it.

## Reward and distance in TensorBoard

The original writer already exposes `episodes/.../return` and
`attempt_axis/.../return` for total episode reward, plus `opening_advance` as a
fraction of the short verified opening. Those values were confirmed in actual
event files; they were not explicit path-distance series.

`scripts/export_training_metrics.py` now reads the original attempt ledger and
episode CSVs without modifying/restarting collectors, reward, replay or weights.
It backfills completed attempts from the current trial and follows new ones.
Its continuous TensorBoard runs, spanning all reviewed blocks, are:

- `telemetry/RMGE01/simple-impala` for SMG1;
- `telemetry/SB4E01/simple-impala` for both SMG2 missions, with separate tags.

Under `by_attempt/MISSION/START_KIND/` and `by_real_decision/MISSION/START_KIND/`:

| Scalar suffix | Definition |
| --- | --- |
| `episode_reward` | Sum of actual rewards in that completed attempt |
| `path_distance_xz` | Sum of horizontal distances between recorded RAM positions, including backtracking |
| `path_distance_xyz` | Corresponding3D sampled path length, including vertical movement/jumps |
| `net_displacement_xz` | Start-to-last nonterminal horizontal displacement; omitted after an invalid/discontinuous sample |
| `valid_route_advance` | Confirmed ordered grounded opening advance from that attempt's start, in world units |
| `opening_advance_percent` | Percentage of the short mapped opening only |
| `best_valid_route_advance` | Best completed collection advance in that mission/start group |
| `episode_reward_mean_last100` | Mean reward of the latest at most100 attempts in the group |
| `path_distance_xz_mean_last100` | Corresponding mean sampled horizontal path length |
| `valid_route_advance_mean_last100` | Corresponding mean valid forward route advance |
| `excluded_position_samples` | Invalid, terminal or discontinuous samples omitted from path integration |

Distances use game world units, not calibrated meters. They are sampled every
decision, not integrated at every game frame, and can underestimate curved paths.
Reset boundaries are never connected; death/star cinematic jumps and steps beyond
the configured maximum telemetry displacement are excluded. Path length can grow
while the agent goes in circles, jumps or backtracks: only the separate grounded
route metric represents verified forward progress. No metric is a native full
mission percentage, a new reward term or proof of a learned star.

The exporter refreshes every15 seconds and logs when an attempt has completed.
Its own `.local/impala-telemetry-state.json` records export/controller ownership
separately from the training supervisor. Normal restarts retain last exported
attempt indices and rolling windows; missing attempts fail rather than silently
fabricating an axis. It stops when the trial completes/stops/needs review or at
its independent seven-hour ceiling. Its failure cannot stop or change a learner.
The exporter-specific `.local/impala-telemetry.STOP` stops only telemetry.
Five focused portable tests passed for loops versus net progress, vertical motion,
terminal/discontinuous positions, backfill/resume axes and missing-attempt refusal;
Ruff passed. [HTTP verification evidence](assets/performance/tensorboard-distance-20261026.json)
confirms that the running TensorBoard server actually serves the new scalars.

```powershell
# One-time backfill; do not start a second exporter while the watcher is alive.
.venv\Scripts\python.exe scripts/export_training_metrics.py

# Explicit finite watcher, already running for the current trial.
.venv\Scripts\python.exe scripts/export_training_metrics.py --watch --max-seconds 25200 --interval 15
```

## First hourly review: October26,11:42 America/Sao_Paulo

The [read-only review](assets/performance/impala-hourly-review-20261026T1441.json)
confirmed six owned emulators, actual live supervisor/controller creation times,
fresh statuses, empty current error logs, finite optimization/Q metrics and valid
agent/replay hashes for both completed blocks in each game. The current one-hour
blocks were still running; no successor, evaluation or settings change was
dispatched by this review. Pending real/replay differences were the expected
four/eight n-step decisions, not rejected experience. The telemetry exporter was
healthy with1213 completed attempts exported across both games.

| Game | Real decisions | Optimizer updates | Completed attempts | Current epsilon |
| --- | ---: | ---: | ---: | ---: |
| SMG1 |51790|12435|472|0.5523|
| SMG2, aggregate |88968|21729|741|0.6089|

No RAM stars occurred. Canonical collection maxima were37.46% Sweet Sweet,
100% Spinning and52.29% Double Time **of the short mapped openings only**.
Spinning full-start attempt158, worker00 episode72 at real decision55028,
confirmed the opening exit with33 grounded credited samples, then died after273
decisions. Its exploration epsilon was approximately0.762. This verifies a real
exploratory route traversal, not a learned full-course solution or a frozen
policy improvement. The exact CSV, actor-update age, actions and RAM scene are
retained in the evidence.

SMG1 needs continued scrutiny: canonical mean opening advance in the latest50
completed attempts was5.74%, versus13.91% in the first50. Its single best trace
does not establish sustained improvement. SMG2's corresponding first/latest50
means were15.70/21.64% Spinning and15.31/23.84% Double Time. These changing-policy,
changing-epsilon collection averages are not a controlled efficacy comparison.
No action occupied more than19% of the current-block aggregate choices, but
high remaining epsilon can mask a weak greedy policy. Logged Q means were near
-1.50 in both games, with finite losses/gradients rather than a numerical blow-up.

The decision is to retain the existing finite budget and inspect the SMG1 stall,
action/Q calibration and route coverage again after further allocated data.
No extra fitting, genetic run, reward changes or early evaluation was authorized
by this review. Health/data integrity is distinct from useful policy learning.
