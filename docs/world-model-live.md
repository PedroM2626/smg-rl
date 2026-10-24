# Authorized online world-model experiment

## Scope and authorization

On October 22, 2026 the owner explicitly authorized continuing the Dreamer-inspired
workflow, including collection and training. The earlier local hold was archived
with that authorization before either emulator job was launched. The preserved
PPO jobs remain stopped. No previous PPO decisions are added to these new agents.

## Running snapshot (October 23, 2026, 17:48 UTC)

Both jobs were still running at the dated snapshot. Both target 1,000,000 new
real decisions; Galaxy 2 uses four workers and Galaxy 1 one. Counters are
cumulative since each learner's current run started; the Galaxy 1 counter
includes 174,229 decisions collected before its repair resume, which requested
825,771 additional decisions.

| Game | Online run | Real decisions | Model updates | Actor updates | Imagined transitions | Best verified progress |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Galaxy 2, both variations | `runs/world-model-20261022-200121-619365` | 732288 | 22748 | 1182 | 1192992 | 19.5304% `spinning`; 13.7609% `double_time` |
| Galaxy 1, Sweet Sweet | `runs/smg1/world-model-20261023-104046-961482` | 229349 | 30508 | 421 | 431104 | 5.0624% `sweet_sweet` |

The progress values are the maximum valid RAM-coordinate high-water projection
in completed training episodes, recomputed from the run CSVs with the `report`
filters (start reset near the calibrated position, no calibration/validation
fixtures). They are geometric projections, not native game-completion
percentages. There were no successful full-start episodes. The best evidence is:

- `runs/world-model-20261022-200121-619365/worker-02/spinning/episode-001749.csv`
- `runs/world-model-20261022-200121-619365/worker-01/double_time/episode-001888.csv`
- `runs/smg1/world-model-20261023-104046-961482/worker-00/sweet_sweet/episode-000080.csv`

The Galaxy 1 world model remains below the preserved PPO record of 7.152%
(`runs/smg1/parallel-20261022-083713-478823/worker-00/sweet_sweet/episode-000057.csv`).

### Gate status at the dated snapshot

| Scenario | Posterior image MSE | Open-loop image MSE | Terminal reward MAE | Terminal Brier | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `spinning` | 0.01923 | 0.03290 | 0.56142 | 0.02957 | blocked only by terminal reward |
| `double_time` | 0.01681 | 0.03236 | 0.57966 | 0.02506 | blocked only by terminal reward |
| `sweet_sweet` | 0.01580 | 0.03635 | 0.55039 | 0.17255 | blocked only by terminal reward |

Limits: posterior MSE <= 0.02, open-loop MSE <= 0.04, terminal reward MAE <=
0.5 and terminal Brier <= 0.2. The gates are recomputed every cycle on fresh
held-out batches, so a metric can pass in one cycle and fail in the next. A
blocked cycle pauses new imagined actor updates; it does not stop collection,
discard the actor or invalidate earlier updates. At the last recorded cycle all
other checked metrics passed.

### Measured budget projection

At the measured cumulative rates (10.83 Galaxy 2 decisions/s and 3.71 Galaxy 1
decisions/s), a fresh 1,000,000-decision world-model run would need about
**25.6 h** on Galaxy 2 and about **75 h (range 63-86 h)** on Galaxy 1. At the
snapshot, the remaining budgets imply about **6.9 h** and **57.7 h**
respectively. These are projections for consuming the budget, not time to learn
or complete a mission. The Galaxy 2 run started from zero; the Galaxy 1 estimate
is extrapolated from a resumed run and earlier measurements. The rates include
model fitting, validation and checkpoint I/O. Recent PPO measurements imply
roughly 17-18 h (Galaxy 2) and 55-61 h (Galaxy 1) for the same 1M decisions
under concurrent load; the world model is slower per real decision and still has
to demonstrate that it needs fewer of them.

## Final stop (October 23, 2026)

At the owner's request, both jobs were stopped before the PC was shut down.
Each learner acknowledged the stop request, saved `world-final.pt` with its
SHA-256 manifest, and wrote a final `status.json` with `state: stopped`; neither
run directory contains `error.txt`. The local training hold is active again and
`AGENTS.md` records that all training remains stopped until the owner explicitly
authorizes resuming.

| Game | Final run | Real decisions | Model updates | Actor updates | Imagined transitions |
| --- | --- | ---: | ---: | ---: | ---: |
| Galaxy 2, both variations | `runs/world-model-20261022-200121-619365` | 870272 | 27060 | 1291 | 1302360 |
| Galaxy 1, Sweet Sweet | `runs/smg1/world-model-20261023-104046-961482` | 278773 | 36688 | 1017 | 1041408 |

At the final stop the Galaxy 1 imagination gate passed (terminal reward MAE
0.353), while both Galaxy 2 scenarios remained blocked by terminal reward MAE
only (0.634 `spinning` and 0.607 `double_time`, limit 0.5). These are the last
recorded cycle values; gates are re-evaluated per cycle. Neither job reached
its 1,000,000-decision budget or a full-start mission completion. The counters
in this section are the authoritative final values; the sections above retain
their dated mid-run snapshots.

## Initial jobs (October 22)

Two independent learners started at approximately 20:01 America/Sao_Paulo. The
Galaxy 1 job listed here was later stopped for repair and resumed; the current
Galaxy 1 run is the one in the snapshot above.

| Game | Configuration | Online run | Workers | New real-decision budget |
| --- | --- | --- | ---: | ---: |
| Galaxy 2, both Twisty Trials variations | `config/world-model.yaml` | `runs/world-model-20261022-200121-619365` | 4 | 1000000 |
| Galaxy 1, Sweet Sweet Galaxy | `config/smg1-world-model.yaml` | `runs/smg1/world-model-20261022-200121-657138` | 1 | 1000000 |

These are compact original Gaussian RSSM agents inspired by Dreamer, not a
DreamerV3 reproduction. See the [implementation specification](world-model.md)
for equations, architecture, loss functions, gates and limitations.

## Stages and measurement

Collection starts from the configured initial savestates and records actual
84 x 84 grayscale observations. Warm-up actions are random. Model optimizer
updates start only after at least 4096 compatible replay transitions and the
presence of both training and held-out episodes. An episode belongs wholly to
one split. Unflushed transitions can make the replay counter temporarily lower
than the real-decision counter.

Every 32 vector decisions, the learner refreshes replay, performs four model
updates when eligible, then checks held-out prediction quality. This corresponds
to 128 new real decisions per cycle for Galaxy 2 and 32 for Galaxy 1. It is a
different replay-to-interaction ratio across games, not a controlled comparison.
Policy updates in imagination require every scenario's gate to pass. A running
model-fitting job may therefore still have zero actor updates; this is an
explicit safety condition on model quality, not proof of policy learning.

Use the run's `status.json` for the current `real_decisions`, `model_updates`,
`actor_updates`, `imagined_transitions`, replay split, losses and per-scenario
validation metrics. The speed shown there is new real decisions per wall-clock
second for the current attempt, including fitting (since the resume point for a
resumed run), rather than imagined steps per second. It is not a measured
improvement in time to solve the mission.

The dated initial observation is retained separately in
[the live evidence record](assets/performance/world-model-live-20261022.json).
The earlier [synthetic validation record](assets/performance/world-model-validation.json)
remains evidence about software behavior only.

At approximately 20:09 local time, both learners had begun fitting real visual
data. These are the initial snapshots, not current counters:

| Game | Real decisions | Model updates | Actor updates | Imagined transitions |
| --- | ---: | ---: | ---: | ---: |
| Galaxy 2 | 6912 | 80 | 0 | 0 |
| Galaxy 1 | 4176 | 4 | 0 | 0 |

At that initial snapshot, both statuses were `running`; learner PID/create-time
identity and all five owned native worker processes were verified. All recorded
losses and validation metrics were finite, neither replay reported rejected
archives, neither run contained `error.txt`, and both periodic checkpoint
SHA-256 manifests matched. The policy gates remained closed because early image
and terminal predictions were insufficient. Fitting the model was working;
imagined policy learning had not yet been demonstrated. Imagined actor updates
were subsequently observed, and the current counts are in the dated snapshot
above. The evidence JSON retains the complete per-scenario reasons and
contemporaneous GPU resource observation.

## Runtime checks and recovery

```powershell
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/world-model.yaml status
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/smg1-world-model.yaml status
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/world-model.yaml inspect-replay
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/smg1-world-model.yaml inspect-replay
```

Both learners use CUDA and isolated hidden Dolphin profiles. No keyboard focus
is needed. ROMs, replay, checkpoints, runtime profiles and logs remain local and
ignored by Git. Five native emulator workers are expected while both jobs run.

Periodic checkpoints are saved every 4096 new real decisions; the latest three
are retained. A normal stop or caught exception saves `world-final.pt` with its
checksum/signature JSON manifest. The world-model workflow does not use the PPO
auto-supervisor. If a job fails, inspect `error.txt`, `status.json` and worker
logs, fix the cause, and explicitly resume its latest valid checkpoint:

```powershell
# Replace both placeholders; --steps is the additional real-decision budget.
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/world-model.yaml train-world-model --steps REMAINING --resume CHECKPOINT.pt
```

Use `config/smg1-world-model.yaml` for the Galaxy 1 equivalent. Never launch a
duplicate job against a still-running learner. Keep replay and manifests with
checkpoints; weights alone cannot restore the dataset. Recovery resets the game
to the initial savestate rather than restoring an exact in-flight emulator state.

## Interpretation limits

On October 23, Galaxy 1 received an objective and rollout-training repair after
its visual and terminal quality gates remained blocked. The repaired learner had
421 imagined actor updates at the dated snapshot; Galaxy 2 had 1,182. See the
[repair method and measured evidence](world-model-smg1-repair.md).

Only a real full-start trajectory with the game's RAM star-completion predicate
establishes success. Neither fitting a model, an imagined success, nor a near-star
fixture counts. Actual RAM positions remain the basis of top-down trajectories
and calibrated geometric progress. This percentage is not native mission
completion from the game. The Galaxy 2 records are not a controlled comparison
with the preserved PPO baseline, and the Galaxy 1 world model remains below the
PPO progress record. No sample-efficiency advantage, time-to-solution gain, or
full-course learned success has yet been established for this experiment.
