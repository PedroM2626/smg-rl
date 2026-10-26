# SMG1-only continuation: six collectors for two hours

Date: October26,2026. The owner requested concentrating all six collectors on
Sweet Sweet for a two-hour continuation. SMG2 is not solved: frozen policies
traverse its mapped openings and then die, with zero stars. Its jobs remain
stopped and its artifacts are preserved.

## Source, migration and unchanged task

One shared IQN/IMPALA learner resumes at296150 committed real decisions,73525
optimizer updates and1911 completed attempts. Online/target weights, optimizer,
CPU/CUDA RNG, exploration schedule, eight actions, reward and curated-start
mixture remain unchanged. `config/smg1-six-two-hour.yaml` changes worker count
and owned output/profile paths. Its replay signature and `dqn` settings match
the original profile. The original checkpoint is immutable.

Replay worker count cannot be changed implicitly: its frame capacity depends on
collector count. `expanded_workers` requires flushed cuts and remaps every
retained generation ID into the larger ring, preserving previous-frame links.
Physical-slot copying would corrupt wrapped histories. All50000 transitions,
priorities and sample RNG were preserved; frame capacity grew100044 to100100.

The migration checkpoint at `.local/smg1-six-source/checkpoint-six-workers`
has the same agent SHA256 as its source:
`2cf9e0180bf21e0469e6d570b9a721c8b3f56fcb984838fa3cd4ac58e1e6a0c2`.
Deep equality checks covered model/optimizer/RNG/counters, and all retained
frames/links/transitions were compared. This migration added no learning.

The reward still supervises only a short verified opening. No route expansion,
new demonstration, reward change or extra intrinsic bonus was silently enabled.
That incomplete coverage remains a known task-design limitation.

## Finite schedule and ownership

`scripts/run_smg1_six.py supervise` refuses duplicate state, owner stop files and
existing emulators. It records pending commands before dispatch, then actual
run/controller identities. Each clean block is audited before a successor:

1. Health: at most3072 new decisions,512 per collector,300-second wall ceiling.
2. At most30 minutes, then at most60 minutes, then only the remainder.
3. Total accounted wall budget7200 seconds, including health/startup and final
   close/checkpoint time. Inter-block review/restore overhead is separate.
   A720000-new-decision safety cap and eight-dispatch limit are ceilings, not
   targets or permission to extend two hours.

Windows commit/physical memory guards, lean Torch-free collectors and pre-spawn
single-thread BLAS remain active. Failure, pressure or owner stop retains partial
data and prevents successors; there is no blind retry or extra fit allocation.
The completed six-hour and stochastic-comparison state files are not modified.

The active ledger is `.local/smg1-six-two-hour-state.json`. Its296150/1911 source
baselines distinguish new decisions/attempts from previous training. Full-start
and assisted RAM stars, deaths, HP, paths and opening advances stay separate.

```powershell
.venv\Scripts\python.exe scripts/run_smg1_six.py snapshot
New-Item -ItemType File .local/smg1-six-two-hour.STOP
```

## Metrics and final assessment

New learner events have distinct run directories. The read-only telemetry
exporter was backfilled through the source, then pointed to this phase with
`--training-state .local/smg1-six-two-hour-state.json` and a finite watcher.
Continuous `telemetry/RMGE01/simple-impala` reward, path/route distance and attempt
series preserve cumulative indices. SMG2 receives no new data. No metric changes
the reward or performs optimization.

Only after clean completion and review, the declared final assessment may run
seeds11/23/47 in frozen greedy and stochastic modes, at most600 decisions each:
at most3600 separate evaluation decisions. Preserve failures and checkpoint/
optimizer/replay hashes. The separate ledger is
`.local/smg1-six-two-hour-evaluations.json`; owner stop/hold blocks evaluations.
No automatic training extension is allocated for a zero-star outcome.

Success requires native completion RAM; assisted starts are not canonical
successes.100% mapped opening is not finishing Sweet Sweet. Zero stars after
two hours warrants investigating coverage, exploration, motor control,
observation and optimization, but does not prove a defect in the ROM, mission
or SMG1 engine. The hourly follow-up stays quiet on healthy unchanged state and
retires after this finite allocation and its final assessment.

## Initial verification

215 portable tests and Ruff passed, including wrapped-ring expansion, retained
sampling RNG, pending/shrinking rejection, task-identity equality and duplicate
phase refusal. The actual six-emulator health block completed3072 new decisions
and passed audit before the30-minute successor started. No synthetic fixture or
collector migration is counted as a trained completion.
