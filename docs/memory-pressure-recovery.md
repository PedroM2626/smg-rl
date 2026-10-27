# Windows commit-exhaustion diagnosis and bounded recovery

> **Historical context:** Experiment status, authorizations, schedules and live counters in this document refer to their stated dates. Later work may supersede them; consult [current evidence and next experiments](current-status.md) and check actual process state before acting.

Date: October26,2026. This records the owner's explicit request to identify the
crash cause and resume the [six-hour IQN/IMPALA trial](simple-reward-impala.md).
It does not allocate six additional hours or change any learning objective.

## Cause and evidence boundaries

At12:37:20 America/Sao_Paulo, Windows System event2004 from
`Microsoft-Windows-Resource-Exhaustion-Detector` reported insufficient virtual
memory. Its largest named consumers were `RtkAudUService64.exe` PID9788 at
9,458,786,304 committed bytes and two learner Python processes at3,972,005,888
and3,961,274,368 bytes. System event26 simultaneously warned of low virtual
memory. Dolphin and the desktop app failed in the same second. Dolphin's
Application1000/WER1001 records show `ucrtbase.dll`, exception`0xC0000409`,
subtype7. Collector EOF/BrokenPipe errors followed the native process exit.

This establishes system-wide commit exhaustion at the failure, rather than a
reward/optimizer exception. The
[Microsoft fail-fast documentation](https://learn.microsoft.com/en-us/cpp/intrinsics/fastfail)
explains that`0xC0000409` is a fatal fast-fail family; the code alone does not
prove a native bridge buffer overflow. An individual native allocation/abort
stack was not symbolicated, so no unsupported exact C++ throw site is claimed.

During diagnosis the same Realtek helper reached approximately28GiB private
commit. Windows commit approached90GiB, with approximately38GiB paged kernel
pool and7GiB nonpaged pool. Free physical RAM alone did not reveal this failure
mode. The helper was a signed per-user`-background` process owned by Acer and
parented by Explorer, distinct from system service PID5852. Its executable had
a valid Microsoft Windows Hardware Compatibility Publisher signature.

Only that leaking per-user helper was terminated and relaunched under the
owner-authorized cause repair. The system audio service, startup configuration,
pagefile settings, drivers and unrelated applications were preserved. Allocation
release was delayed: initial samples still showed retained process/kernel
memory after process exit. Later verified samples returned to approximately
2.3GiB paged and1.7GiB nonpaged pool, with39–43GiB commit headroom while all six
emulators were active. The replacement helper used approximately5MiB private
commit. This is a successful observed cleanup, not proof that an external driver
can never leak again. Future pressure stops project work safely.

## Project fixes

The previous collectors unpickled SB3 wrappers/factories and imported the full
PyTorch/CUDA runtime even with curiosity disabled. NumPy's default BLAS thread
creation also reserved substantial per-process Windows memory. Neither was
needed to execute the model-free environment; only the two learners own Q nets.

- `LeanWorkerFactory` and a small raw-Gymnasium command loop avoid Torch/SB3 in
  model-free collector processes. Legacy PPO/RSSM transport remains available.
- Curiosity imports occur only when intrinsic learning is actually enabled.
- BLAS/OpenMP thread limits are set **before spawn/import**, and before learner
  subprocess launch. Setting them after NumPy import cannot reclaim existing
  Windows thread-stack reservations. Parent environment values are restored
  after each collector spawn.
- Spawned-collector tests compare actions/rewards/autoresets/terminal pixels to
  SB3 and verify no Torch/SB3 import in the child. Four-frame observations,
  scenario IDs, ordering and all signed reward/action/learner settings remain.
- `GetPerformanceInfo` supplies Windows commit total/limit/headroom and kernel
  pool diagnostics. Before starting an owned block and every64 vector ticks,
  the learner checks at least6GiB commit headroom and512MiB physical availability.
  Low headroom closes owned environments, flushes committed n-step replay and
  saves a checkpoint with`stop_reason=memory_pressure`; no successor is allowed.
  This reserve is an engineering safeguard, not an OS guarantee.
- TensorBoard/status now expose commit headroom. No external process is killed
  by the guard and it never silently retries a failed/pressure-stopped block.

Actual restarted collectors reported approximately61–62MiB private memory each,
with`torch_imported=false` and`sb3_imported=false`. A cold environment import
with single-thread BLAS used approximately57MiB. These are measured collector
footprints, not a claim that the learners or Dolphin use that little memory.
A side-by-side cold-import comparison measured1,600,372,736 private bytes for
the legacy factory/transport import path versus60,743,680 for the lean imports
(27 versus4 process threads). The approximately4GB Python processes named by
the Windows exhaustion event were learners, not a direct measurement of every
collector's footprint. These distinct measurements must not be conflated.

## Retained partial data and exact continuation

Both failed final checkpoints passed agent/replay SHA256, signature/settings,
real/replay count, completed-attempt and reward-component checks before recovery.
The original failed jobs remain`failed_accounted`, retaining their errors,
status, logs, checkpoints and recovery audits. A pre-recovery state snapshot is
retained locally; no historical artifact is overwritten.

| Game | Committed real decisions | Completed attempts | Already consumed seconds | Remaining original seconds |
| --- | ---: | ---: | ---: | ---: |
| SMG1 |96000|710|7311.3838378|14288.6161622|
| SMG2, aggregate |174504|1241|7384.6418894|14215.3581106|

SMG1 CSV coverage reconciled exactly. SMG2 trajectory CSV coverage was46 decisions
short of committed replay and contained one additional terminal CSV not
acknowledged by the parent. A worker can finish/write an episode while another
worker fails before the vector reply is committed. All acknowledged completed
attempts retained their CSV files; the missing coverage concerns active tails.
No missing RAM positions or reward traces were invented. The complete hashed
replay/agent counts define the continuation source; the extra unacknowledged
CSV is retained as failure evidence, not inserted into replay or counted as a
learned completion. There were no stars in either failed checkpoint.

`run_impala_trial.py recover` is an explicit once-only owner-authorized operation,
not a retry loop. It charges the failed block's elapsed time exactly once,
preserves its committed counts and restores only its own new branch. It does
not resume older pilots or reset exploration, optimizer, targets or attempts.

The resumed supervisor starts with at most512 additional decisions per worker
and180 seconds per game, then audits that finite recovery health block before
reviewed one-hour successors. New log labels avoid reusing previous budgets.
The six-emulator limit, original21600 accounted seconds per game and500000/
1000000 real safety caps remain. Final evaluations are still allocated only
after both clean training completions and no owner stop.

The recovered health blocks completed with1024 new SMG1 and2048 new SMG2
decisions and passed the actual-game audit before`block05` began. The hourly
follow-up and finite read-only reward/distance exporter were resumed with the
same scope. The exporter remains separate from learning and replay.

## Verification

The full portable regression suite subsequently passed209 tests, and Ruff
passed. [Recovery evidence](assets/performance/memory-pressure-recovery-20261026.json)
retains failed-source hashes/counts, the46-decision CSV coverage gap, recovery
health reviews, actual new controller identities, measured collector footprints
and post-recovery Windows commit/pool counters.

34 focused portable tests passed, covering Torch-free spawned collector
equivalence, terminal observations, clean memory-pressure checkpointing,
once-only partial-budget accounting, regional rewards and native protocol
checks. Ruff passed. The live recovery audit verifies real games, source counts,
new checkpoints and collector footprints. These checks establish infrastructure
recovery, not learned mission completion or an algorithm efficacy result.
