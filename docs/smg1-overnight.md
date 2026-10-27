# SMG1 overnight continuation

> **Historical context:** Experiment status, authorizations, schedules and live counters in this document refer to their stated dates. Later work may supersede them; consult [current evidence and next experiments](current-status.md) and check actual process state before acting.

**Superseded remaining allocation:** after the completed genetic pilot, the
owner selected [continuous GA training until07:00](genetic-overnight.md) instead
of IQN. IQN is intentionally paused and retained; do not resume it automatically.
The protocol below records its earlier authorization and history.

On October 26, 2026 the owner extended the existing six-collector Sweet Sweet
training until **October 27 at 07:00 America/Sao_Paulo**, equivalent to
**2026-10-27T10:00:00+00:00**. Progress reviews occur every three hours.
This replaces the previous two-hour duration; it does not restart learning.

## Ownership and finite dispatch

`scripts/run_smg1_overnight.py supervise` adopts the live initial supervisor
without launching a second trainer. It mirrors the initial segment's ledger,
waits for clean completion, and audits the final artifact before continuing.
The active ledger is `.local/smg1-overnight-state.json`. The original ledger,
checkpoints, migration and 296150-real/1911-attempt baseline are preserved.

Successors use the same `config/smg1-six-two-hour.yaml`, six SMG1 collectors,
shared IQN/IMPALA policy, optimizer, replay, RNG, reward and exploration schedule.
There are no SMG2 collectors, world model, curiosity or genetic algorithm.
Each successor has at most 3600 seconds and passes the existing artifact audit
before another dispatch. These mechanical audits are separate from the requested
three-hour progress analysis. At most 24 successor dispatches and 4800000 total
new real decisions from the original baseline are safety ceilings, not targets.

Every new trainer receives `--deadline-utc 2026-10-27T10:00:00+00:00`.
The collection loop checks the absolute deadline before each vector decision;
an already executing action may finish before shutdown. Final checkpoint and
emulator cleanup may finish after 07:00. Expired or timezone-free deadlines are
rejected. Scheduled expiration is a clean completion with
`stop_reason=scheduled_deadline`, not an owner-stop or failure.

Memory guards remain active. Failure, memory pressure, stale initial supervisor
or owner stop retains partial evidence and blocks successors without blind
retry. Create `.local/smg1-overnight.STOP` to stop this allocation; it also
forwards a stop to the adopted initial segment when necessary. Duplicate state
is rejected. Progress inspection is read-only:

```powershell
.venv\Scripts\python.exe scripts/run_smg1_overnight.py snapshot
```

## Evidence and assessment

During setup the initial trainer stopped with a collector EOF at 389342 real
decisions and 4344 completed attempts. The underlying worker exit remains
unconfirmed; no application crash event was found and commit headroom remained
above 44 GiB. Its immutable partial checkpoint passed the existing failed-block
audit (hashes, finite losses, real/replay/attempt accounting and reward checks).
The 184 unacknowledged CSV decisions are retained as a gap, not learner steps.
Owned orphan emulators were closed only after confirming their parents exited.
The reviewed recovery manifest permits one 3072-decision/300-second health
probe before normal successors; a repeat failure blocks dispatch. See
[the recovery audit](assets/performance/smg1-overnight-recovery-20261026.json).
The failed adoption state is archived locally rather than erased. This is a
reviewed partial continuation, not a clean initial-segment completion.

TensorBoard continues cumulative reward, sampled path distance, valid route
advance and attempt series through the existing exporter, now watching the
overnight ledger. Full-start and assisted attempts/stars remain distinct.
The mapped reward still covers only the verified opening: 100% opening is not
100% mission. Training collection maxima do not establish frozen policy quality.

The initial two-hour final assessment is superseded, not run twice. Its existing
six frozen canonical episodes are deferred until clean overnight completion and
artifact review: greedy and stochastic modes, seeds 11/23/47, at most 600
decisions each (3600 evaluation decisions total). The pending ledger is
`.local/smg1-overnight-evaluations.json`. A stop/hold prohibits evaluations.
Preserve failures, exact image/Q/control overlays and before/after hashes;
evaluation decisions are separate from training. Reviews retire after the final
report; no further training extension is allocated automatically.
