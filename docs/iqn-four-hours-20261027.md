# Four-hour six-worker IQN allocation

**Interrupted, not completed:** a curriculum reset exhausted C: at approximately
13:30 BRT. Final checkpoint writing also failed. The last valid checkpoint is
749006 real decisions; all training is stopped and no retry/extension was run.

On October27 the owner explicitly requested focusing on IQN with six workers
for four hours and returning the results. Only Sweet Sweet Galaxy resumes.
All genetic runs, SMG2 jobs and world-model/PPO checkpoints remain retained and
stopped; there is no algorithm switch or extra fitting inside this allocation.

## Exact source and budget

Resume the latest reviewed IQN pause checkpoint at **571424 real decisions,
142344 optimizer updates and8779 completed attempts**, not either genetic
champion. The launcher re-audits its clean owner-stop hashes, replay, CSV,
reward and episode accounting before dispatch. Online/target weights, optimizer,
replay/PER, CPU/CUDA/NumPy RNG, exploration schedule, eight actions, frame
skip12, canonical/curated-start mixture and original reward remain unchanged.
The IMPALA online visual encoder remains trainable; the separate target network
is intentionally gradient-free. Real first-update probes verify weight changes.

The finite supervisor uses `--training-seconds 14400`: **14400 accounted seconds
including health, startup and final close/checkpoint time per segment**. A
3072-decision/300-second health block precedes a reviewed30-minute block, then
reviewed at-most60-minute blocks and only the remaining allocated time. The
1440000-new-real and12-dispatch caps are safety ceilings, not training targets.
Source checkpoint restores and inter-block audits add small overhead outside
the accounted runtime. The expected finish is approximately15:15 BRT after an
11:12 start; a one-time result review is scheduled for15:30 BRT. No automatic
successor or new frozen-policy evaluation is allocated.

Each completed segment must pass the existing exact artifact/real/attempt
audit before a successor. Original quality gates remain unchanged. Duplicate
state, active emulator ownership conflicts, owner stop, memory pressure and
failures block dispatch; retain any partial collection without blind retries.
All processes have explicit finite budgets and hidden owned emulator profiles.

The active state is `.local/iqn-four-hour-20261027-state.json`. Create
`.local/iqn-four-hour-20261027.STOP` to stop this allocation. Inspection:

```powershell
.venv\Scripts\python.exe scripts/run_smg1_six.py snapshot --state .local/iqn-four-hour-20261027-state.json
```

## Whole-coordinate-span progress, not opening completion

The owner specifically requested the furthest advance in the whole mission.
The result report must not substitute100% mapped opening for that question.
`scripts/report_iqn_progress.py` scans saved **canonical** IQN training episode
CSVs, verifies finite/continuous positions, credits only alive-grounded samples,
and computes maximum X/Z projection from the original calibrated start toward
the near-star goal. It uses the same geometric definition as the GA comparison.
Assisted starts are excluded; saved collector cuts retain explicit partial
labels. Native completion requires the validated RAM scene, independently of
this geometric score. While training runs, active unflushed episodes are not
covered; all saved episodes are available after clean segment closure.

This remains a geometric estimate across the start/goal coordinates, not a
native game percentage or a calibrated fraction of the curved full-course path.
It is also a collection maximum under changing weights/epsilon, not a frozen
policy evaluation. Report the mapped opening only as a separately labeled
secondary metric, and separate canonical versus assisted native stars.

The source baseline audit found **6.3192979740% global grounded projection**
before this allocation, compared with the historical GA8.0664975654% estimate.
The new phase is tracked independently; prior trials are never charged again.
Generate the before/after report without controlling an emulator:

```powershell
.venv\Scripts\python.exe scripts/report_iqn_progress.py --state .local/iqn-four-hour-20261027-state.json --output docs/assets/performance/iqn-four-hour-goal-progress-20261027.json
```

The baseline scan includes the recorded Nature/IMPALA IQN lineage with source
real counts below571424. New phase runs come only from this explicit ledger.
Calibration, near-star states, GA runs and recordings are excluded. Synthetic
fixtures never establish a trained star or a mission-completion percentage.

## Telemetry and interpretation

The exporter explicitly watches this allocation, preserving cumulative
TensorBoard reward, sampled X/Z/XYZ distance, valid route advance and attempt
axes. Source episode indices were reconciled at4475 assisted and4304 canonical
attempts before restarting the watcher. New optimization and trajectory events
retain distinct run directories and exact real/optimizer counters.

The original IQN dense reward still covers only the verified opening, with
native star/death/HP terms. This known guidance limitation is not silently
changed by extending training. Four additional hours are an experiment, not a
guarantee of progress. The final review must report actual duration, new and
cumulative decisions/attempts, whole-coordinate-span estimates before/after,
native stars, damage/deaths, action occupancy and the best top-down trajectory,
including failures and missing evidence. No extra budget is implied by zero
stars or a plateau.

## Initial verification

The actual health block completed3072 new decisions and passed its artifact
audit before the30-minute successor. The first real visual update142345
(source142344) changed the first convolution kernel, with gradient L2
0.0189908985. Six owned emulators, fresh advancing status, the explicit telemetry
watcher and TensorBoard HTTP200 were verified without errors.
244 portable tests and Ruff passed, including the four-hour/default-budget
boundaries, ground-gated global projection versus opening credit, assisted-start
exclusion, native RAM terminal validation and keeping a valid native completion
independent from an invalid geometric sample. Synthetic fixtures are not trained
outcomes. See [the source/dispatch evidence](assets/performance/iqn-four-hour-authorization-20261027.json).

## Failure and actual results

Three segments completed and passed re-audit: health,30 minutes and60 minutes.
Their accounted runtime was5623.076 seconds (1h33m43s). The next segment failed
after about44 minutes, for approximately **2h18 attempted**, not four hours.
The latest status lower-bound sum is8261.190 seconds; its final few seconds
were not published after checkpoint serialization failed. Approximately1h42 of
the nominal allocation was not dispatched. No attempt was made to recover that
time automatically after the failed block.

| Accounting | Source | Latest valid checkpoint | Latest acknowledged observation |
| --- | ---: | ---: | ---: |
| Real decisions |571424 |749006 |846674 |
| New real decisions |0 |177582 |275250 |
| Completed attempts |8779 |12125 |13553 |
| New completed attempts |0 |3346 |4774 |

The97668 acknowledged decisions after749006 are real observations but have no
usable final policy/replay checkpoint. The failed directory contains an
`.incomplete` marker and a1150976-byte truncated `agent.pt`, with no valid
manifest or final replay. It must never be loaded as a resumable policy.
The final legacy status misleadingly remained `running` at846542 decisions
because checkpoint saving raised before status publication. The collector
ledger proves the higher846674 acknowledged total; that does not make the
weights recoverable. Raw status/error files are preserved as failure evidence.

There were4774 acknowledged completed attempts,4539 acknowledged deaths and
**zero canonical or assisted RAM-confirmed stars**. Available CSVs recorded no
nonfatal HP loss. They cover148 fewer decisions than the acknowledged ledger,
so that incomplete tail is explicitly missing evidence, not invented telemetry.

The observed best canonical goal projection rose from **6.3192979740%** before
the allocation to **6.5860194019%**, a0.26672 percentage-point increase.
The best point was X770.4669,Y-50.0002,Z-2941.6125 at episode decision134,
followed by death. It belongs to the failed uncheckpointed block, so this is a
real saved trajectory, not a recoverable best-policy checkpoint. The completed
segments alone reached6.0092333167%; the earlier lifetime maximum was6.3193%.
The historical GA maximum8.0665% remains higher under the same geometry, but
these are different collection experiments, not a controlled algorithm ranking.
None of these values is a native mission percentage or full-route path fraction.
No frozen-policy evaluation was run.

## Root cause and implemented prevention

The worker failed in `NativeDolphin.load_state` while copying a curated
savestate, with `OSError(28, No space left on device)`. The learner then received
EOF and failed again while writing its final checkpoint. This was not a
reported GPU/gradient/RAM failure. The first convolution updated successfully
in the failed block as well, so frozen IQN vision does not explain this stop.

The restore implementation renamed every previous **disposable reserved-slot9
copy** to a new UUID-retained file on every reset. The IQN profile subtree now
contains5862 such retained slot copies totaling225667442367 bytes (225.7GB,
about210.17GiB). They are retained staging copies; no claim that all contents
are identical is made without hashing them. At review C: had about1.32GiB free,
below the new4-GiB reserve. D: had about3.82TB free. No file was deleted,
archived, relocated or silently replaced to free space during this review.

The code now stages a restore into a unique temporary file, preserves an
initially inherited slot9 at most once, and atomically replaces only its own
subsequent staging copy. A failed copy leaves the original slot/source intact
and removes only that call's incomplete temporary. The 5862 historical retained
files were untouched during the failure review; a subsequent owner-requested
cleanup is documented below. Output/profile volumes now require4GiB free at IQN
preflight and are checked every64 vector ticks; disk pressure triggers a clean
stop while space remains for a checkpoint. Finalization errors now attempt to
publish an explicit failed/checkpoint-failure status instead of leaving a stale
healthy-running state. A completely full volume can still prevent publication.

250 portable tests and Ruff passed, including bounded restore retention,
copy-failure preservation, per-volume disk guards, controlled disk-pressure
checkpointing and checkpoint-error status. These tests are engineering fixtures,
not trained completions. The three good checkpoints passed their unchanged
hash/replay/CSV/attempt audit. See [the failure audit](assets/performance/iqn-four-hour-result-20261027.json),
[global progress evidence](assets/performance/iqn-four-hour-goal-progress-20261027.json)
and [the preserved best trajectory](assets/performance/iqn-four-hour-result-20261027.png).

```powershell
.venv\Scripts\python.exe scripts/audit_iqn_four_hour_result.py
```

The completion follow-up is retired after reporting the failure. A future
allocation must address storage headroom and explicitly resume from the last
valid checkpoint, while retaining failed time/decisions as separate accounting.

## Subsequent verified storage cleanup

On October 27, the owner authorized a cleanup after the failure review. The
cleanup removed 9647 disposable, previously retained slot-9 staging copies
(397032026141 logical bytes) from project-owned parallel worker profiles. Each
removed file was checked with SHA-256 against an intact state in the immutable
SMG1 or SMG2 curriculum bank; source digests were checked against their bank
manifest. The oldest retained file per worker and at least one copy per
worker and matching state digest were kept. The 22 candidates that did not
match a verified source were also kept. Original ROMs, savestates, profiles,
and checkpoints were not removed. Per-file local audit logs are retained in
`.local/cleanup-verified-duplicates-20261027.jsonl` and
`.local/cleanup-other-verified-duplicates-20261027.jsonl`. The nine curriculum
sources and the last valid IQN checkpoint were hash-verified after cleanup.
No training was resumed as part of this cleanup.
