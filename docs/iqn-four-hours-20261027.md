# Four-hour six-worker IQN allocation

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
