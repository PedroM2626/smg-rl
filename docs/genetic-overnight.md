# Continuous genetic continuation until 07:00

> **Historical context:** Experiment status, authorizations, schedules and live counters in this document refer to their stated dates. Later work may supersede them; consult [current evidence and next experiments](current-status.md) and check actual process state before acting.

**Completed:** collection stopped at October27,07:00 BRT, with zero stars and
no improvement over the inherited pilot champion. The owner subsequently
authorized [returning to IQN](iqn-return-20261027.md); this GA remains stopped.

On October26 the owner chose to continue the GA instead of IQN, without
scheduled training pauses, and requested progress analysis at07:00. The existing
absolute cutoff remains **October27,2026 at07:00 America/Sao_Paulo**,
equivalent to **2026-10-27T10:00:00+00:00**. This supersedes the remaining IQN
overnight dispatch permission. It does not allocate indefinite training.

## Method and inheritance

The one-hour pilot improved the frozen head's grounded geometric fitness from
1119.906610 to2361.440219 world units (3.8255% to8.0665% of the calibrated
start-to-goal projection). It reached100% of the short mapped opening, with
zero stars. These are local policy gains under a different objective/horizon,
not proof of general GA superiority or native mission-completion percentage.
The plateau after the seventh generation remains an unresolved limitation.

Continue the same24-individual elitist GA,4104 action-head genes, mutation
standard deviation0.02,4 elites, tournament selection and uniform crossover.
The original IMPALA/IQN features and original481532-real checkpoint remain
frozen. The more recent IQN pause checkpoint is retained but is not substituted
as a new feature extractor, which would invalidate the evolved genomes.
There are no gradient updates, replay fitting, world model or novelty bonus.
Fitness remains the largest alive-grounded X/Z projection toward the calibrated
goal. Death/damage are recorded without extra fitness terms. Every trial starts
from canonical savestate1, uses deterministic visual actions and has at most
1200 decisions. No assisted starts are used.

`resume_population` verifies the completed pilot's source hashes, champion hash,
population dimensions, archived genome hashes, physical-trial fitness and RNG
state. It reconstructs the pending generation at index41 by applying the saved
reproduction step to completed generation40. Its22 completed physical trials
are inherited, identified in metadata and never charged as new attempts or
unseen tests. Only individuals4 and17, whose prior trials were budget-cut, are
evaluated again from the canonical start under the new allocation. Their old
618 partial decisions remain retained in the pilot accounting. Only after all
24 eligible scores are present may generation41 reproduce.

The prior champion file is copied unchanged into the new run; its origin run
and old trajectory remain explicit until a newly evaluated individual improves
it. New real decisions, complete/partial trials and generations are distinct
from inherited counters. There is no reset of the population or mutation RNG.

## Continuous but bounded execution

`scripts/run_genetic_overnight.py supervise` intentionally stops the active IQN
supervisor, waits for its controller to close, and audits its exact owner-stop
checkpoint before launching any new emulator. It verifies exclusive ownership:
at most six SMG1 Dolphins, one GA parent and six Torch-free environment workers.
The new state is `.local/genetic-overnight-state.json`. IQN is retained and
does not automatically resume at the end of this allocation.

One persistent GA process runs through the night. A completed generation has
its physical fitness, finite population and artifacts checked before selection,
without an emulator restart or a scheduled human-review pause. Memory guards,
partial-trial accounting and source immutability remain active. Errors or
pressure retain evidence and stop successors without a blind retry.

The trainer receives `--deadline-utc 2026-10-27T10:00:00+00:00` and
`--confirmation-seconds 0`. It checks the absolute wall deadline independently
of its monotonic clock and the app follow-up. A12-hour outer wall ceiling and
1000-new-generation safety ceiling remain additional finite limits, not targets.
An extended duration is rejected unless a future timezone-qualified absolute
deadline is supplied. At07:00 no new collection is dispatched; an in-flight
action, checkpoint, plot and emulator cleanup may finish shortly afterward.
There is no early confirmation reserve or additional final evaluation budget.

Create `.local/genetic-overnight.STOP` to stop this allocation. Existing jobs,
stop files and duplicate dispatch states are rejected. Inspection is read-only:

```powershell
.venv\Scripts\python.exe scripts/run_genetic_overnight.py snapshot
```

## Morning analysis

The final follow-up runs at07:00. The former three-hour IQN follow-up now checks
only GA safety, stays quiet while healthy and never pauses/resumes or dispatches
training. Morning analysis checks actual PIDs/creation times, freshness,
artifacts/hashes, new versus inherited/cumulative real decisions and attempts,
generation lineage, source immutability, deaths/HP, action occupancy, champion
trajectory and native RAM-confirmed stars from full starts.

The report must distinguish geometric projection from mapped-opening fraction
and native mission completion, inherited bests from new learning, partial trials
from complete fitness trials, and evolution collection maxima from frozen
evaluation results. The new overnight champion has no independently allocated
frozen confirmation episodes; do not silently run them. Synthetic fixtures and
reused audit data are not unseen tests or trained stars. TensorBoard maintains
separate new GA events and the historical IQN/pilot series. Retire both follow-ups
after the final report; preserve all checkpoints, ROMs, savestates and profiles.

## Dispatch verification

The live IQN pause passed its existing clean-stop audit at523004 real decisions
and8000 completed attempts. That newer checkpoint is retained; GA features
continue to come from the original481532-real source. The pending population
was reconstructed as24 by4104 parameters at generation index41, with exactly
22 inherited complete trials and only individuals4/17 awaiting reevaluation.
Actual new supervisor/controller creation times and six owned emulators were
verified; collection counters advanced with zero reported errors. See the
[authorization/ancestry evidence](assets/performance/genetic-overnight-authorization-20261026.json).
233 portable tests and Ruff passed, including absolute deadline enforcement,
extended-budget rejection without a deadline, exact population/mutation-RNG
restoration and exclusion of inherited partial trials from cached fitness.

## Final morning results (October27)

The trainer completed with `stop_reason=scheduled_deadline`, no errors and
unchanged source/checkpoint hashes. Actual runtime was27903.437 seconds
(7h45m03.437s), at29.05 real decisions/second. The latest sampled emulator
response arrived41.8ms after07:00, consistent with an already in-flight action;
the supervisor finished closing and auditing at07:00:06.458 BRT. Actual
supervisor/controller identities are no longer alive and no Dolphin remains.
IQN did not resume. Its newer523004-real/8000-attempt checkpoint is retained.

| Accounting | Overnight only | Pilot + overnight |
| --- | ---: | ---: |
| Evolution decisions, including partials |810628 |900164 |
| Separate frozen-confirmation decisions |0 |1642 |
| All real emulator decisions |810628 |901806 |
| Completed generations |278 |319 |
| Completed evolution trials |6668 |7674 |
| Separate completed frozen confirmations |0 |4 |
| Partial trials |6 |8 |
| Partial decisions (already included above) |254 |872 |
| Native full-start stars |0 |0 |

The22 inherited fitness records were not counted again. Overnight completed
generations have indices41 through318. Generation319 had18 completed trials
and6 deadline-cut trials, so it did not reproduce. Every new trial used a
canonical start; no assisted state or synthetic completion is included.

**No overnight record improved the pilot:** the retained best remains
2361.440219 world units of grounded projection, equivalent to8.066498% of the
calibrated start-to-star straight-line projection and100% of the short mapped
opening. The best new complete trial and even the best unselected deadline-cut
trial only matched that fitness. The exact champion file hash remained
`23eb316bff29daec4a02e182e5122480cbfa0952387d7a44ae06ff814b546018`.
Its originating episode is still the pilot's generation6/individual21, which
eventually died. Neither percentage is native mission progress or completion.

Overnight recorded6358 deaths,310 episode timeouts and848 nonfatal HP lost.
Jump accounted for58.2194% of decisions, forward22.3903%, crouch12.5278% and
right5.8694%; other actions together occupied less than1%. This action occupancy
describes actual collection, not a newly frozen evaluation. No additional
frozen confirmation episodes were allocated or run during morning analysis.

This result demonstrates a sustained plateau in the fixed-feature,4104-gene
action-head search. Additional time under the same settings did not solve it;
it does not establish a defect in Sweet Sweet/SMG1 or general superiority or
inferiority of genetic algorithms. Candidate limitations to investigate before
another allocation include the frozen representation/action head and whether
a straight-line maximum sufficiently ranks safe onward routes. These remain
hypotheses; no fitness, representation or control change was made overnight.

The final artifact audit recomputed grounded fitness from all6674 new trial
CSVs (including partials), reconciled real/attempt/star counts, compared frozen
source tensors and hashes, and reproduced all278 new generation populations
and mutation/selection RNG transitions, including the22 inherited records.
It separately audited the final unselected generation. All checks passed.
The following audit does not collect training, but it writes its result artifacts
and figure. Preserve existing evidence before rerunning it:

```powershell
.venv\Scripts\python.exe scripts/audit_genetic_overnight.py
```

Detailed evidence is in [the final audit](assets/performance/genetic-overnight-result-20261027.json).
The [trajectory and stagnation figure](assets/performance/genetic-overnight-result-20261027.png)
uses the inherited champion and the complete generation histories. This reused
data analysis is not unseen testing. The morning and safety follow-ups are
retired after reporting; further training needs another owner allocation.
Final verification passed234 portable tests and Ruff. Shared trajectory audit
helpers are importable from both direct command-line reporters and tests; the
accounting tests explicitly reject treating inherited fitness as new trials.
