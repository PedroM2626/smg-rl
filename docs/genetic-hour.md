# One-hour SMG1 genetic intervention

**Latest allocation:** the completed pilot is now the ancestor of an explicitly
authorized [continuous overnight GA continuation](genetic-overnight.md), replacing
the remaining IQN run until October27,07:00 BRT. The results below remain fixed.

On October 26 the owner explicitly requested implementation and a one-hour
genetic algorithm (GA) pilot, with fitness defined by furthest advance. This
supersedes earlier statements that no genetic experiment was allocated. Only
Sweet Sweet Galaxy is selected, using at most six owned Dolphin instances.

## Method and scope

The algorithm is an elitist genetic search, not gradient descent. It evolves
the deterministic action head of the paused IQN policy: 8 actions times 512
features, plus 8 biases, for 4104 genes. The previously learned IMPALA encoder,
feature projection, quantile embedding and first advantage layer are frozen.
Midpoint quantiles are averaged before the final linear action head; the shared
value and advantage-centering terms cannot affect the action argmax. The source
genome therefore reproduces the source policy's greedy action selection.

This is **transfer-initialized genetic head evolution**, not evolution of the
entire CNN from scratch, and not evidence that a GA is generally superior to
IQN. No optimizer is stepped and no replay fitting, curiosity, world model,
synthetic transition or genetic modification of the original checkpoint occurs.

The population has 24 individuals. Individual 0 in generation 0 is the unchanged
source head; the others receive independent Gaussian mutations. Every fully
evaluated generation retains its top 4 unchanged, then uses size-3 tournament
selection, uniform crossover and Gaussian mutation (standard deviation 0.02,
parameters clipped to [-5,5]) to produce the other 20. Stable tie ordering avoids
inventing tie-break rewards. Candidate fitness is never cached between
generations. Only complete, finite populations can reproduce. Population
arrays, fitness, candidate genome hashes and reproduction RNG state are saved.

## Simple fitness, boundaries and interpretation

For calibrated canonical start s and near-star goal g in the horizontal X/Z
plane, define u=(g-s)/||g-s||. Fitness is the maximum observed
clip((position-s) dot u,0,||g-s||), in game world units, sampled while the native
grounded signal is true and Mario is alive. Death/fall samples cannot add
fitness, and revisiting a position cannot increase the high-water mark.
Nonfinite positions or implausible inter-sample jumps invalidate the trial and
stop successors. There are no additional damage, death, time or novelty fitness
terms; the environment's original reward is recorded but ignored by selection.

This scalar is a straight-line geometric projection toward the star, not a
native mission-progress variable or a validated full-course path distance.
Curves and lateral movement can distort it.100% projection does not prove
completion; native completion RAM is recorded separately. It also differs from
the old short mapped-opening percentage, which is reported as a separate field.
The existing IQN objective and route specification are unchanged.

Every fitness trial uses canonical savestate 1 and seed 11, with no assisted
starts, action noise or epsilon sampling, and at most 1200 decisions. All
individuals have the same start and horizon. A GA-only config copy sets the idle
frame budget equal to the episode budget, preventing the short opening's idle
rule from terminating a candidate while it is exploring beyond that opening.
This GA horizon differs from some historical IQN evaluations, so those results
are not a matched algorithm comparison. The GA's global geometric fitness also
differs from IQN's mapped-opening learning objective. Only the frozen source
and evolved head are evaluated under the same new GA conditions; any observed
improvement is local to this transferred-policy intervention, not proof of
general GA-versus-gradient sample efficiency.

Six persistent, Torch-free collector subprocesses evaluate independent
individuals while one parent batches visual inference. Completed workers receive
the next individual; idle workers are not stepped. The run has a 3600-second
wall budget including model/environment setup. The last 300 seconds are
reserved for frozen source-baseline and champion trials on confirmation seeds 23
and 47. Finishing those trials early can end the run before the ceiling.
Partial deadline-cut episodes are retained but cannot establish candidate
fitness or become selection wins. The attempt ledger distinguishes evolution,
frozen confirmation and partial trials. Every completed generation is audited
before reproduction; at most 1000 generations is an additional safety ceiling.

## Coordination and retained evidence

`scripts/run_genetic_hour.py supervise` first records an intentional IQN pause,
waits for its supervisor/controller to stop and audits the clean owner-stop
checkpoint using all existing hash/replay/CSV/reward checks. It verifies that no
Dolphin remains before GA dispatch. It never starts another population while
IQN owns the six emulators. The state is `.local/genetic-hour-state.json`.

After clean GA completion, source immutability, champion hash and trial/decision
accounting are checked before the original overnight IQN allocation resumes
from the exact paused checkpoint, including optimizer/replay/RNG and counters.
The original October27 07:00 BRT deadline remains unchanged. GA decisions and
fitness attempts never enter the IQN replay or step/attempt counters. The
paused overnight ledger is archived, and pending final IQN evaluations remain
deferred until overnight completion. The telemetry exporter is resumed too.

Create `.local/genetic-hour.STOP` to stop this intervention and prevent automatic
IQN resumption. Failure or resource pressure also blocks automatic successors;
no blind retries are allocated. Original ROMs, states, profiles and checkpoints
are preserved and excluded from Git. All run outputs remain under
`runs/model-free/genetic-smg1`, including `trials.jsonl`, generation archives,
`champion.pt`, `summary.json`, every CSV and `best-trajectory.png`.

TensorBoard adds a separate `genetic` series for fitness, geometric projection,
mapped opening, sampled X/Z path distance, environment return, trial decisions,
real decisions and generation best/mean fitness. The final report must compare
the unchanged source and champion on confirmation seeds, state all failures, give
actual runtime/generations/attempts/real decisions, and distinguish RAM stars
from geometric maxima. The completion follow-up is retired after reporting.

```powershell
.venv\Scripts\python.exe scripts/run_genetic_hour.py snapshot
```

Confirmation seed labels23/47 are not unseen states: both policies use the same
canonical savestate and deterministic actions. Changing a Gymnasium reset seed
does not randomize the Wii RNG restored from that savestate. These repeats check
repeatability; they do not establish independent generalization.

## Completed pilot results (October 26, 22:45 review)

The pilot completed cleanly in **3360.062 seconds (56 minutes)** inside its
one-hour ceiling. The reserved confirmation block finished before the ceiling;
the allocation was not extended to fill the remaining four minutes. There were
41 complete generations, 1006 completed evolution trials, 2 partial budget-cut
trials (618 decisions, excluded from selection), and 4 completed frozen
confirmation trials: 1010 completed attempts plus the 2 partial attempts.
Evolution consumed89536 real decisions including the retained partials;
confirmations consumed1642, for **91178 new real decisions**. There were zero
gradient updates,981 deaths,52 nonfatal HP lost and **zero native full-start
stars** across all phases.

Both canonical deterministic confirmation repeats reproduced these outcomes:

| Frozen policy | Grounded fitness (world units) | Geometric projection toward star | Mapped initial opening | Outcome |
| --- | ---: | ---: | ---: | --- |
| Original IQN action head |1119.906610 |3.825515% |79.415796% |Death,552 decisions |
| Evolved genetic action head |2361.440219 |8.066498% |100% |Death,269 decisions |

The evolved head increased this fitness by **2.1086 times (+110.86%)** under the
same new evaluation conditions. It reached the end of the mapped opening and
made further geometric advance, but did not finish Sweet Sweet. The best result
first appeared in generation index6 (the seventh generation), with no further
record by the end of generation41's partial evaluation. This plateau limits the
claim that simply continuing this search would keep improving it.

In its best recorded trajectory, the peak grounded sample was at decision63,
X515.5874,Y-50.0002,Z-4575.6050,HP3. It lost one HP at decision77, retreated and
eventually died. Fitness measures the maximum reached, not the final position
or total distance traveled.100% opening and8.0665% straight-line projection are
different measurements; neither is a native mission-completion percentage.

An independent artifact audit recomputed fitness from every one of the1012
trial CSVs, reconciled all decisions/attempts, checked native completion flags,
verified all41 generation archives and exact RNG-based reproduction lineage,
and compared the frozen source tensors and checkpoint hashes. The champion
genome is `e35aa9a250eef3bdd9996819b9d9e388105ccdca15ae9d4ad82208e4e41594fb`.
Evidence and comparative trajectories are retained in
[the final audit](assets/performance/genetic-hour-result-20261026.json) and
[the comparison figure](assets/performance/genetic-hour-comparison-20261026.png).
The audit is reproducible with:

```powershell
.venv\Scripts\python.exe scripts/audit_genetic_result.py
```

That point-in-time audit also verifies live IQN restoration. After the subsequent
owner-authorized switch back to GA, its live-IQN assertion is intentionally no
longer satisfied; retain the completed audit JSON as historical evidence and do
not restart IQN merely to satisfy a superseded runtime-health check.

IQN resumed its original paused checkpoint at481532 real decisions and119871
updates, preserving its optimizer/replay/RNG and its October27 07:00 BRT
deadline. GA decisions did not enter its replay or counters, and the evolved
head was not silently promoted into IQN. Actual resumed supervisor/controller
identities and fresh advancing status were verified. The telemetry exporter is
active; the TensorBoard HTTP server was found stopped during review and was
restarted on127.0.0.1:6006, with HTTP200 confirmed. No extra training, genetic
generation or retry was allocated during this final review.
Final review validation passed230 portable tests and Ruff, including independent
trajectory-fitness recomputation and rejection of inflated fitness/completion
claims. Synthetic fixtures are not counted as emulator outcomes.
