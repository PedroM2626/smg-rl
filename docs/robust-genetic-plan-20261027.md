# Robust genetic selection: preregistered staged comparison

The owner authorized a bounded calibration followed, if its preregistered gates
pass, by an equal-wall-clock comparison of stochastic and deterministic genetic
selection. This plan uses one canonical SMG1 game state and action randomness;
seed labels alone do not vary the Wii environment. It does not claim that the
earlier 27-episode probe establishes statistical superiority. Stars must be
confirmed from full-start native RAM, not from geometric proximity.

## Stage 0: mutation calibration

From the verified frozen-feature champion, evaluate a baseline and 16 children
for each of sigma 0.02, 0.05 and 0.10. Mutations share underlying normal draws
across scales. Six workers use canonical start, deterministic argmax and a
400-decision horizon. Maximum wall time is 30 minutes; finish earlier if all
49 episodes complete. Partial episodes cannot qualify. A child is viable if it
collects a native star, or survives more than 60 decisions and obtains positive
fitness; both the 60-decision cutoff and this viability definition are
heuristics, not claims about mission success. Behavior differs when the first
up to 60 applied actions differs from the baseline; a shorter episode hashes
its shorter sequence. Record full traces and interpret this alongside
positional progress, not as proven useful diversity.

Select the **largest** sigma with at least 8/16 viable children and at least
4/16 distinct action prefixes. If none qualifies, stop and report instead of
assuming that this proves a local optimum. Record stars, outcomes, steps,
position-derived fitness, action prefixes, source hashes and incomplete trials.

## Calibration result: blocked

The 49 episodes completed in 239.42 seconds, with 8,495 real decisions,
31 deaths, one nonfatal HP lost, no partial episodes and zero RAM stars. The
baseline exactly reproduced the prior 2,075.671-unit, 269-decision death.
The [independent audit](assets/performance/robust-genetic-calibration-reviewed-20261027.json)
recomputed all 49 trajectory fitnesses and action hashes and verified the
source checkpoint and champion hashes. This calibration process was started
before the later per-process network digest instrumentation was added; its
audit does not claim a post-run in-memory tensor digest.

| Sigma | Viable, out of 16 | Prefix differs from baseline | Mean fitness | Median fitness |
| --- | ---: | ---: | ---: | ---: |
| 0.02 | 4 | 16 | 871.994 | 856.324 |
| 0.05 | 2 | 16 | 428.285 | 300.773 |
| 0.10 | 2 | 16 | 330.402 | 128.201 |

**No sigma passed the preregistered viability gate**, so the two three-hour arms,
sampling smoke and held-out confirmation were not launched. The six Dolphins
were closed; there is no automatic successor. With sigma 0.10, nine children
survived over 60 decisions, but seven of those nine made zero proximity progress;
the viability rule required positive progress as well as survival, unless a
star was collected. Action-prefix differences alone do not establish useful
behavioral diversity. These are 16 children per scale from one source and one
canonical state; the result does not prove that the entire search space is a
local optimum or that other mutation methods cannot work.

The epsilon sampling and grouped-selection infrastructure has portable tests,
but the real-game sampling smoke, arm execution and held-out confirmation have
not run because the gate failed. Comparison code requires a matching, completed
calibration review and refuses this blocked result. This is not an end-to-end
validation of the proposed training arms.

The status and complete local trajectories are retained in
`runs/model-free/genetic-robust-smg1/calibration-20261027-223807-924862/`.
The [original review](assets/performance/robust-genetic-calibration-20261027.json)
is retained alongside the stronger audit, which also records summary and ledger
hashes. No previous fitness was reused.

## Planned selection comparison

After complete calibration, review its counts and protected-source identity
before dispatch. Treatment B and control C use separate fresh populations,
48 individuals, eight elites, tournament size three, the chosen mutation scale,
six workers, the same fixed source/start and a 1,200-decision horizon. Each arm
has at most three hours of collection; they run sequentially and never overlap.
No fitness or partial trials are reused from another mode. B selects on the
mean of three epsilon-0.05 action draws and records deterministic argmax as a
separate diagnostic. C selects on one argmax episode. Elites are reevaluated
each generation. Per-generation action-randomness seeds are shared across
candidates and change between generations; reset state itself is unchanged.

This paired seed arrangement reduces some noise but is not equivalent to
independent game states. Equal wall time, number of generations, distinct
candidates and real emulator decisions must all be reported: the two arms do
not have equal evaluations per generation. Two 3-hour arms plus calibration
and confirmation require over 6.5 hours in total. Prior 24-individual/argmax
results are historical context, not a matched fresh control. Softmax is not
selected because the earlier T=0.5 probe was nearly random and logits are not
calibrated between genomes.

Confirm the top eight of each arm with at least five held-out epsilon-0.05
action seeds and separate argmax evaluation. Selection and confirmation seeds
must be disjoint; reusing confirmation data to select a final winner requires
another independent final test. Compare candidate behavior, success counts,
distance, deaths, duration, action sampling, per-generation L2/pairwise and
behavioral diversity, and time/decision budgets. A three-draw training mean or
single population seed is exploratory, not a significance claim. Repeat the
winning arm with a separate population seed before any general claim, only
after its reviewed bounded stage passes. If a required integrity check or
calibration gate fails, retain partial evidence and stop rather than retrying
blindly or relaxing the gate.
