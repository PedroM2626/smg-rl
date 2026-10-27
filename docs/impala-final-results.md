# Six-hour IQN/IMPALA results and Sweet Sweet diagnosis

> **Historical context:** Experiment status, authorizations, schedules and live counters in this document refer to their stated dates. Later work may supersede them; consult [current evidence and next experiments](current-status.md) and check actual process state before acting.

Date: October26,2026. Both finite training branches and all nine declared frozen
canonical episodes are complete. That protocol allocates no further training or
new algorithm trial. Original artifacts remain retained. The owner later
authorized a separate [two-hour SMG1-only continuation](smg1-six-two-hour.md),
which does not change these results or reopen their budgets.

The owner subsequently requested a separate
[frozen stochastic execution comparison](stochastic-iqn-evaluation.md), now also
complete. It preserves the greedy results below, adds no training, and locates
the historical Sweet51.80% exploratory record on a top-down coordinate map.

## Final training and frozen behavior

SMG1 finished296150 committed real decisions and1911 completed attempts;
SMG2 finished462968 aggregate decisions and5486 attempts. Both consumed the
original six-hour accounted allocation, including retained failed partial work.
There were zero RAM-confirmed full-start or assisted training stars.

All three canonical evaluations per mission used epsilon0, disabled parameter
noise, seeds11/23/47 and the existing600-decision maximum. These seeds do not
randomize the identical restored game state; the trajectories repeated exactly.
They are reproducibility checks, not three independent training replicas or a
test of varied platform phases/generalization.

| Mission | Frozen mapped-opening advance | Outcome in all3 episodes | Decisions/episode |
| --- | ---: | --- | ---: |
| Sweet Sweet |23.6402%|Idle timeout, HP3, no star|198|
| Spinning |100%|Death after the mapped opening, no star|66|
| Double Time |100%|Death after the mapped opening, no star|44|

The100% figures concern short verified openings only. No native whole-mission
percentage was measured. Training collection maxima and rolling averages must
not be presented as these frozen-policy results. SMG2 learned reproducible local
opening traversal; none of the three policies solved its full mission.

The [final audit](assets/performance/impala-frozen-final-20261026.json) verifies
all924 evaluation decisions separately from training, encoded video-frame counts,
exact-image digests, finite raw-Q vectors and applied-control overlays. Agent
files (including saved optimizer state) and replay hashes remained unchanged.
There were zero evaluation stars; all failures remain retained.

## Direct evidence for the Sweet Sweet failure

Each198-decision canonical episode selected jump170 times. Its last80 decisions
were all jump; horizontal position remained exactly X166.7315/Z-315.1401 while Y
oscillated from45 to310.2. Scene-frame counters advanced, and11 of those final80
samples were grounded. Thus this was an active jumping-in-place behavior, not a
paused emulator, a frozen RAM read or a stationary video frame.

The last newly rewarded route progress occurred at decision18. The subsequent
180-decision no-progress budget ended the episode. Video inspection shows Mario
remaining at the starting platform edge instead of proceeding onto the course.
No HP was lost and no death occurred in these evaluations. This does not isolate
all learned effects of risk penalties, but nonfatal damage was not the immediate
cause of the observed termination.

At a late sampled state, all expected action returns were approximately-1.98;
jump won by a small value difference. This is consistent with weak discrimination
among mostly failing actions, not a positive intrinsic reward for jumping.
Curiosity is absent and jumping without new grounded route advance pays zero.
The new objective removes delayed-failure preference but does not force motion
when the learner has not discovered a better continuation.

## Limitations supported by the evidence

1. **Reward coverage is incomplete.** Both maps end at a short opening. There is
   no further incremental route reward beyond that boundary until the remote
   star event. No training star was observed, so replay contains no positive
   terminal-star experience. This is a limitation of the current environment
   design, not evidence that changing the optimizer alone solves the missions.
2. **Exploration has not produced a reliable motor sequence.** Random collection
   reached51.80% of the Sweet opening, but the frozen policy reached23.64% and
   repeated jump. More wall time cannot be assumed to bridge that difference.
3. **The existing curriculum is not full-course coverage.** Its three Sweet
   anchors are right-approach, first-pink and first-chocolate, all within the same
   short opening. Assisted starts already exist; adding more of the same is not
   a new proposal or evidence of canonical success.
4. **Control/vision remain hypotheses, not proven faults.** Eight exclusive
   actions require directional movement and jump to be coordinated over time.
   Twelve-frame decisions and four84x84 images provide limited timing/history.
   The existing scripted directional sequence reached later safe anchors,
   showing that basic SMG1 movement/emulation is possible. It does not prove
   that every control, contact region or camera condition is correct.
5. **Game and mission are confounded.** One SMG1 mission versus two variations
   of one SMG2 mission does not isolate a general SMG1-versus-SMG2 effect. The
   easier-to-learn SMG2 opening is not a controlled game-engine comparison.

## Recommended next tests, not dispatched

First validate a continuous ordered regional route and safe intermediate starts
farther through each actual course. Keep the reward simple: new grounded advance,
verified star, failure and nonfatal damage. Fix coverage rather than stacking
unrelated bonuses or treating screen change as progress.

For Sweet, isolate the initial-platform exit as a finite motor/learning subtask.
Compare the existing control with a separately signed direction-carry jump profile
if needed; do not silently change the owner's eight physical actions or replay
semantics. Known scripted reachability is a fixture. If short demonstrations are
used to teach that transition, label them separately and evaluate the learned
policy from the canonical state without the script or random actions.

Remembering reachable frontier states and deliberately exploring beyond them is
motivated by [Go-Explore](https://arxiv.org/abs/2004.12919). Small demonstrations
also have [primary research support](https://arxiv.org/abs/1709.10089) for sparse
exploration, although that paper's algorithm/tasks are different. These sources
motivate Mario experiments; they do not establish a predicted speedup or justify
counting assisted/scripted trajectories as learned full-start completions.

Only afterward consider an isolated observation-resolution/history ablation and
replay retention of useful examples. Earlier frame4 tests performed worse, so
smaller frameskip should not be assumed better. No new CNN, genetic population,
world model, automatic long extension or silent penalty change was launched.
