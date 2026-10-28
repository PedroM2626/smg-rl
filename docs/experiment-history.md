# Experiment history and evidence boundaries

This concise register links the dated reports rather than replacing their
protocols and logs. As of October 28, 2026, no full-start, native RAM-confirmed
star has been demonstrated in the reported training or frozen evaluations.
Figures referring to route prefixes, distance or projections are not native
percentages of the mission. A local run, a Git-tracked summary, and a video or
trajectory have different retention and reproducibility properties.

| Stage | What was established | What remains uncertain |
| --- | --- | --- |
| Dolphin integration and RAM | Frame-stepped visual collection, coordinates, death/star checks and an SMG1 HP-loss detector were validated on the documented game builds; see [native backend](native-backend.md), [memory](memory-profile.md), [SMG1](smg1.md) and [damage](damage-and-recovery.md). | Detector fixtures and near-star states are not learned full-start completions; other game builds require validation. |
| Novelty and world model | Stationary visuals can create intrinsic signals; objective attribution, offline ablations and gated online trials exposed prediction/control limitations; see [exploration audit](exploration-failure-analysis.md), [world-model ablation](world-model-ablation.md) and [staged results](staged-training.md). | These experiments do not establish that intrinsic reward alone caused failure, nor that every model-based method is unsuitable. |
| Controls and reward | Eight-action controls, grounded regional credit and bounded intermediate starts were tested; see [motor controls](motor8-and-algorithms.md) and [regional intervention](regional-intervention.md). | The mapped opening covers only a small part of the missions; changing control timing or reward changes the experiment signature. |
| Rainbow and IQN | The equal-budget exploratory comparison and its frozen evaluations found no stars; see [model-free results](model-free.md#final-iqn-result-and-comparison-closure). | One training seed per game is insufficient for a general algorithm ranking. |
| IQN and IMPALA | Frozen SMG1 evaluation of the 749,006-decision checkpoint obtained zero stars in six attempts; deterministic attempts reached the opening endpoint, then died; see [frozen evaluation](frozen-iqn-evaluation-20261027.md). A later allocated four-hour run failed early on disk exhaustion; see [failure audit](iqn-four-hours-20261027.md). | Seed labels restored the same game state. Collection maxima and the failed block's unsaved weights are not frozen-policy success. |
| Genetic alternatives | The frozen-vision genetic XYZ-distance run completed 134 generations without improving its reevaluated champion or collecting a star; see [three-hour result](genetic-goal-distance-20261027.md#owner-requested-three-hour-run). A full-visual trial also did not improve its unchanged baseline; see [visual trial](genetic-visual.md). | These are different interventions and budgets, not proof that GA is better or worse than IQN or that training vision cannot help. |
| Robust genetic selection | The 49-episode mutation calibration failed its preregistered viability gate; the proposed training arms were not run; see [calibration](robust-genetic-plan-20261027.md#calibration-result-blocked). | One initial genome and one restored game state do not characterize all mutation operators or environments. |
| Expanding route | The bounded IQN/IMPALA run collected 100,884 new decisions and zero stars, stopped after two main blocks without a validated extension, and reached a 66.25% opening maximum; see [result](dynamic-route-night.md#completed-result). | The maximum was exploratory collection, the route was not extended, and the new reward scale and reset optimizer/replay confound comparison with the source. |

## Interpretation and next decision

Engineering throughput, detector validation and bounded recovery improved the
ability to run and audit experiments. None alone establishes reliable learned
control or a completed mission. In particular, compare the *same metric* on
the same policy-evaluation protocol before saying that one method progressed
farther. A straight-line goal projection and the fraction of a winding mapped
opening answer different questions.

The practical next step is to identify the failure of the final frozen policy
at the mapped frontier, measure the new reward scale and its interaction with
death/timeout on saved trajectories, then run an isolated bounded comparison.
Candidate changes include a tested simple reward scale and stable frontier
starts. The original weights, signed replay and canonical-start evaluation
remain controls; assisted starts are exploration aids, not completion claims.
Evaluate independently trained seeds where feasible and vary actual game-state
or timing conditions, recording what changed. This is a research plan, not an
authorization for an unattended or unbounded run.

First evaluate both the source and the final expanding-route checkpoint without
learning, using the same action policy, decision cap and canonical start, plus
clearly recorded action-noise and validated platform-timing variations. Record
videos, death locations, per-attempt route and geometric measurements, and RAM
stars. Repeated deterministic resets alone are reproducibility checks, not
independent game environments. The expanded-branch checkpoint is available
locally under `runs/model-free/dynamic-route-smg1-20261028/` in the final block;
its replay and optimizer are specific to that branch.

Then audit reward per attempt, movement/idle/damage contributions and return
under stationary, forward, reverse, fall and timeout cases. The expanding-route
profile reduced opening payout from 1.0 to about 0.0437 while retaining a -2
death/timeout penalty. Test a small, explicit alternative scale or normalization
without violating the existing objective guard, record any remaining incentive
and do not infer a causal fix from the audit alone. A reward change requires its
own signature and compatible replay; compare fresh branches with the same
weights, replay/optimizer-reset policy, starts, observations and budget so that
only the declared intervention changes.

If that comparison supports further work, review validated intermediate starts
as an exploration aid or use an archive-and-return approach inspired by
[Go-Explore](https://arxiv.org/abs/2004.12919). This is a hypothesis for this
project, not an implementation result. Preserve canonical full-start evaluation
and test actual route extensions separately: the last run never exceeded its
opening, so it cannot establish whether the expansion mechanism helps beyond it.
