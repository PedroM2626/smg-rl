# Documentation index

[Current evidence, settings and next-experiment plan](current-status.md) are
summarized separately from dated historical records. The
[project experiment history](experiment-history.md) links successes, failures,
and limits across stages. The [expanding-route trial](dynamic-route-night.md#completed-result)
stopped with 100,884 real decisions, no accepted extension and no stars.
The
[frozen IQN checkpoint evaluation and goal-distance candidate](frozen-iqn-evaluation-20261027.md)
report six canonical attempts, videos, trajectories and the offline-only metric.
The
[four-hour IQN/IMPALA allocation](iqn-four-hours-20261027.md) failed after
about2h18 due to disk exhaustion; it did not complete four hours. No training
was active at the October 27 audit. Its last valid checkpoint and failed-block
evidence are retained.
The [full-visual genetic trial](genetic-visual.md) completed with zero stars and
no improvement over the unchanged baseline. Its artifacts and the intermediate
[IQN return](iqn-return-20261027.md) remain retained.
The [overnight SMG1 genetic allocation](genetic-overnight.md) completed at
October27,07:00 BRT: zero stars and no improvement over the inherited pilot
champion. Its population/RNG ancestry from the
[completed genetic pilot](genetic-hour.md) and all earlier IQN/world-model
checkpoints remain retained; no automatic successor is allocated.

Documentation describes the checked implementation, measured engineering
validation, and an experimental protocol. A method being specified here does
not mean its ablations or full-course learning results have been obtained.
Use actual process identity together with run artifacts for live status; an
error can leave `status.json` stale. Historical observations below are dated
evidence, not dashboards. World-model snapshots and final stop are in the
[historical experiment record](world-model-live.md).

## Start and operate

| Need | Document |
| --- | --- |
| Experiment timeline, negative results and comparison boundaries | [Experiment history](experiment-history.md) |
| Frozen checkpoints, state drift, reward audit and matched scale A/B | [Reward-scale study](reward-scale-study-20261028.md) |
| Bounded dynamic-route trial and actual outcome | [Dynamic route trial](dynamic-route-night.md) |
| Installation, supported tasks, rewards, and recommended quick start | [Main README](../README.md) |
| Robust genetic calibration, blocked gate and planned paired comparison | [Robust genetic plan and results](robust-genetic-plan-20261027.md) |
| Model-free Rainbow/IQN implementation, PER, TensorBoard and completed trials | [Model-free learning](model-free.md) |
| Exact command arguments, backend, and side effects | [CLI reference](cli.md) |
| Local ROM and savestate filenames and roles | [Asset guide](../data/README.md) |
| Native build, shared-memory capture, exact frame stepping, profiling, and benchmarks | [Native backend](native-backend.md) |
| Protocol v2, curiosity batches, bounded exploration, and the stagnation audit | [Optimization revision](optimization-v2.md) |
| Historical protocol v3, event wakeups, native preprocessing, and worker barrier profiling | [Native pipeline and collection](optimization-v3.md) |
| Vulkan comparison, dependency audit, transactional checkpoint pairs and bounded recovery | [Renderer and security revision](optimization-v4.md), [security policy](../SECURITY.md) |
| Hidden rendering, isolated inputs, CUDA, timing, and throughput | [Background training](background-training.md) |
| Two learners, resource measurements, status, and cross-game reports | [Concurrent training](concurrent-training.md) |
| Health penalties, checkpoint recovery, and measured damage evidence | [Damage and recovery](damage-and-recovery.md) |
| Practical reward tuning, evaluation, and troubleshooting | [Training guide](training.md) |
| Visual replay, recurrent world model, prediction gates and imagined actor/critic experiment | [World-model implementation](world-model.md) |
| Authorized online world-model experiment, dated snapshots, final stop, gates and recovery procedure | [Live world-model experiment](world-model-live.md) |
| Galaxy 1 terminal sampling, visual-loss balance and prior prediction repair | [Galaxy 1 imagination repair](world-model-smg1-repair.md) |
| Isolated Galaxy 2 objective transfer tests, paired seeds and offline audit boundaries | [Galaxy 2 world-model ablation](world-model-ablation.md) |
| Reviewed continuation budgets, paired online results, curiosity trial and final stop | [Staged training results](staged-training.md) |
| Full-start policy videos, stability-screened savestates and assisted-reset accounting | [Frontier exploration](frontier-exploration.md) |
| Paired horizon-8/16 audits, frozen-world actor fitting and real-game evaluation | [Horizon trial](horizon-trial.md) |
| Intrinsic reward hypothesis, actor-gradient attribution, stationary visual bonus and control/reward fixtures | [Exploration failure analysis](exploration-failure-analysis.md) |
| Reviewed curiosity/entropy factorial, sixteen motor actions, grounded route prefix and compatible replay pilots | [Exploration interventions](exploration-interventions.md) |
| Current eight exclusive actions, actual-input videos, game-specific transfer boundaries and Rainbow assessment | [Motor8 and algorithms](motor8-and-algorithms.md) |
| Hourly reviews, actual reward-label audit, stationary waypoint credit and isolated route correction | [Motor8 reward review](motor8-reward-review.md) |
| Completed trained motor8 videos, six failures, trajectories and finite-pilot closure | [Motor8 evaluation results](motor8-evaluation-results.md) |
| Repaired SMG1 terminal prediction, preserved dynamics, fresh confirmation and gated actor smoke | [SMG1 boundary repair](smg1-boundary-repair.md) |
| BTR Appendix J.1, inspected Wii code, RAM-probe limits and staged environment hypotheses | [BTR research review](btr-research-review.md) |
| Corrected regional potential, nine validated intermediate starts,36 frozen control attempts and frame4 equivalence | [Regional intervention](regional-intervention.md) |
| Completed six-collector blocks, exact 8192-per-mission accounting, prediction gates and full-start/assisted results | [Six-instance regional training](regional-six-training.md) |
| Historical throughput and budget measurements | [Native backend](native-backend.md) |

## Understand and reproduce

The completed six-hour protocol is [simple regional reward and IQN/IMPALA](simple-reward-impala.md),
including the duration-neutral objective, residual encoder, gradual exploration,
attempt axes and finite block supervisor. Earlier experiment closure statements
refer to their own dated allocations.
The [memory-pressure recovery](memory-pressure-recovery.md) records the actual
Windows failure, lean collector fix and explicitly authorized remaining-budget
resumption without resetting real decisions or attempts.
That allocation and all nine frozen evaluations are complete; see the
[final IQN/IMPALA results and Sweet Sweet diagnosis](impala-final-results.md).

| Need | Document |
| --- | --- |
| POMDP, action/observation spaces, reward equations, novelty, PPO, and limitations | [Methodology](methodology.md) |
| Seeds, baselines, ablations, metrics, uncertainty, and evidence register | [Experimental protocol](experiments.md) |
| Module architecture, environment, configuration, checkpoint semantics, and outputs | [Reproducibility](reproducibility.md) |
| Galaxy 2 pointer chains, executable checks, terminal states, and derivation | [Galaxy 2 memory profile](memory-profile.md) |
| Galaxy 1 pointer chains, health, detector checks, and integration | [Galaxy 1 report](smg1.md) |
| Historical portable tests, live detector tests, smoke learning, and failures | [Validation register](validation.md) |
| Primary research and upstream provenance | [BibTeX bibliography](references.bib) |
| Original project license and material outside its scope | [MIT License](../LICENSE), [third-party notices](../THIRD_PARTY_NOTICES.md) |

## Evidence boundaries

- Portable tests verify code behavior without booting Dolphin or a ROM.
- Near-star positive checks and deliberate falls verify live detectors.
- The measured Galaxy 1 3-to-2 HP trace verifies health reading; reward replay
  verifies the corresponding penalty, rather than learned damage avoidance.
- Smoke training verifies collection, updates, and checkpoint persistence.
- Recovery observations verify continuation after an infrastructure failure.
- Full-course success requires a full-start trajectory and the game-specific
  completion event. Geometric progress and novelty returns alone do not prove it.
- Multi-seed online policy comparisons remain a protocol. Completed paired
  offline objective ablations and single-seed bounded online comparisons are
  reported in the ablation and staged-results documents linked above.

When changing behavior, update the relevant specification, command reference,
configuration example, and validation record together. Keep live counters in
reports and each run's `status.json`; any counters summarized in the README must
be explicitly dated snapshots. Preserve negative evidence and timing revisions
when reporting performance.
