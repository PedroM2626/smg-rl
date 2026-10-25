# Staged training continuation

The owner authorized development, fitting and online training on October 24,
2026, with explicit reviews between bounded blocks. The former shutdown hold
is superseded; retained final checkpoints remain immutable starting points.

## First stage

- Galaxy 2: original checkpoint, 1024 model updates per arm, seeds 11, 23, 47;
  compare `continued_baseline` with `terminal_reward_weight` (coefficient 5).
  Use `ablate-world-model --arms continued_baseline terminal_reward_weight`.
  No new Galaxy 2 collection during this frozen-replay comparison.
- Galaxy 1: resume the corrected final checkpoint for 2000 real decisions.
  This is a health check, not evidence of convergence. Save a separate final
  checkpoint and close owned emulator processes at the budget boundary.

The previous audit data are reused for this follow-up and are no longer an
independent unseen test set. Report this selection limitation. Final conclusions
about navigation require fresh real-game episodes. Never select an objective
using pixel reconstruction alone, or weaken quality gates to permit imagination.

## Reviews and subsequent budgets

Inspect progress/status timestamps, controller identity, errors, measured real
decisions per second, replay validity, imagination gate reasons, model and actor
update deltas, deaths, health loss, RAM-confirmed completion and trajectories.
Compare episode progress distributions as well as isolated high-water records.
Geometric progress is not a native completion percentage; falling movement can
mislead. Detector fixtures, near-star states and imagination are never successes.

After a healthy initial block, Galaxy 1 may receive 8000 additional decisions
to reach a 10000-decision first block. Review again before allowing a further
20000 decisions (30000 total). Stop on faults or unresolved measurement issues.

After the Galaxy 2 offline comparison, review all seeds and both scenarios.
Choose an objective only if the trade-offs support a bounded real-control trial.
Use identical starting checkpoints, random seeds, worker allocation and real
budgets when comparing control and candidate; preserve separate replay roots so
one arm does not learn from the other's new trajectories. Any settings migration
must produce a new checkpoint manifest. Begin with a health check, then a
25000-decision first block and at most 100000 decisions per arm after review.
Report each phase separately even though the shared learner budget is aggregate.

These are experiment budgets, not promises of learning. Review checkpoints are
not repeated significance tests. Three seeds describe variability, not proof.
Online A/B fitting is separate from the three offline seed replications.

## Monitoring and stopping

Local `.local/staged-training-state.json` records launched jobs and reviewed
stages. A thread follow-up checks every 15 minutes while this experiment is
active, remaining quiet unless a stage finishes, a failure occurs, or a decision
is required. Each subprocess has its own finite budget: a missed follow-up
cannot silently start another block. No automatic unreviewed stage chaining.
Use the run's `STOP` marker to request an orderly stop and checkpoint save.
Never stop unrelated jobs. Pause the follow-up once this staged experiment ends
or if the owner asks to stop. Log stage decisions and publish results in English;
communicate with the owner in Portuguese.

## Initial Galaxy 1 review (October 24 local time)

Run `runs/smg1/world-model-20261024-225907-822079` completed its exact
2000-decision budget at 280773 lifetime real decisions in 344.766 seconds
(5.80 decisions/s). It added 248 model updates, 15 actor updates and 15360
imagined transitions. The owned emulator closed, stderr was empty and replay
reported no rejected archives. Five episodes timed out; the sixth was interrupted
at the budget boundary. All 2000 logged actions had health 3, with no damage,
death or RAM-confirmed completion. Maximum geometric progress was 2.68%, below
the historical record; this is not evidence of improved navigation.

The last stochastic four-batch gate failed terminal reward MAE (0.570 > 0.5)
and terminal Brier (0.226 > 0.2). Before proceeding, a read-only paired audit
compared the retained source and new checkpoint on identical held-out draws and
Torch seeds 11, 23 and 47, with 16 batches per seed. Both checkpoints passed
two of three audits, failing the same seed on terminal reward MAE. Source
terminal MAE was 0.324/0.515/0.459; current MAE was 0.353/0.519/0.492.
Current terminal Brier was 0.105/0.180/0.150, all below 0.2. The weak terminal
prediction is pre-existing and sampling-sensitive; the small MAE regression in
all three paired draws remains a risk. These are validation seeds, not three
independently trained policies, and no gate threshold was changed.

Decision: allow a bounded 8000 additional decisions, reaching 10000 new
decisions in total, with another review before any extension. Actor updates
continue only on cycles passing every existing quality gate. The diagnostic
is recorded in [the paired audit](assets/performance/smg1-staged-initial-audit-20261024.json).
Galaxy 2 offline fitting remains in progress and no online arm has started.

## Galaxy 2 longer offline comparison review

The 1024-update, three-seed comparison completed all six fits and final audits in
`runs/ablation-20261025-015911-220971`. Relative to the equal-update control,
terminal reward weight 5 reduced terminal reward MAE by a paired mean of 52.34%
for Spinning and 49.81% for Double Time. All three candidate seeds passed every
original gate, compared with zero of three controls. Future-image error changed
by -0.62% and +0.31%, respectively, which does not establish an improvement in
visual prediction or navigation. Continuation Brier increased by 14.41% in
Spinning but remained small in absolute terms (0.00704). Full metrics and
per-seed results are in [the follow-up record](assets/performance/smg2-world-model-followup-20261024.json).

Decision: advance to paired real-control health checks. Use offline seed 11
(the first predeclared seed, rather than the best measured seed) for both arms.
Both checkpoints received exactly 1024 offline model updates from the same
source and retain identical actor weights and real-decision counters. Use
online seed 11 and four workers per arm, with sequential 2000-real-decision
health checks before any longer budget. Each arm receives a separate copy of
the same initial replay, verified with per-file SHA-256 and sizes. Each arm's
new episodes stay within its own replay. The original replay and final source
checkpoint are preserved. Additional data, gate behavior and policy adaptation
will diverge naturally; this is a small adaptive online comparison, not a
three-seed real-control study or proof of reduced time to completion.

## First-block and control-health review (October 25 local time)

Galaxy 1 completed the additional 8000 decisions at 288773 lifetime decisions,
reaching 10000 new decisions across the two stages. The extension took 1940.766
seconds (4.12 decisions/s), added 1000 model updates and 86 actor updates, and
ended with 18 timeout episodes, one death and one budget interruption. Maximum
geometric progress in the extension was 2.36%; the median across 19 completed
episodes was 1.78%. No RAM-confirmed completion or health loss was logged.
The first health check's 2.68% remains the best across these 10000 new decisions;
neither block exceeds the historical record. Replay had no rejected archives,
stderr was empty, and the saved checkpoint digest was verified. The final gate
blocked imagination on terminal MAE 0.5048, keeping the original limit 0.5.

Decision: keep Galaxy 1 stopped at this boundary and diagnose exploration,
reward balance and action behavior before spending the optional 20000-decision
extension. Frequent timeouts indicate a navigation/exploration limitation;
additional elapsed time alone is not evidence of useful learning. The zero
legacy intrinsic reward column is expected when the world-model disagreement
bonus is used only in imagination; it does not mean that bonus was disabled.

The Galaxy 2 continued-control health check completed 2000 decisions in 238.062
seconds (8.40 aggregate decisions/s), ending at 872272 lifetime decisions. It
added 60 model updates and one actor update. Forty-four episodes ended in death,
four were interrupted at the budget boundary, and none completed the mission.
Best geometric progress was 11.90% in Spinning and 7.92% in Double Time; median
completed-episode progress was 6.18% and 3.60%, respectively. The final gate
blocked imagination on Spinning terminal reward MAE. No stderr error or replay
rejection was reported, and the final checkpoint digest was verified.

Decision: start the matching 2000-decision candidate health check; do not extend
either arm until both results have been reviewed. Aggregate speed measurements
include startup and shared hardware contention, so they are not isolated
algorithm benchmarks. Detailed counters and trajectory provenance are in
[the block review](assets/performance/staged-first-block-review-20261025.json).

Trajectory inspection also highlights a measurement limitation: the Spinning
high-water sample is at Y=-1740.68, below the initial Y=-1150, shortly before
death; the Double Time maximum is at Y=-1428.96. The existing minimum reward
height (-1800) does not prove platform contact. These projections can include
airborne/falling motion and must not be interpreted as verified traversed course
segments. Plot legends therefore say "Reward-gated max projection". Review
support/contact or course geometry before treating a geometric gain as navigation
improvement; preserve reward settings throughout the paired health checks.

![Galaxy 1 extension best trajectory](assets/performance/staged-smg1-sweet_sweet-20261025.png)

![Galaxy 2 control Spinning trajectory](assets/performance/staged-smg2-control-spinning-20261025.png)

![Galaxy 2 control Double Time trajectory](assets/performance/staged-smg2-control-double_time-20261025.png)

## Paired health-check decision and Galaxy 1 exploration hypothesis

The Galaxy 2 candidate completed 2000 real decisions in 205.547 seconds, with
42 deaths, four budget interruptions and zero mission completions. Spinning
maximum/median episode projection was 11.97%/6.79%; Double Time was 7.86%/3.48%.
These small differences from control do not demonstrate better navigation,
especially given falling-motion contamination and one training seed. Both
arms added 60 model updates; candidate added 11 actor updates versus control's
one. The candidate's last gate passed. Its final checkpoint hash was verified;
stderr was empty. Proceed to a bounded 23000 additional control decisions
(25000 total), then review before giving the candidate the matched extension.
Do not alter real rewards or thresholds during this comparison.
The [candidate health record](assets/performance/smg2-candidate-health-20261025.json)
retains its raw metrics, counters and episode summary.

A read-only Galaxy 1 diagnostic used eight shared ordinary held-out batches,
deterministic real posterior seeds and stochastic eight-step imagined rollouts
(seed 11, 8192 transition samples per checkpoint). No optimizer, game collection
or checkpoint write was performed. Mean actor entropy remained 3.44 nats out
of a 3.99-nat maximum, with mean maximum-action probability 0.143. This argues
against deterministic action collapse on these sampled states. Mean imagined
extrinsic reward was -0.0191, while the weighted disagreement bonus was only
0.000422. These predictions do not establish the cause of poor real navigation,
but motivate testing curiosity strength instead of increasing random action
probability for an already diffuse policy. The source checkpoint's corresponding
entropy was 3.43 and bonus 0.000422. Raw diagnostics are in
[the exploration audit](assets/performance/smg1-exploration-diagnostic-20261025.json).

Predeclared exploratory trial: resume two copies of the 288773-decision Galaxy 1
checkpoint for 2000 real decisions each, sequentially. Control retains imagined
disagreement coefficient 0.01; the sole candidate change is coefficient 0.1.
Both receive the same original actor/model/optimizer, same online seed 42,
separate initially identical replay copies, and unchanged model quality gates.
The candidate uses an explicitly migrated checkpoint with its own settings and
integrity manifest; its payload is a byte-identical source copy, preserving
optimizer and Torch/CUDA/replay RNG states while changing only the settings
manifest. Retained originals are immutable. Real reward labels and
the action space are unchanged. The first arm is the control, with review before
the candidate. Count all 4000 decisions against the 30000-new-decision ceiling,
leaving at most 16000 for a later reviewed extension. This short single-seed
screen cannot establish a general improvement or reward-coefficient optimum.
