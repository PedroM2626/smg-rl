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
