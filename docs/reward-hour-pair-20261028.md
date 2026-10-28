# One additional hour per Sweet Sweet reward

Owner-authorized October 28 continuation. This is a new finite study following
the completed inconclusive short A/B. It does not overwrite or retroactively
change that study's gate. One training seed (142) per reward is intentionally an
exploratory trade-off: results cannot establish statistical significance or
generalization.

## Source and schedule

The opening control and full-human-route candidate each resume their own
reviewed 6,144-real-decision seed-142 checkpoint from
`.local/human-route-ab-fixed-20261028/state.json`. Each exact resume preserves
that branch's online and target weights, optimizer, replay, counter and RNG;
the branches never exchange replay or reward data. Source artifacts and the
original 749,006-decision common ancestor remain unchanged. Both use IQN,
IMPALA, four stacked grayscale frames, eight exclusive 12-frame actions, six
workers and the same canonical-start distribution. There is no world model,
imagination or novelty bonus.

One additional hour per arm is split into two 1,800-second finite blocks. The
order is control half, candidate half, control second half, candidate second
half; one training job at a time. Every block requires a clean wall-budget
finish, reconciled real/replay/attempt counts, finite metrics and matching
checkpoint/replay signatures before its evaluation or a successor. The normal
training clock includes startup; checkpoint and cleanup can add wall time.
Each block has a 999,996-decision ceiling in addition to the time cap, and must
stop for memory, disk or owned stop markers. Such a premature stop is not an
hour and must be reviewed, never silently retried. Source and final counts are
reported separately. Equal wall time may yield unequal physical decisions or
optimizer updates; these are reported and limit direct sample-efficiency claims.

After each half-hour block, run one greedy and three stochastic frozen full-start
evaluations, action seeds 11/23/47, epsilon 0.1, at most 600 decisions per
episode. Evaluation decisions and videos are separate from training. The
original Wii savestate is identical in those action draws. Record both
training-collection and frozen outcomes without conflating them.

## Predeclared measurements

- Native full-start star, verified from game RAM, is the actual success signal.
- Strict grounded human-route progress uses the same signed schema-5 route for
  both arms; its previously observed zero floor remains an important limitation.
- First-segment progress is a **diagnostic**, not a reward: maximum alive and
  grounded X/Z projection onto the first human-route segment, with a 750-unit
  horizontal corridor, 500-unit height tolerance and 2,500-unit discontinuity
  limit. It ranges from zero to one for that first segment only. The wide
  corridor can count paths beside the human route and the measure saturates at
  the end of the first segment; neither saturation nor the fraction establishes
  full-course completion. It was defined before dispatching the new blocks.
- Also retain alive-grounded geometric projection toward the near-star point,
  health/damage, death, timeouts, attempts, actions, real decisions and updates.
  The geometric projection is not a native mission percentage or route length.

The already-recorded seed-142 baseline has greedy first-segment scores of
100% for both controls. Its three stochastic draws average 93.83% control and
100% candidate under this new diagnostic. These are reused baseline data and
already show saturation; they are not independent validation or proof that
the candidate is better.

The state and exact command ledger are `.local/reward-hour-pair-20261028/state.json`;
the supervisor alone owns all four stages. After they finish, audit real totals,
hashes, evaluation policy identity and outcomes before reporting. Never duplicate
an active/completed block, blindly retry partial data, or automatically add
another hour. Reused prior evaluations and source checkpoints remain identified.

## Interrupted outcome

The study stopped during the candidate's first block. The configured memory
guard ended that block with `memory_pressure`; the supervisor refused to treat
the stopped block as a clean 30-minute completion and did not launch its
evaluation or either second block. The post-cleanup memory snapshot does not
establish which resource crossed the guard while the workers were active.
No automatic retry or extra training was made.

| Arm, first block | Time including cleanup | New real decisions | Completed attempts | Deaths | RAM stars |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opening control | 1,812.96 s | 39,942 | 396 | 378 | 0 |
| Human-route candidate | 1,052.21 s | 29,184 | 231 | 157 | 0 |

Both source checkpoints and both resulting checkpoint/replay pairs passed
SHA-256 and signed-profile checks. Their retained CSV decisions reconcile
exactly with the acknowledged new real decisions; each has six interrupted
collector cuts, and neither reports nonfatal HP loss. The candidate checkpoint
is retained as a partial result, not a completed half-hour or an exact matched
comparison.

The control's frozen 30-minute evaluation completed four full-start attempts,
all timeout, with zero RAM stars. Its greedy first-segment diagnostic was
62.92%; the three stochastic draws averaged 58.23%. These numbers describe
only the first segment under the wide diagnostic corridor. The corresponding
pre-continuation stochastic mean on reused seed-142 evaluation data was 93.83%;
this small same-state comparison is descriptive and does not establish the
cause of the change. The strict full-route score remained zero in all four
episodes. The candidate was not evaluated after its interruption; therefore
there is no matched result after one hour, or even a matched half-hour result.

The [interruption evidence](assets/performance/reward-hour-pair-interrupted-20261028.json)
includes exact counts, retained CSV/checkpoint hashes, evaluated control
trajectories and videos, and the post-cleanup memory snapshot. Original ROMs,
states, profiles, source checkpoints and all partial artifacts remain local.
