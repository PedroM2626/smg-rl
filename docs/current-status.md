# Current evidence and next experiments

This page summarizes the evidence audited through October 28, 2026. It is a dated
snapshot, not a live process monitor. Older documents may use present tense for
their own dated experiments; their original observations remain historical.
No command here automatically allocates training or changes an existing budget.

A later owner-requested [reward-density comparison](human-route-density-pair-20261028.md)
tests the full human route at the previous per-distance scale against a
matched full-route control. Status and outcomes require inspecting its local
ledger; no result is implied by this document.

Later on October 28, the owner recorded full-start human completions of Sweet
Sweet and both Twisty Trials variations; exact isolated replays reproduced their
RAM star events. The human demonstrations are not trained-agent successes.
Separate candidate route profiles were prepared from those traces. The
[one-hour-per-reward SMG1 comparison](reward-hour-pair-20261028.md#final-result-after-recovery)
finished after the documented memory recovery, without a star or a clear
winner. Its first-segment percentages have been **withdrawn as invalid progress
measurements**: [video and telemetry](progress-metric-interpretation-20261028.md)
show a large score at the initial platform. A conservative landmark reanalysis
found no candidate matches across 16 frozen attempts and no native stars.

The subsequently requested [frozen IQN checkpoint evaluation](frozen-iqn-evaluation-20261027.md)
completed six canonical episodes with zero stars. Greedy evaluation repeatedly
reached the mapped opening and then died. A goal-distance tracker was tested
offline. A subsequent [frozen-vision genetic experiment](genetic-goal-distance-20261027.md)
selects distance fitness explicitly; it does not alter the IQN reward or replay.
Its five-minute pilot ended without stars or improvement over the imported head;
see the [completed pilot audit](genetic-goal-distance-20261027.md#completed-pilot-and-audit).
The owner's later three-hour distance-fitness run also completed with no stars
and no improvement over its reevaluated initial champion; see the
[three-hour results](genetic-goal-distance-20261027.md#owner-requested-three-hour-run).
The subsequent [robust-selection calibration](robust-genetic-plan-20261027.md#calibration-result-blocked)
completed 49 episodes, but no mutation scale passed its preregistered gate.
The proposed training arms were therefore not launched.

The subsequent [reward-scale A/B](reward-scale-study-20261028.md#completed-matched-comparison)
completed four 6,144-decision blocks and sixteen frozen evaluations, with zero
full-start stars. The candidate missed the preregistered improvement gate for
both training seeds; no larger block was allocated. This short two-seed screen
does not establish a general conclusion about reward scales.

The later [chocolate demonstration comparison](chocolate-frontier-search-20261028.md#completed-demonstration-comparison)
used equal offline IQN updates and real scripted partial full-start trajectories.
Neither demonstration seed passed the continuation gate; sixteen frozen episodes
contained zero stars. Two control policies each reached beyond the mapped opening
at least once during noisy evaluation, but subsequently died. This is a small
same-Wii-state screen, not a full-level success or a general rejection of demonstrations.

The later [scripted Sweet Sweet search](sweet-manual-search-20261028.md#first-search-and-reviewed-continuation)
replayed a 114-decision partial path from the canonical start to a grounded
location, with zero stars. This is physical scripted navigation, not a trained
policy or a full course map. The search blocks have ended.

The later [expanding-route trial](dynamic-route-night.md#completed-result) used
IQN/IMPALA with six canonical-start workers and stopped after two main blocks
without a validated extension or star. It recorded 100,884 real decisions and
666 completed attempts; the largest exploratory opening advance was 66.25%,
and the largest grounded goal-axis projection was 4.016% in a different
episode. Neither quantity measures native mission completion. Its new reward
scale and reset replay/optimizer mean this is not a controlled comparison with
the source checkpoint. The [experiment history](experiment-history.md) links
the evidence across stages.

## What is established

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Earlier four-hour SMG1 IQN/IMPALA allocation | Stopped after roughly 2h18 of the intended 4h due to disk exhaustion; zero observed RAM stars | Failed block, not a completed four-hour experiment |
| Last valid checkpoint from that allocation | 749,006 total real decisions; 12,125 completed attempts | Source candidate only after checking its manifest, replay, signatures, state and available resources |
| Later expanding-route IQN checkpoint | 100,884 new real decisions and 666 completed attempts in a fresh reward/replay branch | A subsequent frozen evaluation found no stars and lower mean opening progress than the source; different reward, replay and starts prevent attributing the change to one factor. See [frozen comparison](reward-scale-study-20261028.md#frozen-policy-baseline). |
| Failed IQN block | 97,668 acknowledged decisions after the valid checkpoint; 148 acknowledged decisions missing from CSV tail | Real collection occurred, but the final weights/replay were not saved and the tail is incomplete |
| Best new IQN collection trajectory | 6.5860% geometric projection from canonical start toward goal, followed by death | Exploratory maximum under changing weights/epsilon; not a frozen result or mission completion fraction |
| Earlier frozen IQN/IMPALA evaluations | No stars; Sweet reached 23.64% of its mapped opening, SMG2 cases reached 100% of their mapped openings and died | Short opening measures, not full-mission percentages |
| Full-visual GA | Zero stars; 8.0665% unchanged baseline vs 7.9463% best mutated-vision collection projection | No measured improvement in that trial; not a controlled algorithm or vision ablation |

Sources: [IQN failure audit](iqn-four-hours-20261027.md),
[IQN result data](assets/performance/iqn-four-hour-result-20261027.json),
[frozen evaluation](impala-final-results.md), and
[full-visual genetic audit](genetic-visual.md). The geometric progress projection
uses X/Z positions and the start-to-near-star direction. Curved paths, vertical
motion and platform geometry make it unsuitable as a true mission-completion
percentage. An opening-route fraction has a different denominator. Neither
metric replaces full-start, game-specific RAM-confirmed star collection.

The disk failure was traced to repeated slot-9 staging retention and a later
checkpoint write failure. The implementation was changed to bound retention and
check headroom. Owner-authorized subsequent cleanup removed verified duplicate
staging copies and compressed inactive worker images without changing their
content. See [failure and cleanup](iqn-four-hours-20261027.md#root-cause-and-implemented-prevention).
Free space can change; measure it again before a run. NTFS-compressed profiles
can have decompression overhead. ROMs, savestates, profiles and checkpoints are
local; GitHub cannot restore them. Failed partial artifacts remain evidence, not
valid checkpoints.

## Which configuration and objective

The generic CLI default is `config/impala-iqn-smg2.yaml`; the last SMG1 run used
`config/smg1-six-two-hour.yaml` and a separately audited local checkpoint. The
former six-hour runs, initial Rainbow/IQN comparison, PPO and RSSM runs each
have separate signed identities. Do not treat the CLI default as the last SMG1
configuration or transplant old replay into a changed action/reward profile.

The source SMG1 IQN used an IMPALA-style residual encoder, stacked grayscale
frames, IQN with Rainbow-style additions, eight exclusive actions and 12-frame
control. Its online visual network trains through gradients. The reward is
new grounded high-water advance through a verified opening, RAM star +20,
death/timeout -2 and nonfatal HP loss -0.25 per HP; gamma is 1 and there is no
time cost or curiosity. The short route ends before most of the mission, so
additional route credit is absent beyond the opening. The later expanding-route
trial used a changed normalization and canonical starts only, but no extension
was accepted. No evidence shows that
merely running longer will solve that coverage or exploration limitation.

## Status and evaluation boundaries

Before starting a job, inspect actual controller and Dolphin PID/creation times,
checkpoint hashes, available disk/commit memory and any stop markers. A stale
`status.json` may still say `running` after failed checkpoint finalization.
The dated read-only SMG1 snapshot command is in the [README](../README.md#installation-and-local-operation).
The CLI `status`/`stop` commands apply to their documented parallel-run layout;
they should not be assumed to control an arbitrary model-free supervisor.

Measure decisions and completed/partial attempts separately. Split full-start
and assisted starts; keep training collection maxima distinct from frozen
evaluation. Record deaths, HP, action occupancy, reward components, grounded
trajectory, model/replay hashes, failures, and RAM star evidence. The same saved
Wii state under seed labels 11/23/47 does not imply independent game RNG or
unseen environmental conditions. Repeatability, stochastic policy behavior and
generalization require explicitly varied, documented states or timing.

## Next experiments

These are proposals, not results or an authorization to dispatch a job:

1. **Use the completed frozen comparisons as baselines.** Reuse their videos,
   exact action logs and trajectories while keeping same-Wii-state action draws
   distinct from independent environments. See [the latest comparison](reward-hour-pair-20261028.md#final-result-after-recovery).
2. **Improve measurement and validate the mapped course.** Full-start human
   recordings now cover each selected mission. Ground contact, moving surfaces,
   route order and evaluation metrics still require independent checks; the
   withdrawn first-segment projection must not be reported as visited progress.
3. **Screen RAM fields with negative controls.** Exact replay of the complete
   human paths can provide more diverse paused memory snapshots. An address
   becomes a progress signal only after checks against idle animations,
   backtracking, falls, death, resets, multiple stages and separate game boots.
   No native course-progress address has been established in either game.
4. **Choose a new bounded experiment from the evidence.** Reward scale, action
   timing, controls and starts remain competing hypotheses. Keep signed replay,
   source checkpoints and quality gates separate and report full-start RAM
   stars separately from geometric or assisted-start diagnostics.

Direction-carry jumps, demonstrations, alternative algorithms, observation
resolution and additional curriculum starts are further hypotheses, not proven
fixes. The prior frame-4 test was unfavorable and the full-visual GA did not
establish a benefit. Change one factor at a time, rather than combining them
before identifying the reward and control failure modes.
