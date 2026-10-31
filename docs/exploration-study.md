# Exploration study: October 29, 2026

The owner authorized implementing the critique after the original day run plateaued. The old run stopped intentionally with an audited checkpoint at 1,383,998 cumulative real decisions and 9,545 attempts; zero RAM stars. Originals remain retained.

## Physical feasibility and curriculum

The human demonstration reproduced its RAM star and all 1,735 images. Eleven intermediate anchors passed independent restore, health, grounded and idle checks. Loading an anchor pays zero reward. Assisted success is never canonical success.

Normal/carry-only jump fixtures cleared the next belt in 0/16 trials. Raw long-jump controls cleared it in 9/9, and the actual eight-label motor state machine cleared it in 3/3. These are scripted assisted checks, not learned success or proof that all ordinary jumps are impossible. One earlier motor recording failed JSON serialization; its exact decision count is unknown (upper bound120), retained separately without counting successes.

## Bounded comparison

All arms transfer the same online IQN/IMPALA weights, with fresh optimizer, target and250,000-transition replay, seed142, four stacked frames and four workers. Gamma is0.997; artificial timeout is a zero-reward truncation with bootstrap. Death remains-2, HP loss-0.25 and RAM star+20. Human route landing verification and progress density remain intact.

- A: corrected objective and retention package, original eight-action controls.
- B: A plus direction carry and crouch-prepared long jump, still eight logical actions.
- C: B plus backward curriculum; half the resets remain canonical. Eight stars among the last ten current-anchor attempts unlock an earlier anchor.

Order C,A,B. Maximum four controller hours per arm, including cleanup, health3072 decisions then reviewed finite blocks. Absolute deadline October30 13:12:55 BRT. One million decisions without new canonical credit, stars or curriculum advancement triggers a plateau stop. No automatic extension or retry after failure.

Each arm receives one greedy and three stochastic frozen canonical episodes before and after training, at most600 decisions each, with videos and unchanged checkpoint hashes. Same Wii start and one training seed limit generalization. Route fractions are not native mission percentages. A tests a package, not gamma alone; B tests the motor package; C adds curriculum to B.

Local authority: `.local/exploration-study-20261029/state.json`; supervisor: `scripts/run_exploration_study.py`. The old day supervisor must not resume. Draft transfers/configurations are retained as superseded preparation. Optional demonstration minibatch fitting is deferred because two-frame combined human controls require a separately validated action conversion.

Portable validation:42 relevant tests passed before dispatch; Ruff is required before commit. Final evidence and frozen outcomes remain pending while the bounded study runs.

## Outcome and owner stop (October 30)

C completed its four hours (489,776 decisions, 3,542 attempts, 1,777 assisted) with zero RAM stars; its frozen canonical evaluation had four timeouts and zero route credit. The owner stopped A at 457,368 decisions (owner_stop, zero stars); B never ran. Partial data is preserved without retry.

## Anchor-40 unit test of the learning pipeline

From `human-waypoint-40` the human needs 470 frames (about 40 decisions). Before any further objective comparison, the pipeline was checked on this short segment without optimizer updates:

1. Raw human controls through `GalaxyEnv` reach the RAM star; the replay row has +20, zero discount, terminal and assisted.
2. Majority-vote quantization of the two-frame human controls into twelve-frame labels dies after six decisions (lost long-jump timing). A bounded savestate beam search over the eight actual `motor8-longjump-v1` labels found a 40-decision sequence that reaches the RAM star, confirmed through the production environment reset (reward 21.4). The control profile is sufficient; this is an assisted scripted fixture, not learned success.
3. The 969 assisted deaths concentrate near X 19,000-20,000 / Z -12,500 and X 22,000-22,500.
4. A uniform random policy: 0/100 stars, 100/100 deaths, median 24 decisions.

Conclusion: the gap was exploration, not control or reward plumbing.

## Anchor diagnostic S1

`config/anchor-diagnostic-s1-20261030.yaml` transfers C's final online weights (fresh optimizer, target and replay; epsilon 0.3 to 0.1 over 20,000 decisions) and starts every attempt from the curriculum (`curriculum.anchor_diagnostic: true`, the only setting that may exceed probability 0.5). One in ten anchor-40 starts per worker replays the 40-decision search sequence (`demonstration`). Those transitions are real emulator experience, but their outcomes are counted as `scripted_demonstration`: never learned success and never curriculum evidence. Pre-registered pass: at least 10% policy-driven stars over the last 100 anchor attempts or a frontier event, and a frozen assisted-start star.

| Block | Real decisions (cumulative) | Policy stars / attempts | Scripted stars | Notes |
|---|---|---|---|---|
| 1 (30 min) | 43,076 | 35 / 844 | 97 / 97 | Last 44 attempts: 13 stars |
| 2 (30 min) | 88,728 | 124 / 580 | 45 / 45 | 8/10 stars moved the frontier from waypoint 40 to 38 at 69,356 |
| 3 (60 min) | 188,872 | 44 / 711 | 12 / 12 | Waypoint 38: no policy episode passed X 21,000 |
| 4 (60 min, `s1b`) | 295,492 | 75 / 790 | 90 / 90 | Waypoint 40: 51 of last 100; waypoint 38: 0 / 630 |
| 5 (60 min, `s1c`) | 366,348 | 51 / 508 | 129 / 129 | First waypoint-38 policy stars: 11 / 423 |
| 6 (60 min, `s1d`) | 434,336 | 70 / 473 | 122 / 122 | Waypoint 38: 34 / 379; waypoint 40: 36 / 94 |
| 7 (60 min, `s1e`, epsilon 0.03) | 500,732 | 128 / 572 | 45 / 144 | Waypoint 38: 31 of last 50; frontier moved 38 to 34 at 444,348 |
| 8 (60 min, `s1e`) | 544,404 | 29 / 207 | 5 / 53 | Waypoint 34: 0 / 169, but 140 reach the waypoint-38 region and time out |

Frozen evaluations after block 2 (`record-dqn --assisted-start --anchor`): at waypoint 40, epsilon 0.1 reached the RAM star in 2/3 episodes (61 and 77 decisions); greedy timed out. At waypoint 38, 0/4. Both blocks passed `audit_block`. Same Wii state and one training seed: not generalization and not full-start success.

After block 3, waypoint-38 policy episodes never passed X 21,000: the conveyor carries Mario back, and the human clears a 1,250-unit gap with a long jump after deliberately waiting. Progress-scored search ran ahead of the human clock and landed on a lower surface; a time-synchronized search found a 62-decision eight-label sequence from waypoint 38 (environment-confirmed RAM star, reward 22.4). `config/anchor-diagnostic-s1b-20261030.yaml` (same replay signature) adds it through `demonstration.sequences`. In block 4 the policy reproduced the first seven demonstration labels from waypoint 38 and diverged at the long-jump preparation.

After block 4 a frozen greedy evaluation from waypoint 40 reached the RAM star in 48 decisions, completing the pre-registered pass for this anchor. Block 5 (`s1c`) only shortens the demonstration interval from ten to five starts.

Block 6 adds a DQfD large-margin loss (margin 0.8) on scripted demonstration decisions only; the replay stores a per-decision `demonstration` label (older replays load as unlabeled). After block 6, frozen greedy evaluations reached the RAM star from waypoint 38 (62 decisions) and waypoint 40 (40 decisions), matching the demonstration lengths: the learned policy reproduces the demonstrated segments from these exact Wii states, which is imitation on fixed states, not generalization. Collection at epsilon 0.1 limited the 8/10 frontier rule on the 62-decision segment, so block 7 (`s1e`) re-signs the checkpoint with only epsilon changed to 0.03 (weights, optimizer, counters and replay unchanged).

Demonstrations may now end before the episode: after the sequence, the policy continues, the episode stays labeled scripted, and only scripted decisions carry the margin label. Anchor-to-anchor searches use `--goal-anchor` (grounded within 200 units of the next anchor); waypoint 34 reaches the waypoint-38 region in 14 decisions (environment-confirmed, alive). A single full-route search from waypoint 01 drifted off the human path and was replaced by these chained segments. Beam-search node savestates once filled the disk (about 226 GB); they are now deleted as the beam advances and the stale ones were removed, keeping every report.

Searches chained anchor-to-anchor segments 29-34 (33 decisions), 23-29 (53), 18-23 (21), 14-18 and 08-14, all environment-confirmed and alive. Waypoint-34 demonstrations that hand over near waypoint 38 never produced a star (0/99): the policy had memorized open-loop sequences from the exact anchor-38 and anchor-40 states, and arrival from waypoint 34 has a different conveyor and platform phase. With 0-60 random no-input frames after the verified reset (`environment.assisted_start_jitter_frames`; the route is initialized from the verified pre-jitter context, so no prefix is paid), the block-8 greedy policy fell from deterministic success to 1/4 at waypoint 40 and 0/4 at waypoint 38. Block 9 (`s1f`) therefore trains with that jitter and without scripted demonstrations.

Actor/learner overlap (`parallel.overlap_learning`) slowed this machine: recent medians were 135 ms per update and 18 ms per inference versus 66 ms and 10 ms serially (about 16 versus 19 decisions per second) with idle CPU headroom, consistent with the four Vulkan emulators and CUDA sharing one GPU. It stays off by default.

Block 9 (603,584 decisions, jitter): waypoint 34 reached the waypoint-38 region in only 2/244 attempts (previously 140/169), waypoint 38 scored 0/23 and waypoint 40 12/46. Much of the earlier competence was exact-state memorization. Lineage S2 (`config/anchor-diagnostic-s2-20261030.yaml`) keeps the S1 weights and replay (only epsilon re-signed to 0.05) but restarts the backward frontier at waypoint 40 under the jittered start distribution.

S2 under jitter: block 1 stopped on the physical-memory guard at 634,560 decisions (audited as a resource stop; the frontier moved from waypoint 40 to 38 at 605,812). Block 2 reached 723,124 decisions: waypoint 40 scored 71/107, waypoint 38 3/451 with most attempts never passing X 20,500 before the long-jump gap. `config/anchor-diagnostic-s2b-20261030.yaml` keeps jitter for 80% of assisted starts and restores the exact anchor for 20%, where only the open-loop demonstrations (with the margin loss) run. Its first launch could not allocate the 3.3 GiB replay because an unrelated workload held most of the machine's commit memory; it has not trained yet.

S2 block 3 (`s2b`) stopped on the physical-memory guard at 738,228 decisions while unrelated desktop applications held about 11 GB; `config/anchor-diagnostic-s2c-20261031.yaml` continues with three workers (replay re-signed with `reduced_workers`, data unchanged). The three-worker block reached 826,344 decisions: waypoint 40 scored 66/107 under jitter, every exact-start demonstration from waypoint 38 succeeded (105/105), yet jittered policy starts from waypoint 38 scored 0/411. The episodes show a learned retreat: the policy lets the conveyor carry Mario back, walks backward and waits for the no-progress cut, which paid zero with bootstrap while attempting the long-jump section risks the -2 death.

Lineage S3 (`config/anchor-diagnostic-s3-20261031.yaml`) therefore uses `regional_reward.timeout_mode: idle_terminal_penalty`: a 60-decision no-progress cut is an absorbing failure with the death penalty (-2), while only the global episode budget remains an unpenalized bootstrap truncation. It transfers the S2 online weights with fresh optimizer, target and replay and restarts the frontier at waypoint 40.

S3 block 1 (85,668 decisions): the frontier moved from waypoint 40 to 38 at 28,419. Jittered waypoint-40 policy starts recovered to about 40% per 50 attempts (147/669 overall). From waypoint 38 the retreat disappeared: 481 deaths versus 283 no-progress cuts, and 67 attempts passed X 21,000 (previously 2), still with zero policy stars; scripted exact-start demonstrations succeeded 421/422.

S3 block 2 (170,658 decisions, audit passed) kept the frontier at waypoint 38. Jittered waypoint-40 policy starts reached 51, 63 and 70 stars per 100 attempts (192/321); scripted exact-start demonstrations succeeded 430/430. Jittered waypoint-38 policy starts scored 1/1,143. Per-step telemetry shows the block-1 "no retreat" reading was too optimistic: most failures end at X 18,000-20,250, behind the jittered start (X 20,231-20,550). The policy is carried back by the conveyor and dies or waits, and the median maximum X was 20,514. Because every exact start runs the open-loop demonstration, policy episodes from waypoint 38 only ever start up to 60 frames of conveyor drift away from the demonstrated states. `config/anchor-diagnostic-s3b-20261031.yaml` changes only `environment.assisted_start_jitter_frames` from 60 to 24 (outside the replay signature, so weights, optimizer and replay continue) as the first step of a jitter curriculum.
