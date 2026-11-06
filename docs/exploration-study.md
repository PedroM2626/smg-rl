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

S3b block 1 (253,239 decisions, jitter 0-24 frames, audit passed): waypoint 40 kept 148/264 policy stars, but waypoint 38 scored 2/1,043, with more no-progress cuts (604) than deaths (437), most ending at X 18,000-19,250. Narrower jitter did not help, so the start distribution alone is not the obstacle. Because nearly every attempt dies, letting the conveyor carry Mario back and paying the discounted -2 cut is still worth slightly more than attempting the section.

`config/anchor-diagnostic-s3c-20261031.yaml` adds `demonstration.handover: uniform_prefix`, a backward-from-demonstration schedule (Salimans and Chen, 2018). Each exact start replays a uniformly random prefix (0 to the full length) of the anchor's verified sequence, then hands over to the policy. Only the replayed prefix carries the margin label, the whole episode stays a scripted fixture (never curriculum evidence or a policy star), and the attempt ledger records `script_prefix`. Exact starts rise from 20% to 40%. Neither change touches the replay signature.

S3c block 1 (341,409 decisions, audit passed): jittered waypoint-38 policy starts scored 0/814, yet prefix-handover episodes from the exact waypoint-38 state succeeded 36/89 when the policy took over within the first 8 decisions, rising to 65/65 and higher for later handovers. From waypoint 40: 19/33 at the earliest handovers and 131/205 for jittered policy starts. The policy therefore solves the waypoint-38 section closed-loop from the exact state, but 0-24 frames of no-input drift on the conveyor still defeat it.

Emulation is deterministic and the final criterion is a frozen-policy RAM star from the canonical start. `config/anchor-diagnostic-s3d-20261031.yaml` therefore drops start jitter (robustness moves to the later benchmark) and keeps exact anchors with prefix handover on every third start. It adds the verified anchor-to-anchor goal segments 08-14, 14-18, 18-23, 23-29 and 29-34 as demonstrations. Earlier-anchor demonstrations hand over at the next anchor region in the phase the previous segment actually produces, so the backward frontier trains on arrival states rather than on isolated anchor savestates. Frontier moves still require 8 of the last 10 policy-driven (non-scripted) starts. Segments 01-08 have no verified sequence yet.

S3d block 1 (432,372 decisions, audit passed): the frontier moved from waypoint 38 to 34 at 365,220. Exact-start policy episodes scored 94/306 from waypoint 38 and 100/157 from waypoint 40. From waypoint 34, policy starts scored 0/774 and handovers 0/388, even after the full 14-decision segment. Policy episodes from waypoint 34 reach the waypoint-38 region (median maximum X 20,209) and die at the conveyor and long-jump section in the phase they arrive in. A continuous search from waypoint 34 to the star, resumed from an earlier run's 54-decision prefix that had already crossed the gap (X 24,441) before its emulator crashed, runs alongside S3d block 2.

The resumed search found an 82-decision sequence from waypoint 34 to the RAM star (environment-confirmed, reward 21.2) whose first 14 decisions equal the earlier segment. S3d block 2 (527,055 decisions, audit passed) kept waypoint 34 at 0/838 policy and 0/425 handover episodes, with retreats rising again (536 cuts versus 302 deaths). `config/anchor-diagnostic-s3e-20261031.yaml` replaces the waypoint-34 demonstration with the continuous 82-decision sequence, so prefix handovers cover the conveyor and long-jump section in the phase that arrival from waypoint 34 actually produces.

S3e block 1 (614,340 decisions, audit passed): waypoint-34 policy starts scored 0/797 and handovers 4/414. Even full 80-82-decision handovers scored 0/20, although the same sequence earned the RAM star in the search's environment confirmation. Every such episode ended at decision 81 with a no-progress cut. Route credit is paid only for sweeping the human route's waypoints in order. The search path leaves those waypoints after region 38, so it earned nothing for 60 decisions and hit the 720-frame cut one decision before the star. The no-progress cut, not the policy, was blocking valid off-route paths from waypoint 34 (and probably many policy attempts).

Lineage S4 (`config/anchor-diagnostic-s4-20261031.yaml`, prepared by a local script that records its transfer) raises `regional_reward.max_idle_frames` from 720 to 1440 (120 decisions). That changes the replay signature, so the S3e online weights transfer with fresh optimizer, target and replay. The same-bank frontier (waypoint 34) carries over with an empty recent window, and the event is recorded in the progress file.

S4 block 1 (80,400 new decisions on fresh replay, audit passed) produced the first policy stars from waypoint 34: 48/546 (6-16 per 100). Handovers from waypoint 34 rose with prefix length, from 1/25 for the earliest takeovers to 28/28 at prefixes 72-79. Waypoint 40 scored 33/62 and waypoint 38 5/65, sampled only on the 20% later-anchor share. Searches also verified the remaining early segments: 01-02 in 10 decisions (timed; untimed scoring was lured off the route by a teleport), 02-04 in 36 and 04-08 in 29 with long-jump macros. `config/anchor-diagnostic-s4b-20261031.yaml` adds them as demonstrations, leaving the replay signature unchanged, so every anchor from 01 to the star now has a verified sequence.

S4b block 1 (144,822 decisions, audit passed): waypoint-34 policy starts scored 60/462 (11-20 per 100), waypoint 38 14/49 and waypoint 40 37/59, with handovers 140/234. At epsilon 0.05 an 80-decision timing-sensitive segment receives about four random actions per episode, which caps the 8/10 frontier rule on policy-driven starts. `config/anchor-diagnostic-s4c-20261031.yaml` re-signs the checkpoint with only `dqn.epsilon` changed to 0.02 (weights, optimizer, counters and replay unchanged), as in S1 block 7.

S4c block 1 (201,150 decisions, 2,896 attempts, audit passed): waypoint-34 policy starts rose to 73/145, and the frontier moved from waypoint 34 to waypoint 29 at 161,595 lineage-S4 decisions. Waypoint 38 scored 48/63 and waypoint 40 27/40. From waypoint 29 the policy scored 0/437 and every handover failed (0/215), including full 30-33-decision prefixes. A deterministic replay of the waypoint-29 demonstration followed by the waypoint-34 sequence shows why. The 33-decision segment ends grounded about 120 units short of the waypoint-34 anchor, in a different conveyor phase. The waypoint-34 sequence then survives 54 more decisions and falls at decision 87. Handovers die earlier, at X 19,250-20,500, mostly falling off the conveyor section. This repeats the arrival-phase mismatch seen at waypoint 34. Two bounded continuous searches extended the 61 surviving decisions from waypoint 29. Path-matched scoring lost its whole beam at depth 21. Time-matched scoring found the RAM star in 110 decisions, confirmed through the production environment (reward 20.9). The first two searches stopped when the disk filled. Every frontier savestate also kept a retained duplicate inside the search's own emulator profile, so the search script now drops those duplicates. `config/anchor-diagnostic-s4d-20261031.yaml` replaces the 33-decision waypoint-29 demonstration with the continuous 110-decision sequence. Nothing else changes: the replay signature, epsilon 0.02 and the checkpoint stay the same.

S4d blocks 1-4 stopped early on the memory guard (15,936, 960, 1,536 and 768 decisions), because other local workloads held most of the Windows commit limit. Each partial checkpoint passed the resource-stop audit. A training block itself commits about 10 GB, so later blocks start only when at least 22 GB of commit and 12 GB of physical memory are free. Block 1 already showed that the continuous waypoint-29 sequence fixes the handover: takeovers at prefix 64 or later scored 23/29, earlier ones 0/42. Policy-driven waypoint-29 starts still scored 0/162.

S4d block 5 (69,978 decisions, audit passed, 19 decisions/s) produced the first policy stars from waypoint 29: 94/403, a steady 23-24 per 100. Handovers scored 127/198, and early takeovers now also succeed (prefixes 0-47: 27/91). Later anchors held up: waypoint 34 scored 18/45, waypoint 38 27/44 and waypoint 40 29/30. The frontier stays at waypoint 29, with 5 policy stars in the last 10 attempts.

S4d block 6 (70,674 decisions, audit passed): waypoint-29 policy starts scored 102/411 (19-29 per 100), a plateau near block 5, with handovers at 146/210. The failures split three ways. Some policy episodes stall on the conveyor near the waypoint-34 region (X 17,250-18,250) until the idle cut. Others fall behind the start (X 13,250-14,000). The rest die on the first jumps (X 15,500-16,250). The frontier stays at waypoint 29 (3 of the last 10).

S4d block 7 (66,135 decisions, audit passed): waypoint-29 policy starts rose to 117/365 (29-34 per 100 in full windows), with handovers at 136/178. Waypoint 34 scored 15/30, waypoint 38 22/33 and waypoint 40 34/38. The frontier stays at waypoint 29.

S4d block 8 (78,669 decisions, audit passed): waypoint-29 policy starts scored 120/455 (21-31 per 100), a plateau since block 5. Waypoint 34 fell to 10/32, mostly through timeouts. Timeouts from waypoint 29 are not stalls. In the last 120 decisions Mario still covers about 2,500 X units, spinning and jumping around the conveyor section. Even star episodes stop earning route credit at region 31, decision 20, and then run 90 decisions with no dense signal. The continuous search path passes 130-200 units beside human waypoints 32 and 37-39 (radius 100), and ordered credit cannot pass a missed waypoint.

`regional_reward.waypoint_lookahead` (route.py, default 0, at most 6, schema 5 only) lets a motion segment that sweeps one of the next N+1 waypoints skip the missed ones before it. The same landing confirmation applies. The signed route file and the anchor bank are unchanged. Replayed offline on block-8 star episodes from waypoint 29, the final credited waypoint moves from 31 to 36 with lookahead 1 or 2, and to 45 with lookahead 5, where the last credit lands near decision 102. Lineage S5 (`config/anchor-diagnostic-s5-20261101.yaml`) uses lookahead 5. This changes the replay signature, so the S4d block-8 online weights (505,806 decisions) transfer with fresh optimizer, target and replay. The waypoint-29 frontier carries over with an empty recent window.

S5 block 1 (70,041 decisions, audit passed): waypoint-29 policy starts scored 45/367. With a fresh replay the rate first dropped and then recovered to 9, 6, 11 and 19/67 per window. Deaths now dominate (249 deaths, 73 timeouts), while S4d block 8 had more timeouts. Waypoint 34 scored 1/31, waypoint 38 5/28 and waypoint 40 15/35. The frontier stays at waypoint 29.

S5 block 2 (83,274 decisions, audit passed): waypoint-29 policy starts scored 96/432 (18-25 per 100), back at the S4d plateau, with 240 deaths and 96 timeouts. The later anchors improved: waypoint 34 scored 14/31, waypoint 38 16/32 and waypoint 40 20/27. By geometric projection, about a third of waypoint-29 deaths happen in the first 50 decisions, before the conveyor section. Another third happen in the middle of the course, and many timeouts end near the star after 220 or more decisions. The frontier stays at waypoint 29.

S5 block 3 (65,004 decisions, audit passed): waypoint-29 policy starts scored 99/339 (27, 28, 32 and 12/39 per window), the first rate above the S4d plateau. Waypoint 34 scored 14/24, waypoint 38 19/26 and waypoint 40 17/19. The frontier stays at waypoint 29.

S5 block 4 (96,627 decisions, audit passed): the frontier moved from waypoint 29 to waypoint 23 at 238,467 decisions, with 8 policy stars in the last 10 attempts. Before the move, waypoint-29 policy starts scored 47/161 (31/100). Later anchors kept improving: waypoint 34 scored 27/55, waypoint 38 37/51 and waypoint 40 52/63. From waypoint 23, all 622 policy starts and all 325 demonstration handovers failed, mostly by death. The 53-decision waypoint-23 demonstration ends 180 units from waypoint 29, in a different conveyor phase. The policy trained from the exact waypoint-29 state does not carry over, and replaying the continuous waypoint-29 sequence after it falls off the course at decision 130. A continuous beam search from waypoint 23 to the star (`.local/anchor23-star-search-timed-20261101`) is running, following the same approach as waypoint 29.

The waypoint-23 search (beam 8, timed human tracking from waypoint 29, 53-decision prefix) found the RAM star after 167 decisions, confirmed by the environment (complete, not dead). `config/anchor-diagnostic-s5b-20261101.yaml` (arm S5b-continuous23) replaces the 53-decision waypoint-23 demonstration with this continuous sequence. The replay signature is unchanged, so S5b resumes the S5 block-4 checkpoint and keeps the same run and progress paths. The search remains an assisted control fixture and does not count as a learned completion.

S5b block 5 (109,332 decisions, audit passed): waypoint-23 policy starts went from 0/622 to 52/507 (9, 7, 13, 6, 17 and 0/7 per window). Handovers from the continuous demonstration succeed at almost every prefix of 104 decisions or more, and at 1-3 of every 6-16 before that. Most policy failures are deaths (390 deaths, 65 timeouts). The later anchors hold: waypoint 29 scored 13/37, waypoint 34 20/33, waypoint 38 14/24 and waypoint 40 25/30. The frontier stays at waypoint 23.

S5b block 6 (108,522 decisions, audit passed): waypoint-23 policy starts scored 61/491 (9-17 per 100), only slightly above block 5. Grouped by the last committed route waypoint, 257 of the 359 deaths happen before waypoint 27, and 122 of them right after waypoint 26 (median 74 decisions). Timeouts gather at waypoints 35-36, the same conveyor section that stalled waypoint 29 before lookahead. Most stars pass waypoint 46 at a median of 167 decisions. Waypoint 29 dropped to 6/37, while waypoint 34 scored 19/30, waypoint 38 15/27 and waypoint 40 31/37. The frontier stays at waypoint 23.

S5b block 7 (108,798 decisions, audit passed): waypoint-23 policy starts scored 53/519 (6-16 per 100). The rate has been flat at about 10% for three blocks. 275 of the 396 deaths happen before waypoint 27. All 159 deaths after waypoint 26 are full-health falls: median position (13134, -975, -9884), about 330 units short of waypoint 27, and median decision 74. The continuous demonstration crosses this 1,690-unit segment in decisions 23-29 with a crouch, a long jump and held forward motion. It lands at waypoint 27 by decision 30. The policy reaches the same gap about 45 decisions later than the demonstration and falls short of the landing. Waypoint 29 scored 14/28, waypoint 34 17/27, waypoint 38 22/29 and waypoint 40 21/27. The frontier stays at waypoint 23.

S5b block 8 (81,750 decisions, audit passed): waypoint-23 policy starts scored 44/381 (6-16 per 100), the fourth flat block. 197 of the 295 deaths happen before waypoint 27. Waypoint 29 scored 5/31, waypoint 34 8/22, waypoint 38 14/21 and waypoint 40 16/17.

Frozen evaluations of the block-8 checkpoint (`record-dqn`, no learning) separate exploration noise from policy quality. With evaluation epsilon 0 from waypoint 23, six seeds all reach the RAM star in 167 decisions. They end at the same final position as the continuous demonstration, so the greedy policy reproduces it exactly in this deterministic emulator. Greedy starts from waypoint 18 and waypoint 14 time out at 124 decisions (42.4% and 37.0% geometric projection). The canonical start dies after 27 decisions. The training-time rate of about 11% therefore measures robustness to the 0.02 exploration epsilon over a 167-decision course: an episode avoids every random action with probability 0.98^167, about 3%. It does not measure greedy competence from the frontier. These are assisted-anchor evaluations, not full-start completions.

Lineage S5c (`config/anchor-diagnostic-s5c-20261101.yaml`) uses an owner-approved hybrid frontier rule (`frontier_rule.mode: greedy_hybrid`). Every tenth policy start at the frontier is a frozen greedy probe (evaluation epsilon 0, deterministic quantiles). The probe is real experience in replay, but it is logged as `frontier_probe` and never enters the training-attempt window. The frontier moves only after two consecutive probe RAM stars from the current frontier and at least 10% stars over the last 30 epsilon-greedy training attempts there. The robustness floor is a fixed rate, not the earlier 8/10, because over longer courses an episode with no random action becomes geometrically rarer. The rule is a top-level key, so the replay signature is unchanged and S5c resumes the S5b block-8 checkpoint. The final gate is unchanged: a frozen greedy RAM star from the canonical start.

S5c block 9 (70,566 decisions, audit passed): under the hybrid rule the frontier moved from waypoint 23 to waypoint 18 at 734,268 decisions, after two greedy probe stars and 3/30 training stars. All six greedy probes from waypoint 23 reached the star, against 7/77 epsilon-greedy policy starts there. From waypoint 18, all 524 policy starts, 58 greedy probes and 301 demonstration handovers failed, mostly by death. The 18-to-23 goal segment again ends away from the exact waypoint-23 state, so the continuous waypoint-23 behavior does not carry over. A continuous beam search from waypoint 18 to the star (`.local/anchor18-star-search-timed-20261101`) is running. Waypoint 29 scored 8/31, waypoint 34 20/40, waypoint 38 20/35 and waypoint 40 25/28.

The timed and progress-tracking waypoint-18 searches died at gaps, and two wide-beam searches ended when Dolphin exited with code 0xC0000409. Adding the long-jump macros crossed the waypoint-39 gap, and short chained rounds then found a sequence that reached the RAM star at decision 210. It reproduced exactly in a raw emulator replay, but under S5c it stopped earning route credit after waypoint 26: it passes outside the 100-unit radius of waypoints 27-31. The idle limit would therefore end the episode at decision 160. The search now mirrors the training route credit (schema 5, lookahead 5), ranks nodes by committed waypoint and then distance to the next waypoint, and drops nodes 110 decisions without credit. It reuses the first 40 decisions of that sequence, up to waypoint 26.

The route-aware search (`.local/anchor18-route-search-s1-20261103`) found the RAM star from waypoint 18 in 175 decisions. Its environment confirmation under the S5c training configuration ended complete and not dead, with 29.53 return. `config/anchor-diagnostic-s5d-20261103.yaml` (arm S5d-route18) replaces the 21-decision waypoint-18 demonstration with this sequence. The replay signature is unchanged, so S5d resumes the S5c block-9 checkpoint. The sequence is an assisted control fixture, not a learned completion.

S5d block 10 (41,280 decisions, stopped on memory pressure after 27 minutes; resource-stop audit passed): with the route-aware waypoint-18 demonstration, the hybrid rule moved the frontier from waypoint 18 to waypoint 14 at 833,709 decisions, after two consecutive greedy probe stars and 3/30 training stars. From waypoint 18, policy starts scored 3/201 (0/100, then 3/100), greedy probes 2/22, and demonstration handovers 42/105. Waypoint 23 scored 3/12, waypoint 29 1/6, waypoint 34 6/14, waypoint 38 7/10 and waypoint 40 14/16. The 11-decision waypoint-14 demonstration ends near waypoint 18 rather than at the star, so a route-aware continuous search from waypoint 14 (`.local/anchor14-route-search-r1-20261103`) started before the next training block.

Waypoint-14 route search and S5e (November 3, 2026): a route-aware beam search
from `human-waypoint-14.state` (beam 32, long-jump macros, prefixed by the
11-decision waypoint-14 demonstration) was chained in 45-minute rounds. Rounds
reached route segments 24, 27, 35 and 40; the fourth ended with every node
idle-cut after waypoint 40, so the chain restarted from its segment-38 node and
found a 224-decision sequence to the RAM star. Under the S5d training
configuration it completed with reward 30.29. `config/anchor-diagnostic-s5e-20261103.yaml`
uses it as the waypoint-14 demonstration; the replay signature is unchanged. The
sequence is an assisted control fixture, not a learned completion.

S5e block 11 (November 3, 2026) resumed the block-10 checkpoint for one hour
(wall-clock stop, audit passed): 913,914 real decisions and 9,456 completed
attempts in total (78,720 new decisions). Frozen greedy probes from waypoint 14
reached the RAM star 6 of 24 times and training attempts from waypoint 14
reached it 5 of 231 times, so the hybrid rule moved the frontier from waypoint
14 to waypoint 08 during the block. All 119 training attempts and 13 probes
from waypoint 08 failed; their best route fraction was 0.47. Demonstration
episodes (41 of 136 stars from waypoint 14) are not counted as policy successes.

Waypoint-08 route search and S5f (November 3, 2026): the same chained
route-aware search from `human-waypoint-08.state`, prefixed by the 40-decision
waypoint-08 goal segment, reached route segments 29 and 38 in its first two
45-minute rounds and found a 212-decision sequence to the RAM star in the third.
Under the S5e training configuration it completed with reward 33.47.
`config/anchor-diagnostic-s5f-20261103.yaml` uses it as the waypoint-08
demonstration; the replay signature is unchanged. It is an assisted control
fixture, not a learned completion.

S5f block 12 (November 3, 2026) resumed the block-11 checkpoint for one hour
(wall-clock stop, audit passed): 987,108 real decisions and 10,233 completed
attempts in total (73,194 new decisions). Frozen greedy probes from waypoint 08
reached the RAM star 4 of 19 times and training attempts from waypoint 08
reached it 4 of 182 times, so the hybrid rule moved the frontier from waypoint
08 to waypoint 04. All 189 training attempts and 21 probes from waypoint 04
failed; their best route fraction was 0.23. Later anchors kept improving
(waypoint 29: 8 of 18, waypoint 34: 7 of 11, waypoint 38: 12 of 16 training stars).

Waypoint-04 route search and S5g (November 3, 2026): the chained route-aware
search from `human-waypoint-04.state`, prefixed by the 29-decision waypoint-04
goal segment, reached route segments 15, 27 and 41 in three 45-minute rounds
and found a 237-decision sequence to the RAM star in the fourth. Under the S5f
training configuration it completed with reward 34.81.
`config/anchor-diagnostic-s5g-20261103.yaml` uses it as the waypoint-04
demonstration; the replay signature is unchanged. It is an assisted control
fixture, not a learned completion.

S5g block 13 (November 4, 2026) resumed the block-12 checkpoint for one hour
(wall-clock stop, audit passed): 1,085,418 real decisions and 11,095 completed
attempts in total (98,310 new decisions). Frozen greedy probes from waypoint 04
reached the RAM star 16 of 29 times and training attempts from waypoint 04
reached it 7 of 271 times, so the hybrid rule moved the frontier from waypoint
04 to waypoint 02. All 135 training attempts and 15 probes from waypoint 02
failed with a best route fraction of 0.08. Rehearsal starts from waypoints 08,
14 and 18 produced no training stars in this block (0 of 44), while waypoints
29-40 stayed between 36% and 76%.

Waypoint-02 route search and S5h (November 4, 2026): chained route-aware
searches from `human-waypoint-02.state`, prefixed by the 36-decision
waypoint-02 goal segment, repeatedly stalled near route waypoint 28 with every
node idle-cut; a chain that resumed from an already idle node also wasted
rounds before a no-progress stop was added. Because the block-13 policy already
reaches the star from waypoint 04 in 16 of 29 greedy probes while its best
waypoint-02 route fraction was 0.08, S5h uses a verified partial sequence
instead: 164 search decisions from waypoint 02 to route segment 26, past
waypoints 04 and 08. Under the S5g training configuration it ran without death
or truncation and committed segment 24. It does not reach the star and is an
assisted control fixture, not a learned completion. Replay signature unchanged.

S5h block 14 (November 4, 2026) resumed the block-13 checkpoint for one hour
(wall-clock stop, audit passed): 1,196,580 real decisions and 12,048 completed
attempts in total (111,162 new decisions). The frontier stayed at waypoint 02:
none of 453 training attempts or 50 greedy probes from it reached the star, but
the best waypoint-02 route fraction rose from 0.08 to 0.55, past waypoints 04
and 08. Later anchors still produced training stars (waypoint 38: 11 of 17,
waypoint 29: 6 of 13), except waypoint 23 (0 of 20).

S5h block 15 (November 4, 2026) resumed the block-14 checkpoint for one hour
(wall-clock stop, audit passed): 1,309,356 real decisions and 12,872 completed
attempts in total (112,776 new decisions). None of 387 training attempts or 42
greedy probes from waypoint 02 reached the star. Their furthest positions show
the policy now crosses the waypoint 02-04 section on its own: 234 of 667
waypoint-02 episodes in this block's episode files reached the waypoint 24-27
area, where 221 fell to their death at full health. (Correction: those counts
include demonstration episodes, because the episode files label them
`curriculum`; `attempts.jsonl` holds the true kind. Excluding demonstrations,
99 of 429 attempts and probes reached that area, with all 42 probes among them.) That is the line taken
by the partial demonstration, and every waypoint-02 search from it also stalled
near waypoint 28, while waypoint-04 starts in block 13 continued past it. S5i
therefore truncates the waypoint-02 demonstration to its first 115 decisions
(route segment 15, past waypoint 14; checked under S5h without death or
truncation) so that handover happens before that line diverges. Replay
signature unchanged.

S5i block 16 (November 4, 2026) resumed the block-15 checkpoint for one hour
(wall-clock stop, audit passed): 1,422,216 real decisions and 13,662 completed
attempts in total (112,860 new decisions). None of 375 training attempts or 41
probes from waypoint 02 reached the star, and the truncated demonstration did
not change where they fail. Action traces explain it: waypoint-02 policy
episodes reach waypoint 26, then press nothing or backward instead of the
crouch-jump that crosses to waypoint 27, wander back and fall near waypoint 24.
Passing episodes from later anchors perform that long jump at waypoint 26. The
S5h partial demonstration stopped at segment 26, just before the jump. S5j
therefore extends the waypoint-02 demonstration to 173 decisions (route-search
round b2, depth 2), which includes the long jump and committed segment 27 under
S5i without death. Replay signature unchanged.

S5j block 17 (November 4, 2026) resumed the block-16 checkpoint for one hour
(wall-clock stop, audit passed): 1,524,414 real decisions and 14,381 completed
attempts in total (102,198 new decisions). No waypoint-02 attempt reached the
star, but the extended demonstration changed the failure point: waypoint-02
policy episodes now perform the waypoint 26-27 long jump and reach waypoint 27,
where none did in block 16. They then miss the next crouch-jump toward waypoint
28, which episodes from waypoints 08 and 14 perform before chaining further
jumps to waypoint 30.

S5k replaces the waypoint-02 demonstration with 209 decisions (route-search round
b2, depth 17), through the following crouch-jumps to the waypoint-28 area.
Under S5j it ran without death; its route credit stays at segment 27 with 432
idle frames at the end, below the 1,440-frame cut. Replay signature unchanged.

S5k block 18 (November 4, 2026) resumed the block-17 checkpoint for one hour
(wall-clock stop, audit passed): 1,634,871 real decisions and 15,190 completed
attempts in total (110,457 new decisions). No waypoint-02 attempt reached the
star (0 of 387 training attempts and 42 probes), but the furthest waypoint
reached again moved forward. Mapping each waypoint-02 training attempt and probe
to its nearest route waypoint, excluding demonstration episodes, the furthest
waypoint was 26 in block 16, 27 in block 17 (14 of 389 episodes) and 28-29 in
block 18 (34 of 429 episodes; 71 reached at least 27). Greedy probe fractions
rose from 0.51 to 0.56 during the block. The failures now occur after waypoint 29,
where the policy spins or jumps without a crouch instead of the long jumps that
episodes from waypoint 08 chain toward waypoint 34. Rehearsal starts from
waypoints 08 and 14 reached no star in this block (0 of 10 and 0 of 9).

A route-aware search from waypoint 02 (round c, November 4, 2026) was seeded
with the first 215 decisions of the furthest block-18 episode, which landed at
waypoint 30. That episode was a demonstration episode (204 scripted decisions,
then policy control), not a pure policy attempt as the S5l source string says. Its first 45-minute round timed out; the second, resumed
two depths back, reached the RAM star in 358 decisions without death, confirmed
through the training environment under S5k. S5l uses this sequence as the
waypoint-02 demonstration, so every anchor now has a continuous demonstration to
the star. The sequence is an assisted control fixture, not a learned completion.
Replay signature unchanged.

S5l block 19 (November 4, 2026) resumed the block-18 checkpoint for one hour
(wall-clock stop, audit passed): 1,725,186 real decisions and 15,753 completed
attempts in total (90,315 new decisions). Greedy frontier probes from waypoint 02
reached the RAM star for the first time: 11 of 30, including the last 9 in a row,
each in 356 decisions, so they stay close to the 358-decision demonstration. Training attempts
from waypoint 02, which use epsilon 0.02, reached it once in 278, so the frontier
did not move: the unchanged rule also needs at least 3 stars in the last 30
training attempts. An attempt of about 356 decisions at epsilon 0.02 takes no
random action with probability of only about 0.08%, so meeting that rule requires
the policy to recover from perturbations. Rehearsal starts from waypoints 04, 08,
14 and 18 reached no star in this block (0 of 24 attempts).

S5l block 20 (November 5, 2026) resumed the block-19 checkpoint for one hour
(wall-clock stop, audit passed): 1,825,122 real decisions and 16,324 completed
attempts in total (99,936 new decisions). Greedy frontier probes from waypoint
02 reached the RAM star in all 31 of 31 starts, so the policy executes the
demonstrated line reliably when it takes no random action. Training attempts
from waypoint 02 (epsilon 0.02) reached it in 1 of 281, so the frontier stays at
waypoint 02. Where those attempts end
(nearest route waypoint, demonstrations excluded): 29 at waypoint 1, 27 at 2, 24
at 3, 24 at 10, 23 at 9, and only 7 at 27. They mostly fall in the first ten
waypoints, which means one random action in the opening section is enough to
break the policy; its robustness to perturbation, not its knowledge of the line,
is what the unchanged rule is measuring. Frozen greedy evaluation of the
block-19 checkpoint from waypoint 02 reached the star in 3 of 3 seeds (356
decisions); from waypoint 01 and from the canonical start it died after 22 and
39 decisions. These are assisted-anchor results, not full-start completions.

S5l block 21 (November 5, 2026) resumed the block-20 checkpoint for one hour
(wall-clock stop, audit passed): 1,930,773 real decisions and 16,941 completed
attempts in total (105,651 new decisions). Greedy frontier probes from waypoint
02 reached the RAM star in 32 of 32 starts. Training attempts from waypoint 02
(epsilon 0.02) reached it in 4 of 292, with the last 30 holding one star, so the
frontier rule (at least 3 of 30) is still unmet and the frontier stays at
waypoint 02. As in block 20, most attempts end in the first ten waypoints (47 at
waypoint 1, 29 at 2, 27 at 9, 24 at 3, 23 at 4, 20 at 10). Two blocks with
reliable greedy probes and a training star rate near 1% indicate that the policy
needs recovery from perturbation in the opening section rather than more
repetition of the same line.

Recovery demonstrations (November 5, 2026). To teach recovery from a deviation in
the waypoint-02 opening, the 358-decision sequence was perturbed at decisions 4
(right), 20 (crouch) and 55 (spin), and the route-aware search was asked to find
a way to the star from each perturbed prefix (up to three 45-minute rounds each,
then resumed from the last depth without backing off). None reached the star.
The decision-20 search ended with every beam member dead in all three rounds; the
decision-4 and decision-55 searches reached route segments 20 and 24 and then
died when resumed. No recovery sequence was confirmed, so none is used. The
code now accepts `recovery_sequences` per anchor (commit 91d87d0), unused so far.
A dead search is not proof that no recovery exists, and the opening's narrow
platforms are only a hypothesis for why single deviations are costly.

S5m (`config/anchor-diagnostic-s5m-20261105.yaml`) differs from S5l only in a
collection epsilon of 0.005 instead of 0.02, with the owner's agreement. The first
attempt put the value in `dqn.epsilon`; resume refused it because the checkpoint
manifest signs the whole `dqn` block, so no block started. The value now lives in
a top-level `exploration.epsilon` that only overrides acting during collection
(status.json reports it); the signed agent settings, the replay signature and the
resume path stay as in S5l. The chance
of a 350-decision episode without a random action rises from about 0.08% to about
17%. The frontier rule (two probe stars and at least 3 of 30 training stars),
the demonstrations and the replay signature are unchanged. Training stars under
S5m are not comparable with S5l's: they measure the same policy under less
noise.

### S5m block 22 result

Block 22 (`dqn-20261105-225840-701415`, 87,822 real decisions, 1,235 attempts, audit total 18,176, wall-clock stop at 3,743 s) passed the automatic audit. With the
collection epsilon at 0.005 the waypoint-02 frontier was cleared: 6 of 21 training
attempts from waypoint 02 reached the RAM star (blocks 19-21 had 1, 1 and 4 in about
280 each), two consecutive greedy probes passed and the frontier moved to
waypoint 01. Attempts at waypoint 01 are 0 of 584 so far, all ending in the first
steps (opening advance about 0.042): the waypoint-01 demonstration is still the
10-action prefix to waypoint 02, not the full star sequence, so the policy has no
continuous example from 01. The next config extends it to the verified 02 sequence.
No full-start star was reached (`full_start_stars_this_run` is 0); demonstration
stars in this block are scripted and are not learned completions.

### Algorithm benchmark configs (not yet run)

`config/algorithm-benchmark-f-iqn-20261106.yaml` (control F) and
`config/algorithm-benchmark-btr-20261106.yaml` (BTR-style) train from scratch on
Sweet Sweet from the canonical start: no transfer, no demonstration, no curriculum,
same seed (142), reward, three workers and budget. The BTR arm changes only the
learning package: Munchausen (alpha 0.9, tau 0.03, floor -1), soft target updates
(0.005), AdamW (weight decay 0.01), learning rate 1e-4 and no epsilon, relying on the
noisy layers that both arms already have. Gamma stays 0.997 in both because it must
match the regional reward's per-frame discount. Not included: spectral norm, adaptive
max-pooling, vectorised replay ratio and the paper's 64 environments. A test pins
that the arms differ only in those keys and in their run paths. Nothing has been run.

The owner set the benchmark budget: one seed per arm and 1,000,000 real decisions
each, even if the stage is not completed. The arms run one after another in
one-hour blocks (about 12 per arm at roughly 23 decisions per second), each block
resumed from the previous checkpoint and passed through `audit_block` before the
next starts; the chain stops on any failed audit. A single seed cannot separate a
real difference from run-to-run noise, so a gap between arms will be reported as
indicative only.

### S5n block 23 result

Block 23 (`dqn-20261106-105109-064367`, 77,580 real decisions, audit total 19,157
attempts, one earlier launch of the same block was discarded because two copies
started together and stopped on memory pressure after about 190 decisions) passed the
audit. Waypoint 02 is healthy under S5m's collection epsilon: 7 stars in 17
training attempts and 9 of 11 demonstrations. Waypoint 01 is not: 0 of 458 training
attempts, 0 of 50 greedy probes and 0 of 263 demonstrations reached the star. About
94% of the demonstrations ended at decision 65, a deterministic death: the 10-action
prefix does not land on the exact saved waypoint-02 state, so the 358-decision
waypoint-02 sequence that wins from the saved state dies at its decision 55 when
replayed from waypoint 01. The concatenated waypoint-01 demonstration is therefore
not valid and no learning from it was credited. A route-aware beam search from
waypoint 01, seeded with the first 50 actions of that sequence, is running to find a
continuation that reaches the star.

### Planned algorithm arms and what the official BTR does

Planned arms for the one-seed, 1M-decision algorithm group, after the speedup work:
control F (IQN), BTR-style, BBF-lite (higher replay ratio, periodic resets,
annealed n-step and gamma, weight decay and a wider encoder, no self-predictive loss;
details to be checked against the paper before implementing), the full BBF only if
BBF-lite shows a signal, and the Mario 64 project's BTR parameters if still wanted.

The official BTR (VIPTankz/BTR, commit 093ca77, `main.py`, `Agent.py`, `networks.py`,
`PER.py`) differs from `src/btr.py` in the sibling Mario 64 project and from the
BTR-style arm configured here: the encoder is IMPALA with width 2x, spectral norm on
every convolution and adaptive max-pooling to 6x6; the head uses 8 quantiles, dueling
and noisy layers with 512 hidden units; Munchausen uses alpha 0.9, tau 0.03 and
floor -1, with the log-policy bonus taken from the online network; gamma is 0.997,
n-step 3, learning rate 1e-4, Adam by default (AdamW with decay 1e-4 optional), batch
256, replay ratio 1 with 64 environments, PER alpha 0.2, and a **hard** target copy
every 500 gradient steps by default (a soft update with tau 2.5e-4 is optional).
Epsilon-greedy still runs on top of the noisy layers early in training, rewards are
clipped to [-1, 1], and gradient norms are clipped at 10. The BTR-style arm here
copies the Mario 64 project's soft update (0.005), AdamW decay 0.01 and no epsilon,
so it is not the official default. Not yet read: the preprocessing wrapper and the
Huber-loss body.

The BTR arm now follows the official defaults where the pipeline allows: spectral
norm on the encoder convolutions (new `spectral_norm` option; adaptive max-pooling and
the 32/64/64 widths already existed), 8 quantiles, Munchausen 0.9/0.03/-1, hard target
copy every 500 updates, plain Adam, learning rate 1e-4, PER alpha 0.2, gamma 0.997 and
an epsilon schedule of 1.0 to 0.01 over 40,000 decisions (4% of the budget, the
official 2M of 50M scaled). Kept equal to arm F so the arms differ only by the
algorithm package: batch 32, one update per 4 decisions (official: batch 256 at
replay ratio 1 with 64 environments), replay of 250,000 and no reward clipping.
