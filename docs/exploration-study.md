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
BBF-lite shows a signal, and the alternative BTR ([PedroM2626/mario64ds-rl](https://github.com/PedroM2626/mario64ds-rl), see [related projects](related-projects.md)) with its own parameters if still wanted.

The official BTR (VIPTankz/BTR, commit 093ca77, `main.py`, `Agent.py`, `networks.py`,
`PER.py`) differs from the alternative BTR (`src/btr.py` in [PedroM2626/mario64ds-rl](https://github.com/PedroM2626/mario64ds-rl)) and from the
BTR-style arm configured here: the encoder is IMPALA with width 2x, spectral norm on
every convolution and adaptive max-pooling to 6x6; the head uses 8 quantiles, dueling
and noisy layers with 512 hidden units; Munchausen uses alpha 0.9, tau 0.03 and
floor -1, with the log-policy bonus taken from the online network; gamma is 0.997,
n-step 3, learning rate 1e-4, Adam by default (AdamW with decay 1e-4 optional), batch
256, replay ratio 1 with 64 environments, PER alpha 0.2, and a **hard** target copy
every 500 gradient steps by default (a soft update with tau 2.5e-4 is optional).
Epsilon-greedy still runs on top of the noisy layers early in training, rewards are
clipped to [-1, 1], and gradient norms are clipped at 10. The BTR-style arm here
copied the alternative BTR's soft update (0.005), AdamW decay 0.01 and no epsilon,
so it was not the official default (it has since been aligned, below). Not yet read: the preprocessing wrapper and the
Huber-loss body.

The BTR arm now follows the official defaults where the pipeline allows: spectral
norm on the encoder convolutions (new `spectral_norm` option; adaptive max-pooling and
the 32/64/64 widths already existed), 8 quantiles, Munchausen 0.9/0.03/-1, hard target
copy every 500 updates, plain Adam, learning rate 1e-4, PER alpha 0.2, gamma 0.997 and
an epsilon schedule of 1.0 to 0.01 over 40,000 decisions (4% of the budget, the
official 2M of 50M scaled). Kept equal to arm F so the arms differ only by the
algorithm package: batch 32, one update per 4 decisions (official: batch 256 at
replay ratio 1 with 64 environments), replay of 250,000 and no reward clipping.

### Waypoint-01 demonstration search

The concatenated waypoint-01 demonstration of S5n died at decision 65 in 94% of its
episodes, so a route-aware beam search was run from the waypoint-01 savestate
(`.local/anchor01-route-search-c1..c10-20261106`, macros on, scored against the
human route), seeded with the first 50 actions of that concatenation. Rounds 1 to 5
(beam 32, 45 minutes each) extended the confirmed prefix from 50 to 210 actions; round
6 started from a doomed state and died at once. Rounds 7 to 10 restarted from the
round-5 prefix with beam 64, and round 10 found a RAM star. The winning sequence has
350 actions from waypoint 01 and was replayed in the training environment: complete,
not dead, reward 36.35, final position at the star. It is a search fixture, not a
learned completion. `config/anchor-diagnostic-s5o-20261106.yaml` equals S5n except
that this sequence is the waypoint-01 demonstration. The search savestates were
deleted after the run; the reports remain local.

### S5o block 24 result

Block 24 (`dqn-20261106-181449-218839`, 66,816 real decisions, audit total 19,640
attempts) stopped on memory pressure after 2,769 s with a complete checkpoint and
passed the audit as a resource-stopped block. At waypoint 01 the searched
demonstration reaches the star in 32 of 127 scripted episodes (the rest end after the
handover and are not counted as learned), training attempts reach it 6 times in 230,
and the last three greedy frontier probes were stars. The hybrid frontier rule was
satisfied (two probe stars, 6 of the last 30 training attempts), but waypoint 01 is
the last anchor, so the frontier stays there. These are assisted starts, not full-start
completions.

A frozen greedy evaluation of this checkpoint from the canonical start
(`record-dqn`, seeds 1 to 3, `eval-dqn-video-20261106-190200-346012`) died in all three
episodes after 27 decisions, with no full-start star. The training start selector only
ever draws curriculum anchors (`env.py` takes the canonical start only when no anchor
is chosen), so the policy has never trained or been shown a demonstration from the
canonical start, which is about ten decisions before waypoint 01.

### Canonical-start search and S5p

Correction: the 10-action sequence stored for waypoint 01 leads from waypoint 01 to
waypoint 02; it is not the segment from the canonical start to waypoint 01, which the
human recording places at about 20 decisions. The earlier statement that the
canonical start is "about ten decisions before waypoint 01" was wrong, and a replay of
those 10 actions plus the 350-action sequence from the canonical start (which died at
decision 24) was not a valid test.

A beam search from the canonical start slot (`--full-start`, route-aware, beam 64;
`.local/full-start-route-search-*`) confirmed a prefix of 53 actions that stays alive
and reaches z = -7035 (waypoint 01 is at z = -1022, waypoint 04 at z = -4933), but every
extension from there died, and splicing the waypoint-04 demonstration after prefixes of
26 to 33 actions died in all eight replays: open-loop sequences do not transfer between
states that differ by a few frames. S5p therefore does not use a full open-loop
demonstration from the canonical start. It gives the canonical start only the first 20
confirmed actions (about waypoint 02) as a `full_start` demonstration with a uniform
prefix handover, so the policy takes over from states along that segment, and it lowers
`curriculum.probability` from 1.0 to 0.75 so a quarter of the starts use the canonical
start. The dqn block, the frontier rule and the replay signature are unchanged, so the
block resumes from the S5o checkpoint. Full-start stars of the policy are counted as
full-start; scripted full-start episodes are counted as demonstrations.

S5p's first launch failed before creating a run: the replay signature includes the
whole `curriculum` block (`curriculum_distribution`), so changing `curriculum.probability`
broke the resume check. The canonical-start share now lives in a top-level
`start_mix.full_start_fraction` (0.25) that only affects which start a reset draws, like
`exploration.epsilon`; `curriculum.probability` stays 1.0, and the S5p signature and
dqn block equal the checkpoint's. The helper draws no random number when the fraction
is zero, so earlier runs keep their random streams.

### S5p block 25 result

Block 25 (`dqn-20261107-010030-317498`, 115,506 real decisions, audit total 20,240
attempts, wall-clock stop) passed the audit. Waypoint 01 is now learned in the
assisted setting: 22 of the last 24 greedy frontier probes reached the RAM star, 58 of
218 training attempts did, and 81 of 122 scripted waypoint-01 episodes did (scripted
outcomes are not learning). The canonical-start share produced 110 policy episodes
from the full start: 0 stars (84 deaths, 26 timeouts, median 74 decisions, longest 141),
plus the 20-action handover demonstrations counted separately.

Frozen greedy evaluation from the canonical start (`record-dqn`, seeds 1 to 3,
`eval-dqn-video-20261107-020239-302144`): all three episodes are identical and die at
decision 95 (before this block: decision 27). The action trace shows the policy follows
the opening to about waypoint 02 (route progress 3.5% at decision 19), then loses it and
walks backward with the back label for about 60 decisions before falling; best route
progress stays at 3.9%. The canonical start is therefore still not solved, but the
opening handled by the handover demonstration is now followed, and the failure moved
from decision 27 to a mid-air state after the first long jumps that the anchor-based
training rarely visits. No full-start star has been reached by the policy.

### S5p block 26 result and S5q

Block 26 (`dqn-20261107-020836-322961`, 116,340 real decisions, audit total 20,807
attempts, wall-clock stop) passed the audit. All 23 greedy frontier probes at waypoint
01 reached the star. The 84 policy episodes from the canonical start reached no star
(median 96 decisions, longest 153). The frozen greedy evaluation from the canonical start
(`eval-dqn-video-20261107-031052-272053`, seeds 1 to 3, identical) no longer dies: it
times out at 141 decisions with route progress 4.0%. It passes z = -2869 around decision
20, past waypoint 02 (z = -2608), then walks backward with the back label to about the
start and wanders. The handover prefix leaves the policy about 36 units to the side of
the human line (x about -66 against about -30 at waypoint 02) in the air; the anchor
training starts grounded on the human line, so these states are rarely seen.

S5q (`config/anchor-diagnostic-s5q-20261107.yaml`) changes only collection and
demonstration settings: the `full_start` demonstration uses the first 44 confirmed
actions (to about waypoint 05, z = -5600) instead of 20, so uniform-prefix handovers cover
the whole off-line stretch the policy fails on, and `start_mix.full_start_fraction`
rises from 0.25 to 0.5 because waypoint 01 is saturated in its probes. The signature and
the dqn block are unchanged. It is a hypothesis test: if the canonical start still stays
at about 4% route progress after this block, the cause is not the coverage of that
stretch.

### S5q block 27 result

Block 27 (`dqn-20261107-031726-171496`, 115,407 real decisions, audit total 21,530
attempts, wall-clock stop) passed the audit. All 19 greedy frontier probes at waypoint
01 reached the star. The 224 policy episodes from the canonical start (half of the starts)
reached no star; their median length was 63 decisions, the longest 191, the median opening
advance 6.8% and the best 19.6%. The frozen greedy evaluation from the canonical start
(`eval-dqn-video-20261107-041941-613113`, seeds 1 to 3, identical) follows the line to
z = -5619 by decision 50 (past waypoint 04 at z = -4933), turns back, and dies at decision
80 with route progress 7.7% (4.0% in block 26). The hypothesis that the failure came from
the missing coverage of the 02-to-05 stretch is supported: the stretch is now crossed. The
canonical start is still not solved and the next failure lies beyond the end of the
44-action handover prefix.

### S5q block 28 result and a policy-seeded canonical-start search

Block 28 (`dqn-20261107-042601-744945`, 115,782 real decisions, audit total 22,280
attempts) passed the audit. 17 of 19 greedy waypoint-01 probes reached the star. The 255 canonical-start policy episodes reached no star; the median opening
advance rose from 6.8% to 12.9% (best 19.6%, median length 74 decisions, longest 208).
The frozen greedy evaluation from the canonical start (`eval-dqn-video-20261107-052813-937585`)
is unchanged in shape: it follows the demonstration line to z = -5619 at decision 50, turns
back and dies at decision 78 with route progress 7.7%. The greedy policy therefore repeats the
44-action handover prefix and fails right after its end, so the next lever is a longer
confirmed prefix, not more of the same training.

A beam search from the canonical start seeded with the policy's own first 50 actions (alive
at z = -5619; the same technique that produced the waypoint-02 demonstration) is running
(`.local/full-start-route-search-s1..s6-20261106`, beam 64, route-aware, 45 minutes per
round). Training is paused while it uses the emulator.

### Canonical-start demonstration found

The policy-seeded canonical-start search (`.local/full-start-route-search-s1..s8-20261106`,
beam 64, route-aware, 45 minutes per round) extended the confirmed prefix from 50 actions
(round 1) through 244 (round 7) and found a RAM star in round 8. The winning sequence has
312 actions from the canonical start slot. It was replayed through the training environment
path with `start_kind` `full_start`: complete, not dead, reward 36.80, final position at the
star (`env_confirmation` in the round-8 report). It is a search fixture, not a learned
completion, and it is not a stage-clear by the agent.

`config/anchor-diagnostic-s5r-20261107.yaml` equals S5q except that this sequence is the
`full_start` demonstration (uniform-prefix handover, half of the starts from the canonical
start). The signature and dqn block are unchanged, so it resumes from the block-28
checkpoint. The search savestates were deleted afterwards.

### S5r block 29: frozen greedy policy reaches the RAM star from the canonical start

Block 29 (`dqn-20261107-112836-050758`, 114,975 real decisions, audit total 22,826
attempts, wall-clock stop) passed the audit. During collection (epsilon 0.005) the policy
itself reached the RAM star from the canonical start in 5 of 177 non-scripted full-start
episodes (best opening advance 98.4%, median 18.5%, longest 451 decisions); scripted
demonstration episodes are counted separately and are not learning. Waypoint-01 greedy
probes: 14 of 14.

**Final gate.** The frozen greedy evaluation of this checkpoint from the canonical start
(`record-dqn`, no learning, evaluation epsilon 0, seeds 1 to 3,
`eval-dqn-video-20261107-123101-762968`, with `policy.mp4` and `policy.inputs.jsonl`) reached the RAM
star in all three episodes in 311 decisions, `full_start_ram_star` true, route projection
99.5%. The three runs are identical because the emulator and the greedy policy are
deterministic, so they are one trajectory rather than three independent successes.

What this does and does not show:

- The greedy policy reproduces the searched demonstration: 308 of its 311 actions equal the
  312-action sequence found by beam search from the canonical start (first difference at
  action 307). It is a trained policy that completes the stage from the canonical start, but
  its route was found by search and taught through handover demonstrations; it did not
  discover the route by itself.
- It is brittle. A stochastic frozen evaluation with epsilon 0.005 (seeds 11 to 16,
  `eval-dqn-video-20261107-123243-555587`) reached the star in 3 of 6 episodes: exactly the
  three with no random decision. Episodes with one or two random actions died or timed out.
- One start state, one game build and one training seed. This is not generalisation.

The stage-completion gate named for Sweet Sweet is met under these limits. Checkpoints,
recordings and savestates remain local and are not committed.

### Speedup probes (November 7)

Profile of the S5r block 29 (3,705 s): emulator collection 70% (2,607 s), learner updates
23% (834 s), policy inference 3%, checkpoint 3%; 31 decisions per second, collection and
learning in series. Five 10-minute probes from scratch with the arm-F algorithm settings
(`config/speedup-probe-*-20261107.yaml`, one run each, warm-up included) gave:

| probe | workers | overlap_learning | decisions per second | updates per decision |
| --- | --- | --- | --- | --- |
| b3 | 3 | no | 33.4 | 0.225 |
| o3 | 3 | yes | 34.8 | 0.226 |
| o4 | 4 | yes | 41.0 | 0.229 |
| o5 | 5 | yes | 43.0 | 0.230 |
| o6 | 6 | yes | 42.6 | 0.229 |

Overlap alone gains little at three workers (+4%): the learner's update time rises from
about 20 ms to 45 ms per update once emulators compete for the CPU, so the two do not
hide each other. More workers help up to five (+29% over b3) and six adds nothing, so the
CPU, not the worker count, is the limit there. The update-to-decision ratio is unchanged
by overlap. One run per setting; the differences between o4, o5 and o6 are within
what a second run could change. Further renderer and capture options were already studied
in the optimisation notes and are not repeated here. The two algorithm-benchmark arms now
use five workers and `overlap_learning: true` (for both arms, so they stay matched), which
would take roughly 9.5 hours per arm at 1,000,000 decisions instead of 12.

The two algorithm-benchmark arms were then set to four workers instead of five (still with
`overlap_learning: true`; 41.0 against 43.0 decisions per second in the probes, so about
+23% over three workers) because the machine's free Windows commit memory stayed near 10 GB
and a five-worker run did not fit. The launch script's own start threshold was lowered from
22 GB to 18 GB of free commit with the owner's agreement; the training code's internal guards
(6 GB commit, 2 GB physical, and a clean stop with a checkpoint on memory pressure) are unchanged.

### Waypoint-bonus arm (planned)

Owner request: test the BTR arm with +1 per waypoint, the star kept at 20 and fewer waypoints.
`config/algorithm-benchmark-btr-waypoint-20261107.yaml` equals the BTR arm except for the reward:
mode `discounted_waypoint_bonus` (new, `RegionalWaypointBonus`) pays `waypoint_bonus` (1.0) for each
newly committed route waypoint, never takes credit back, and the objective validation requires the
total bonus (here 15) to stay below the star (20). The owner then set death to -3 and damage to -1 for this arm (the BTR arm keeps -2 and -0.25), so the arm differs from BTR in the waypoint reward and in those two weights;
timeout stays 0. The route is `config/routes/smg1-human-landing-v2-sparse15.json`, made by
`scripts/make_sparse_route.py`: 16 of the 47 human landing points (kept source indices
0, 2, 4, 7, 9, 11, 13, 20, 23, 26, 29, 31, 36, 39, 41, 46), chosen for even cumulative length. Nothing has
been trained with it yet.

A waypoint is committed only after a grounded landing within the landing radius, in order (a swept
later waypoint can skip up to five). Replaying the frozen greedy trajectory that reaches the star
(311 decisions, offline, no emulator) through the routes gives:

| route | waypoints | commit events | waypoints committed before the star | longest stretch without credit |
| --- | --- | --- | --- | --- |
| current (47 points) | 46 | 19 | 45 | not computed |
| sparse | 12 | 0 | 0 | 311 decisions |
| sparse | 13 | 4 | 12 | 136 |
| sparse | 15 | 6 | 14 | 101 |
| sparse | 16 | 6 | 15 | 72 |
| sparse | 17 | 2 | 3 | 251 |
| sparse | 18 | 7 | 17 | 63 |
| sparse | 19 | 2 | 4 | 239 |
| sparse | 20 | 9 | 19 | 64 |

Two cautions follow. Credit arrives in bursts (a landing can commit several waypoints at once), so
even the current route pays on 19 of 311 decisions, not on most of them. And credit is fragile to the
choice of waypoints: with 12, 17 or 19 waypoints that winning trajectory almost never commits
anything, because it lands more than the landing radius from one waypoint and the ordered credit stops
there. The 15-waypoint route works on this trajectory, which is the one that reproduces the search
demonstration, so it says little about how another policy would land. The last waypoint (the star
region) is never committed because the episode ends first, so at most 14 of the 15 are paid.

### Arm F closed at 462,132 decisions

Control arm F (IQN, from scratch, no demonstration, no curriculum, four workers with overlap)
ran three one-hour blocks (`runs/model-free/algorithm-benchmark-20261106/F`, all audited) and
the owner closed it there, short of the 1,000,000-decision budget. Per block: 153,624, 314,744 and
462,132 real decisions; 0 RAM stars in all of them; best route advance 2.6% in block 1 and 0.0% in
blocks 2 and 3 (no route waypoint confirmed in 1,711 attempts); deaths 221, 1 and 6. In blocks 2 and
3 almost every attempt ended by the idle cut at 180 decisions, the policy drifting backward and
using spin, crouch, jump and no-op rather than the forward label. One seed, one reward, no
demonstration: it says this control did not find the opening in 462,132 decisions, not that it
cannot. The checkpoints are kept and the chain state allows resuming it.

### BTR arm block 1

BTR arm (official-default settings, from scratch, four workers with overlap), block 1
(`dqn-20261107-204734-238052`, 142,288 real decisions, 888 attempts, audited): 0 RAM stars, 197
deaths, best route advance 2.6% (the same value as the F arm's first block) and a mean advance of
0 over the last 100 attempts. By the end the policy is close to a single action: the last
episodes use spin in 174 of 181 and 109 of 111 decisions and time out at 180 decisions. This is
the same passive failure as the F arm showed from its second block, reached here within one hour.

### Probe without failure penalties (planned)

Owner suggestion: if the arms share the passive failure, remove the death and damage penalties.
`config/algorithm-benchmark-f-nopenalty-20261107.yaml` is arm F with death 0 and damage 0 (instead
of -2 and -0.25) and nothing else changed (a test pins that). The discounted-objective validation
used to require a negative death weight; it now accepts zero only when the config sets
`regional_reward.allow_unpenalized_failure: true`, and still rejects positive failure pay, a step cost
or a negative timeout. With no penalty, dying costs only the discounted future reward it forfeits, so
trying a risky first jump no longer costs more than idling; whether that is enough to leave the
passive policy is the question the probe tests. Planned as a two-block probe from scratch before
re-running any other arm without penalties, because the reward is a controlled variable and changing
it for every arm would repeat their costs. Not run yet.

### BTR arm block 2 and closing

Block 2 of the BTR arm (`dqn-20261107-215344-257267`, 290,608 real decisions in total, 824 attempts,
audited): 0 RAM stars, 0 deaths, best route advance 0.0%. Every sampled episode is the same: spin in
177 of 181, 179 of 181 and 51 of 51 decisions, standing at z = -50 (the start) until the idle cut at
180 decisions. The policy has collapsed to a single action that never moves. With the F arm's three
blocks this is two arms, two algorithms and one seed each showing the same passive collapse under
the current reward, so the BTR arm was stopped at 290,608 decisions (checkpoints kept, chain state
allows resuming) and the no-failure-penalty probe on arm F is next.

### Probe without failure penalties: block 1

Arm F with death 0 and damage 0 (`dqn-20261107-225955-498040`, 149,852 real decisions, 1,240
attempts, audited, one block): 0 RAM stars, 792 deaths, best route advance 2.6% (the same early value
as arms F and BTR, not repeated), and a mean advance of 0 over every 100-attempt window sampled
(attempts 0 to 100, about 570 to 670 and the last 100). The median attempt length rose from 80 to 116
to 180 decisions and deaths fell from 91 to 77 to 0 per 100, so by the end the policy again idles to the
cut (spin and no-op). Removing the failure penalties therefore did not prevent the passive policy in
this block. A likely reading is that no waypoint payment is ever seen, so all values stay near zero and
the greedy action is arbitrary and constant, which would make the lack of any reward signal in the
opening, not the penalties, the cause. This is one block and one seed; it was not run for a second
block. Related check: the sparse 15-waypoint route's first kept waypoint is about 2,560 units from the
start (the first segments of the 47-point route are about 980 and 1,590 units), so a from-scratch
agent would reach its first payment later, not sooner.

### Position-novelty exploration arm and a dense-start waypoint route

Both arms are built and tested; neither has been trained.

**Exploration arm** (`config/algorithm-benchmark-exploration-20261107.yaml`, new `position_novelty` reward).
From scratch, no route reward, no anchor, no demonstration, no checkpoint seeding. The reward is RAM position
novelty plus the RAM star: 0.05 divided by the square root of the cell's visit count, paid once per
episode on entering a cell of 200 x 250 x 200 units that is new to the episode, while alive and above the
fall limit, with the visit counts kept per worker across episodes; star +20, death -2, damage -0.25 as in arm F.
An episode that finds no new cell for 180 decisions is cut as a bootstrap truncation, so standing still
earns nothing and ends the episode. The policy input stays visual. The route is loaded only so the
route advance can be reported as a diagnostic: it never pays and never ends an episode. A first
version paid on every step in a cell, which made standing still a growing income; the test that caught it
now pins the once-per-cell rule. The environment refuses to combine this reward with regional or route
rewards, a curriculum or demonstrations; the replay signature includes it; the block audit checks that
novelty is paid only within its configured bonus. The counts are per worker (four processes), so the
cross-episode decay is not shared between workers, and they restart with each block.

**Waypoint arm route.** `make_sparse_route.py` takes an optional head: the first `head` waypoints are the
source's own, the rest are spread by length. The waypoint arm now uses `smg1-human-landing-v2-sparse15h8.json`
(source indices 0 to 8, 11, 17, 24, 28, 34, 40, 46; first waypoint 973 units away instead of 2,559). Replaying
the winning greedy trajectory offline over 15-waypoint routes with other heads gave between 1 and 7 commit
events and between 2 and 14 waypoints paid, so the choice is fragile; head 8 gave 7 events, 13 paid and a
longest stretch of 77 decisions, the best of the 15-waypoint variants tried. Even with the nearer first
waypoint, that trajectory's first payment arrives at decision 21 because credit needs a grounded landing
and the first landing is there. Heads of 3 to 7 with 17 or 19 waypoints paid more often on that trajectory
(for example 17 with head 5: 8 events, 16 paid, longest stretch 60), but exceed the 15 the owner asked for.

### Does the Sweet Sweet policy generalise or follow a memorised route?

Frozen stochastic evaluation of the block-29 checkpoint from the canonical start (`record-dqn --stochastic`,
evaluation epsilon 0.01, seeds 101 to 130, `eval-dqn-video-20261108-002506-973818`; 30 episodes, one
checkpoint, one start state). Success by the number of random actions an episode happened to contain:

| random actions in the episode | stars | episodes |
| --- | --- | --- |
| 0 | 3 | 3 |
| 1 | 2 | 16 |
| 2 | 1 | 6 |
| 3 or more | 0 | 5 |
| all | 6 | 30 |

A single random action out of about 311 decisions turns a certain success into roughly one in eight, and three
or more were never survived. Together with the earlier result that 308 of the 311 greedy actions equal the
searched 312-action sequence, this is the pattern of a policy that reproduces one trajectory, not of one that
has learned to play the stage: it has not learned to recover from small deviations. The action sequence is
not the owner's recording (the recording supplies the anchors and the route used for scoring; the sequence was
found by beam search over the eight actions), and the network does map frames to actions rather than counting
steps, but in a deterministic emulator that distinction does not show up in the greedy result. Not
tested: starts perturbed by random actions before the agent takes over, other savestates, other game builds.

### Exploration arm: block 1 (partial)

Block 1 of the position-novelty arm (`dqn-20261108-002312-149241`) stopped on memory pressure after 1,960 s
and 79,104 real decisions, with a complete checkpoint, and passed the audit as a resource-stopped block. The
stop followed a 30-episode robustness evaluation that I ran at the same time on the same machine, which I
should not have done during a training block. Results so far: 378 attempts, 0 RAM stars, 324 deaths, no route
advance (the diagnostic route), but a median of about 404 decisions per recent attempt, well above the 180
decisions at which an idle policy is cut, so the policy is moving and entering new cells rather than idling.
Whether that reaches the first waypoint is not known yet.

### Exploration arm: block 2

Block 2 (`dqn-20261108-030051-357241`, 236,124 real decisions in total, 137 attempts, audited): 0 RAM stars,
9 deaths, route advance 0.0 in every attempt, a median attempt of about 1,134 decisions (it was 404 in the
partial first block), 43 decisions per second. The policy keeps entering new cells (a median of 47 distinct
cells and up to 101 in an episode among the last 40) but does not go forward: it ranges sideways (x from about
-500 to +650 on the start area), most of those episodes never get below z = -500, and several never leave the
start. Its actions are spread over jump, spin, sideways moves, no-op and crouch. So position novelty with this
cell size and bonus produces long, moving episodes but, so far, no route advance. Continuing one more block
before judging the arm.

### BBF-lite arm (built, not run)

Source: Schwarzer et al., "Bigger, Better, Faster" (arXiv 2305.19452v3). The paper's text states: Impala-CNN
with every layer 4 times wider, AdamW with weight decay 0.1, no noisy layers, periodic resets every 40,000
gradient steps with the convolutional layers moved 50% toward a random target and later layers fully reset,
an n-step horizon decreasing from 10 to 3 and a discount rising from 0.97 to 0.997 over the 10,000 updates after
each reset, an EMA target network, a replay ratio of 8 (2 is also reported) and a self-predictive (SPR) loss.
The text and appendix do not give the batch size, learning rate, EMA coefficient or epsilon schedule, and the
component-removal numbers are only in figures I could not read; those values here are mine.

`config/algorithm-benchmark-bbf-lite-20261108.yaml` is arm F with only the learner changed: encoder base
channels 64 (twice the F arm's, the 4x width of the standard 16/32/32 base), AdamW weight decay 0.1, EMA target
0.005 (my choice), plain linear heads (new `noisy: false`) with epsilon-greedy, and a reset every 40,000 updates
(new `reset: {interval, shrink}`; encoder shrink and perturb at 0.5, later layers re-initialised, optimizer state
cleared, target copied). A test pins that nothing else differs from arm F. Not included, and why: the replay ratio
(one update per four decisions stays, because at the measured 29 to 45 ms per update a ratio near BBF's would
leave the learner far slower than the emulators), the n-step and discount schedules (the discount must equal
the regional reward's per-frame discount, and the replay fixes its n-step at construction), and the SPR loss.
So this is a BBF-style learner on the F arm's data rate, not BBF. It has not been trained; given that arms F and
BTR collapsed to a passive policy before any learner difference could matter, a from-scratch run of it would
mostly test the same exploration problem.

### Exploration arm: block 3

Block 3 (`dqn-20261108-040745-704522`, 397,864 real decisions in total, 130 attempts, audited): 0 RAM stars,
route advance still 0.0 in every attempt, 46 deaths in the block (9 in block 2), a median attempt of about 1,144
decisions. The change from block 2 is direction: among the last 30 episodes the lowest z reaches -1,286, -1,148
and -1,101, past the first waypoint's z of about -1,022, where in block 2 it was mostly above -500. The agent is now
making forward progress and dying more (falls), but it has not landed within the landing radius of the first
waypoint, so the route diagnostic has not moved. Continuing for a further block.

### Waypoint-bonus fix, and the exploration arm closed at 551k

Before the waypoint arm was run, a check that drove the environment step by step found that its reward class
had no `value` attribute, which the environment reads on every step, so the first training step would have
crashed; the earlier tests built the environment but never stepped it. `RegionalWaypointBonus` now exposes
`value` (bonus times committed waypoints), the block audit's credit limit now includes the new mode, and a test
walks the 15-waypoint route through the real environment and checks 15 points are paid in total and
the logged potential reaches 15.

Exploration arm block 4 (`dqn-20261108-051504-940475`, 551,288 real decisions in total, 124 attempts, audited):
0 RAM stars, route advance 0.0 in every attempt, 53 deaths in the block, a median attempt of about 1,242
decisions. The lowest z among the last 30 episodes is -1,086, no better than block 3's -1,286, so the forward
trend of block 3 did not continue. Four blocks and 551,288 decisions without a single route waypoint, so the arm
was closed here under the same rule as arms F and BTR (checkpoints kept, chain state can resume).

### Waypoint arm block 1, and why the from-scratch arms collapsed

Waypoint arm (BTR learner, +1 per waypoint on the dense-start 15-waypoint route, star 20, death -3, damage -1),
block 1 (`dqn-20261108-062422-342474`, 142,088 real decisions, 895 attempts, audited): 0 RAM stars, 204 deaths,
no waypoint confirmed, and by the end every sampled attempt is the same idle policy (spin in 179 of 181 decisions,
return 0, cut at 180 decisions), as in the BTR arm.

Correcting the diagnosis I gave after the no-penalty probe: comparing the arms' reward configuration with the
Sweet Sweet curriculum runs shows that arms F, BTR, the probe and the waypoint arm (and the exploration arm)
all used `timeout_mode: truncation_bootstrap` with a timeout weight of 0, so a policy that idles until the cut
loses nothing, while the curriculum runs used `idle_terminal_penalty` (an idle cut is a terminal failure of -2,
as bad as a death), `max_idle_frames` 1,440 and a waypoint lookahead of 5. Under the first setting idling is the
costless action and the passive collapse is what a learner should do; the no-penalty probe removed the death cost
as well and left idling free. The statement that the failure came from the lack of any reward signal in the
opening is therefore not supported by those runs: they never tested a reward in which idling costs something.
`config/algorithm-benchmark-f-idle-20261108.yaml` is arm F with only that reward changed (timeout -2 as an idle-cut
terminal, 1,440 idle frames, lookahead 5; a test pins it). The earlier arms' results stand as results of their
reward, not of their algorithms.
