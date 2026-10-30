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

Frozen evaluations after block 2 (`record-dqn --assisted-start --anchor`): at waypoint 40, epsilon 0.1 reached the RAM star in 2/3 episodes (61 and 77 decisions); greedy timed out. At waypoint 38, 0/4. Both blocks passed `audit_block`. Same Wii state and one training seed: not generalization and not full-start success.
