# Center of the Universe: from-scratch training, November 12 to 14, 2026

Two from-scratch arms of the alternative BTR learner on the Galaxy 1 Center of the Universe setup
([the setup](center-of-universe.md)), one seed each, started at the canonical slot 3 start. No demonstration, no
anchor, no curriculum, no human action in any replay. **Neither arm reached the end location (the gate); the best
arm confirms about 26% of the route.** The owner stopped the chain after block 10 of the second arm; nothing is
training now. The per-block numbers are in
[`assets/performance/center-20261114-blocks.json`](assets/performance/center-20261114-blocks.json).

## What was run

| | Plain arm | Novelty arm |
| --- | --- | --- |
| Config | [`center-altbtr-start-20261110.yaml`](../config/center-altbtr-start-20261110.yaml) | [`center-altbtr-novelty-20261113.yaml`](../config/center-altbtr-novelty-20261113.yaml) |
| Learner | alternative BTR (IQN, IMPALA encoder, 32 channels, 8 quantiles, noisy layers, Munchausen, gamma 0.99, learning rate 1e-4, weight decay 0.01) | the same |
| Route reward | 0.1 per newly confirmed route segment (273 segments, 27.3 in total), paid once, never taken back; 50 for the gate; 0 for death, damage and timeout | the same |
| Extra | none | 0.02 for each 200 x 250 x 200 unit cell that is new to the episode, divided by the square root of the cell's visit count |
| Start | slot 3 only | slot 3 only |
| Workers and blocks | 4 native Dolphin workers, blocks of 7,200 s resumed from the previous checkpoint | the same |
| Decisions | 1,104,608 (4 blocks) | 2,570,500 (10 blocks) |

The reward goes into the learner unchanged: there is no reward clipping or sign transform anywhere in the learner or
the replay. The gate is the calibrated end location (300 unit 3D radius, grounded, alive), not a RAM star and not a
Bowser victory. Each 7,200 s block collected 244,000 to 291,000 real decisions (about 34 to 40 per second for four workers).

## Results

"Route" is the fraction of the route confirmed by the attempt (the `opening_advance` field of `attempts.jsonl`), not a
geometric projection. Values are over all attempts of the block; every attempt of every block ended in a death, a fall
or an idle timeout, and **no attempt of either arm reached the gate**. These are the stochastic collection attempts of
the training workers, not frozen greedy evaluations.

Plain arm:

| Block | Decisions (total) | Attempts | Median route | p90 | Best | Median decisions per attempt |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 246,904 | 2,722 | 1.01% | 3.24% | 4.18% | 35 |
| 2 | 526,376 | 1,305 | 3.24% | 3.24% | 3.48% | 227 |
| 3 | 813,312 | 1,273 | 3.24% | 3.24% | 3.56% | 227 |
| 4 | 1,104,608 | 1,290 | 3.24% | 3.24% | 4.11% | 227 |

The third plain block was started once and lost 21 minutes in, without a checkpoint, when the session that had
launched it ended; it was rerun from the block 2 checkpoint (the table lists the rerun). About 5.5 hours of machine
time passed between the loss and its discovery.

Novelty arm:

| Block | Decisions (total) | Attempts | Median route | p90 | Best | Median decisions per attempt |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 251,812 | 2,606 | 1.63% | 4.24% | 10.17% | 64 |
| 2 | 503,932 | 2,235 | 5.26% | 7.65% | 13.09% | 96 |
| 3 | 748,440 | 747 | 12.27% | 13.68% | 20.39% | 297 |
| 4 | 1,015,460 | 834 | 19.94% | 21.89% | 23.91% | 313 |
| 5 | 1,281,388 | 840 | 21.81% | 22.50% | 24.00% | 314 |
| 6 | 1,546,116 | 852 | 22.61% | 24.30% | 25.65% | 306 |
| 7 | 1,808,744 | 834 | 24.34% | 25.53% | 27.34% | 308 |
| 8 | 2,069,212 | 784 | 25.53% | 25.84% | 26.35% | 316 |
| 9 | 2,323,484 | 706 | 25.89% | 26.32% | 32.08% | 355 |
| 10 | 2,570,500 | 666 | 26.13% | 26.42% | 27.88% | 367 |

## Reading

- **The plain arm froze at 3.24% for four blocks (about 850,000 decisions).** A frozen greedy episode shows Mario on a
  stone platform next to a wall with translucent blue field cylinders, pressing jump in place until the idle limit (194
  of the last 200 decisions were jumps). The expected returns of five of the eight actions are nearly equal (0.08 to
  0.10): the policy has no preference there because it never saw what follows. The cause is missing exploration, not a
  reward error.
- **Position novelty broke the freeze.** With the same learner and the same route reward plus 0.02 per new cell, the
  median rose through a series of plateaus (about 3%, 5%, 12%, 20%, 22%, 24%, 25.5%, 26%). Each plateau was left after
  tens of thousands of decisions, but the later ones gain only about a point per two-hour block. Blocks 8 to 10 moved
  the median from 25.5% to 26.1%: it has flattened again.
- **Where it stops.** A frozen greedy evaluation of the block 10 checkpoint (`record-dqn`, seeds 11, 23 and 47; the
  three runs are identical because the emulator and the policy are deterministic) ends in a death after 359 decisions at
  26.3% confirmed route (30.4% by geometric projection). The video shows Mario crossing a molten sun-like planetoid on
  small stone platforms, with the life meter at 1 and the action values now clearly ordered (forward highest), before he
  dies on the planetoid. This stretch is a skill obstacle (lava and narrow platforms), not an exploration dead end.
- **One outlier:** a single collection attempt in block 9 reached 32.08% of the route. It was not repeated: the next
  most advanced attempt of that block was under 27%.

## Changes and fixes made for this run

- `position_novelty` may now be combined with the regional waypoint reward (it was refused). An attempt that advances
  the route is not idle. The continuous route reward, curricula and demonstrations are still refused.
- The environment's fall limit `min_reward_y` pays and gates novelty. The default of -1800 came from Sweet Sweet; the
  Center start is at y = -4100, so no novelty was ever paid until the Center arm set it to -6000 (found in a smoke test
  before training; the test `test_center_novelty_arm_fall_limit_is_below_the_center_start_height` pins it).
- Evaluation videos now show all four stacked frames the network receives.

## Limits

- One seed per arm and one start state. The novelty arm's visit counts are per worker and restart with every block, so
  the decay across blocks is not continuous; the bonus size (0.02) was my choice, not tuned.
- The gate radius, the grounded condition and the final location are provisional; the owner has not confirmed the
  video's final frame ([setup](center-of-universe.md)).
- Checkpoints, replay and videos are local and not distributed. The comparison of arms is between runs that differ in
  one reward term; it does not rank learners.
- The video of the block 10 policy is local: `runs/model-free/center-novelty-20261113/center-novelty-greedy-seed11-72s.mp4`
  (the first 72 seconds, one episode, re-encoded). It can be regenerated from a checkpoint with
  `record-dqn <checkpoint-dir> --seeds 11 --max-steps 500` under the novelty config.
