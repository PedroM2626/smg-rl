# Center of the Universe: novelty arm continued, blocks 26 to 62 (November 17 to 20, 2026)

Continuation of the novelty arm described in [Center training results](center-training-20261114.md): the alternative BTR
learner (IQN, IMPALA encoder, 32 channels, 8 quantiles, noisy layers, Munchausen, gamma 0.99, learning rate 1e-4, weight
decay 0.01) on the Galaxy 1 Center of the Universe setup ([the setup](center-of-universe.md)), config
[`center-altbtr-novelty-20261113.yaml`](../config/center-altbtr-novelty-20261113.yaml), one seed, canonical slot 3 start,
no demonstration, no anchor, no curriculum. Each block is a two-hour wall-clock budget with four workers, resumed from the
previous block's checkpoint. **No attempt has reached the end-location gate yet**; training was still running when this
page was written (block 63 in progress, 14,139,056 real decisions at the end of block 62).

"Route" is the `opening_advance` field of `attempts.jsonl` (fraction of the route confirmed). The numbers are over the
stochastic collection attempts of each block, not frozen greedy evaluations. Position-novelty visit counts are per worker
and restart with each block.

## Per-block results

| Block | New real decisions | Total real decisions | Attempts | Median route | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 26 | 247,080 | 5,884,600 | 428 | 47.34% | wall 47 to 48% for blocks 26 to 31 |
| 27 | 244,216 | 6,128,816 | 399 | 47.77% | |
| 28 | 244,520 | 6,373,336 | 390 | 47.48% | |
| 29 | 246,620 | 6,619,956 | 388 | 47.22% | |
| 30 | 241,668 | 6,861,624 | 419 | 47.91% | |
| 31 | 241,212 | 7,102,836 | 398 | 47.93% | |
| 32 | 242,192 | 7,345,028 | 449 | 48.88% | |
| 33 | 250,596 | 7,595,624 | 593 | 27.53% | temporary regression, recovered inside the block |
| 34 | 243,920 | 7,839,544 | 396 | 56.32% | step to about 57% |
| 35 | 225,248 | 8,064,792 | 331 | 57.73% | |
| 36 | 209,712 | 8,274,504 | 288 | 60.83% | wall at 60.8% |
| 37 | 208,232 | 8,482,736 | 283 | 60.83% | |
| 38 | 139,008 | 8,621,744 | 192 | 60.80% | stopped early by the memory guard (clean checkpoint) |
| 39 | 249,272 | 8,871,016 | 355 | 60.72% | best 61.94% |
| 40 | 19,628 | 8,890,644 | 31 | 47.22% | stopped by the owner for a video evaluation |
| 41 | 200,556 | 9,091,200 | 359 | 47.76% | regression for about 240 attempts, then recovery |
| 42 | 186,988 | 9,278,188 | 294 | 60.26% | |
| 43 | 240,388 | 9,518,576 | 350 | 60.36% | one attempt at 68.1% |
| 44 | 238,412 | 9,756,988 | 336 | 60.78% | 47 attempts above 65% |
| 45 | 240,068 | 9,997,056 | 337 | 67.42% | step: second half of the block at about 70% |
| 46 | 239,776 | 10,236,832 | 309 | 70.67% | |
| 47 | 241,340 | 10,478,172 | 325 | 71.19% | wall at about 71% |
| 48 | 237,808 | 10,715,980 | 302 | 71.49% | |
| 49 | 241,632 | 10,957,612 | 391 | 71.16% | best 75.20%, regression in the middle |
| 50 | 240,904 | 11,198,516 | 320 | 72.33% | best 78.51% |
| 51 | 242,004 | 11,440,520 | 327 | 59.10% | about 57.7% for 165 attempts, then 75% |
| 52 | 242,804 | 11,683,324 | 286 | 79.07% | |
| 53 | 244,936 | 11,928,260 | 412 | 79.07% | best 87.12%, regression at the end |
| 54 | 244,520 | 12,172,780 | 382 | 36.56% | regression for 300 attempts, then 83% |
| 55 | 244,744 | 12,417,524 | 264 | 83.19% | best 89.75% |
| 56 | 246,176 | 12,663,700 | 288 | 84.34% | |
| 57 | 244,852 | 12,908,552 | 298 | 85.47% | best 90.76% |
| 58 | 244,680 | 13,153,232 | 310 | 86.91% | |
| 59 | 246,000 | 13,399,232 | 256 | 90.28% | best 95.63% |
| 60 | 245,596 | 13,644,828 | 300 | 90.92% | |
| 61 | 246,040 | 13,890,868 | 290 | 91.63% | best 95.57% |
| 62 | 248,188 | 14,139,056 | 376 | 37.33% | regression for about 200 attempts, 92% before and after; best 95.57% |

## What the numbers show

- The median route advanced through plateaus (about 47.5%, 60.8%, 71%, 79%, 84%, 91%). Each plateau had a stretch of
  several blocks where almost every attempt died at the same place, then a minority of attempts passed it, then the
  median followed.
- Seven blocks (33, 41, 49, 51, 53/54, 62, and block 40's first ten minutes) contain a temporary regression of 100 to 300
  attempts where the median fell to 4 to 35% and recovered without intervention, usually inside the same block. The cause
  is not established. A frozen stochastic evaluation of the block 39 checkpoint, recorded while block 40 was in a
  regression, reached the same 47.2% as the training attempts of that time (two attempts, deaths after 586 and 620
  decisions), so at least some of the effect lives in the checkpoint's weights and not only in the collector.
- The best attempt of block 59 reached 95.63% of the route; blocks 60 to 62 did not exceed it. The gate additionally needs
  the agent to be grounded and alive inside the 300-unit sphere, so the last part of the route may be a short wall.

## Operations notes

- A native access violation inside zlib's `compress()` crashed three checkpoint saves (blocks 28, 32 and 33 of the
  chain). Replay checkpoints are now stored without deflate (`np.savez`), which `np.load` reads in both layouts.
- The chain script stops below 25 GB of free disk and now removes the `replay.npz` of old blocks (keeping the two newest)
  below 50 GB. Weights (`agent.pt`) and manifests of every block are kept, so old checkpoints can still be evaluated but
  no longer resumed.
- A 20-block cap in the chain script stopped training for about 5.4 hours after block 62 before it was noticed; the cap
  was raised and training resumed at block 63.
- Nothing here counts imagined or synthetic data as a mission completion; the real-decision counter only grows with
  emulator decisions.
