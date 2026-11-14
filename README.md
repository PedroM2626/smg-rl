# SMG RL

Visual reinforcement-learning experiments for **Super Mario Galaxy** and
**Super Mario Galaxy 2** on Windows, using a patched Dolphin emulator. This is a
research project. One frozen greedy policy reaches a RAM-confirmed star from the
canonical start in the deterministic emulator, but it reproduces a 312-action
sequence found by beam search and fails after one or two random actions, so it has
not learned the stage in a general sense; no agent has learned the stage from
scratch without demonstrations or anchors. The furthest from-scratch result is on
The Center of the Universe: 26% of its route confirmed, no gate reached. Assisted
starts, scripted searches, detector fixtures and imagined trajectories are reported
separately and never counted as learned mission completions.

## Current status (November 14, 2026)

- **Center of the Universe, from scratch (one seed per arm, no demonstration, no anchor, no curriculum; no attempt
  reached the end-location gate).** The alternative BTR learner with a reward of 0.1 per confirmed route segment (273
  segments) and 50 for the gate froze at 3.24% of the route for four blocks (1.1 million decisions): the policy jumped
  in place on one platform and its action values were flat. Adding a position-novelty bonus (0.02 per new cell per
  attempt) broke the freeze: the median confirmed route rose through successive plateaus to about 26% in 2,570,500
  decisions (10 blocks of two hours, four workers), best single attempt 32.1%, typical best about 27%. The frozen greedy
  policy of the last checkpoint crosses a molten planetoid on small platforms and dies there after 359 decisions at
  26.3% of the route. The chain was stopped by the owner after block 10; nothing is training. Details, per-block
  tables and limits: [Center training results](docs/center-training-20261114.md).
- **Sweet Sweet Galaxy gate (met, with limits).** The checkpoint after S5r block 29
  reaches the RAM star from the canonical start in 311 decisions in frozen greedy
  evaluation (`record-dqn`, epsilon 0, three runs that are identical because the
  emulator and the policy are deterministic). 308 of its 311 actions equal the
  searched sequence it was taught with. With evaluation epsilon 0.01, 6 of 30
  episodes reach the star: all 3 with no random action, 2 of 16 with one, 1 of 6
  with two, none of 5 with three or more. One start state, one game build, one
  training seed. The action sequences came from beam searches over the eight
  actions; the human recording supplied the anchors and the route used for scoring.
- **From scratch (no demonstration, no curriculum anchors), one seed each, no RAM
  star in any of them:** arm F (462,132 decisions), the BTR learner with the
  official repository's default settings (290,608), F with an idle penalty
  (296,572), a position-novelty exploration arm (551,288) and the BTR learner with
  a landing-gated waypoint reward (142,088). Random-checkpoint starts without
  demonstrations: 169,656 decisions, no anchor cleared. They idle, wander or fall
  before the first waypoint; the best route advance in them was 2.6%, except the
  exploration arm, which moved but confirmed no waypoint.
- **Projection credit, from scratch (one seed, about one hour per arm, no RAM star from the canonical start in any).**
  A fixed bonus per waypoint passed along a dense human route, with no cost for failure (the recipe of the Mario 64 DS
  project), reaches 2.5% to 5.0% mean recent route advance and 4% to 12.5% best single-attempt advance across the BTR,
  alternative BTR, official BTR, F and BBF-lite learners; the differences are too small to rank them. The attempts
  mostly end in a fall within a median of 17 to 193 decisions. The learners were compared on their stochastic collection
  attempts, not on frozen greedy evaluations. Start methods: a backward frontier over the 11 human anchors made the
  F learner reach the star from the last anchor (60 stars, all from anchor starts, none from the canonical start);
  uniform random anchors did not improve the advance from the canonical start in one hour. Numbers, errors and
  caveats are in the [exploration study](docs/exploration-study.md); the sibling projects and licences are in
  [related projects](docs/related-projects.md).
- **Prepared, not trained:** Galaxy 2 Double Time (projection route, 11 validated anchors, BBF-lite and altBTR
  from-start configs, start-up smoke test); no agent has been trained on it. For The Center of the Universe
  (projection route, 8 anchors, a goal-marker gate that ends an attempt at the end location instead of a star) see its
  [guide](docs/center-of-universe.md) and the results above. RAM signals for the two Luigi's Purple Coins missions, the
  implemented reward for the Galaxy 2 Mario Squared mission (+1 per purple coin, -1 per damage point, no death cost, an
  approach credit once 100 coins are collected, +50 for the star, starting from the owner's savestate) and the Toy Time
  zero-shot evaluation config (SMG1, frozen agent, no training) are in
  [docs/luigi-purple-coins.md](docs/luigi-purple-coins.md); their states are in the `smg-rl-states-luigi-coins` bundle.
  The Galaxy 2 damage signal (energy 3 to 2) has not been observed live.
- **One model for every stage (decision of November 11, 2026):** the alternative BTR learner. In the projection-free
  round it tied the best recent mean (5.0%) with BBF-lite while using a median of 32 decisions per attempt instead of
  193 and making more than twice the attempts; this is a working choice from one seed, not a proven ranking. From-start
  configs exist for every stage: `algorithm-benchmark-altbtr-pf-20261110.yaml` (Sweet Sweet),
  `double-time-altbtr-start-20261110.yaml`, `center-altbtr-start-20261110.yaml` and
  `luigi-mario-squared-altbtr-20261110.yaml`. Only the Center configs (`center-altbtr-start-20261110.yaml` and
  `center-altbtr-novelty-20261113.yaml`) have been trained; the others are untrained.
- **Not done yet:** a from-scratch agent that reaches a gate or star from the canonical start, a replicated benchmark
  (the current one has one seed and one hour per arm), a long training on Sweet Sweet, Double Time or the Luigi
  mission with the single learner, Spinning, GA and ES arms, Go-Explore, a native progress variable (a scan of the
  Sweet Sweet replay found none).

The final success gate is unchanged: a frozen greedy policy must reach the RAM
star from the canonical start, and a gate passed by reproducing a searched
sequence is reported as such.

## Earlier status (November 3, 2026, superseded by the section above)

The active line was a backward human-anchor curriculum on SMG1 Sweet Sweet Galaxy
(Rocky Road), documented block by block in the
[exploration study](docs/exploration-study.md):

- A human full-start recording that reached the RAM star provides eleven verified
  intermediate savestate anchors (route waypoints 01, 02, 04, 08, 14, 18, 23,
  29, 34, 38 and 40). Training starts at an anchor (the *frontier*) and the frontier
  moves toward the canonical start as the policy masters each section.
- In frozen greedy evaluation (`record-dqn`, epsilon 0, no learning) of the
  S5b block-8 checkpoint, six of six seeds started from waypoint 23 reached the
  RAM star in 167 decisions. Greedy starts from waypoints 18 and 14 timed out,
  and the canonical start died after 27 decisions. These are assisted-anchor
  results, not full-start completions.
- In S5c block 9 the hybrid frontier rule moved the frontier from waypoint 23 to
  waypoint 18. All attempts from waypoint 18 in that block failed.
- A route-aware savestate beam search found a 175-decision sequence from
  waypoint 18 to the RAM star. It was confirmed under the training configuration's
  route credit and idle limit. `config/anchor-diagnostic-s5d-20261103.yaml` uses it
  as the waypoint-18 demonstration. The search result is an assisted control
  fixture, not a learned completion.

The final success gate is unchanged: a frozen greedy policy must reach the RAM
star from the canonical start. Earlier studies are summarized in the
[experiment history](docs/experiment-history.md). Some dated status pages, such as
[current evidence](docs/current-status.md), describe the project as of their
own dates.

## Tasks and inputs

| Game | Mission | Full-start slot | Near-star detector fixture |
| --- | --- | --- | --- |
| Galaxy 2, `SB4E01` | Twisty Trials: Spinning and Spinning and Spinning | 1 | 2 |
| Galaxy 2, `SB4E01` | Twisty Trials: Turning Turning Double Time | 3 | 4 |
| Galaxy 1, `RMGE01` | Sweet Sweet Galaxy: Rocky Road | 1 | 2 |
| Galaxy 1, `RMGE01` | The Center of the Universe (end-location gate, not a star) | 3 | 4 (owner-selected location, not a star) |
| Galaxy 2, `SB4E01` | Luigi's Purple Coins Chaos, Mario Squared Galaxy (owner state, `data/luigi-coins/`) | `SB4E01.s02` | `SB4E01.s03` |
| Galaxy 1, `RMGE01` | Luigi's Purple Coins, Toy Time Galaxy (zero-shot evaluation only) | `RMGE01.s01` | `RMGE01.s02` |

The Center of the Universe is trained with its own configs
([guide](docs/center-of-universe.md), [results](docs/center-training-20261114.md)); its slot 4 is a location, not a
completed state. The Luigi missions are described in [docs/luigi-purple-coins.md](docs/luigi-purple-coins.md). The
near-star states validate the completion detector and do not establish learned success. See
[asset names and roles](data/README.md).

## Method of the active line (from November 11, 2026)

One learner for every stage, trained from the canonical start only (no anchors, no demonstrations):

- **Observation and actions:** as below (four stacked 84 x 84 grayscale frames, `motor8-longjump-v1`, 12 emulated
  frames per decision). The reward reaches the learner unchanged: there is no reward clipping or sign transform.
- **Learner:** the alternative BTR: IQN with an IMPALA-style encoder (32 channels), 8 quantiles, noisy layers,
  Munchausen terms, gamma 0.99, learning rate 1e-4, weight decay 0.01, prioritized replay, three-step returns, four
  native Dolphin workers.
- **Reward (route stages):** 0.1 for every newly confirmed route segment (paid once, never taken back), 50 for the
  star or the Center gate, 0 for death, damage and timeout. Optional position novelty: 0.02 per 200 x 250 x 200 unit
  cell new to the episode, decaying with the visit count (used for the Center arm that broke its freeze).
- **Reward (Luigi Mario Squared):** +1 per purple coin, -1 per damage point, no death cost, an approach credit of 10 in
  total once 100 coins are collected, +50 for the star.
- **Evaluation:** a frozen greedy `record-dqn` run from the canonical start; the video overlay shows the four stacked
  frames, the expected return of each action and the applied controls.

### Earlier method (the human-anchor curriculum, October 27 to November 4)

- **Observation:** four stacked, cropped 84 × 84 grayscale frames plus the
  scenario identity. RAM coordinates, health and terminal signals are used for
  reward, curriculum and evaluation only; they are not policy inputs.
- **Actions:** `motor8-longjump-v1`, eight exclusive logical actions (none,
  right, left, forward, backward, jump, spin, crouch). A crouch followed by a
  jump during forward motion produces a long jump. Each decision advances 12
  emulated frames. See [controls and signatures](docs/motor8-and-algorithms.md).
- **Learner:** model-free IQN with an IMPALA-style residual encoder, prioritized
  replay, three-step returns, gamma 0.997 and training epsilon 0.02, with three
  native Dolphin collectors. No world model, imagined experience or intrinsic
  reward is active. See [model-free implementation](docs/model-free.md).
- **Reward:** discounted advance along a landing-audited human route
  (`config/routes/smg1-human-landing-v2.json`, schema 5, waypoint lookahead 5),
  RAM star +20, death or idle timeout -2, HP loss -0.25 per HP. A waypoint is
  credited only after a grounded landing near it. An episode ends after 1,440
  frames without route progress.
- **Curriculum:** 20% of starts rehearse anchors after the frontier. The hybrid
  frontier rule makes every tenth frontier start a frozen greedy probe. The
  frontier moves only after two consecutive probe stars and at least 10% stars
  over the last 30 training attempts there.
- **Demonstrations:** every third episode replays a uniformly sampled prefix of a
  savestate-search sequence for its anchor, then hands control to the policy. A
  DQfD large-margin loss (0.8) is applied to demonstration transitions.
  Demonstration episodes never count as policy successes or frontier evidence.

The completion detector reads a scene-state address (`0x806A1BD8` for the tested
SMG1 mission, `0x807D5318` for the tested SMG2 missions). Address interpretation
depends on the validated game and emulator build. See [SMG1 memory](docs/smg1.md),
[SMG2 memory](docs/memory-profile.md) and [damage limitations](docs/damage-and-recovery.md).

## Savestates

Savestates are never committed to Git. They are published as assets of the
[`states-v1` GitHub release](https://github.com/PedroM2626/smg-rl/releases/tag/states-v1)
as uncompressed tar bundles with a SHA-256 manifest:

| Bundle | Contents | Destination |
| --- | --- | --- |
| `smg-rl-states-core.tar` | Mission starts and near-star fixtures, slots 1-4 of both games | `data/` |
| `smg-rl-states-smg1-human-bank.tar` | Eleven Sweet Sweet human anchors and `bank.json` | `.local/exploration-study-20261029/human-bank/` |
| `smg-rl-states-regional-curriculum.tar` | Earlier SMG1/SMG2 assisted regional starts | `.local/regional-curriculum/` |
| `smg-rl-states-luigi-coins.tar` | Luigi's Purple Coins states (Galaxy 2 Mario Squared, Galaxy 1 Toy Time) | `data/luigi-coins/` |

Download the bundles into the repository root, then verify and install them:

```powershell
.\.venv\Scripts\python.exe scripts/manage_states.py verify smg-rl-states-core.tar
.\.venv\Scripts\python.exe scripts/manage_states.py unpack smg-rl-states-core.tar smg-rl-states-smg1-human-bank.tar smg-rl-states-regional-curriculum.tar smg-rl-states-luigi-coins.tar
```

`unpack` checks every file before installing any of them. It does not overwrite a
different local file unless `--force` is given.

Compatibility limits:

- The savestates come from the USA game builds (`RMGE01`, `SB4E01`) and Dolphin
  2609a. They require your own legally obtained game images. ROMs and trained
  checkpoints are not distributed.
- The anchor banks record the SHA-256 of the patched Dolphin executable that
  created them (`b9f2131c…`). The curriculum loader rejects a bank when the
  configured executable hash differs. That exact Windows build is published in the
  [`dolphin-2609a-native-v1` release](https://github.com/PedroM2626/smg-rl/releases/tag/dolphin-2609a-native-v1)
  (see [Installation](#installation-and-local-operation)). A Dolphin built locally
  from the same source can produce a different binary hash; the anchor banks must
  then be re-prepared with that build.
- A savestate is a RAM snapshot of a commercial game. Downloading and using one
  is subject to the game's rights; it is not covered by this repository's MIT
  license.

## Installation and local operation

Use Windows, 64-bit Python 3.11 or 3.12, an NVIDIA GPU for training, and your own
game images in `data/`. The tested environment used Python 3.11 and Torch
2.14.1+cu126. The lock file records exact packages and an extra CUDA index.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
```

Training and evaluation use the native backend, which requires a patched Dolphin
2609a. Either download `smg-rl-dolphin-2609a-native-win64.zip` from the
[`dolphin-2609a-native-v1` release](https://github.com/PedroM2626/smg-rl/releases/tag/dolphin-2609a-native-v1)
and extract the contents of its top-level folder into `.local/native-dolphin-frontier/`
(the path used by the current configurations), or build it with
`scripts/build_native.ps1`. The release also contains the complete corresponding
GPL source. See [native build and prerequisites](docs/native-backend.md). Hidden workers still
render through a hidden surface; this is not renderer-free emulation.
Configuration files are complete profiles, not overlays. If your paths differ
(for example the Dolphin executable), copy a YAML under `config/`, adjust
`paths` and revalidate. `--config` precedes the subcommand.

```powershell
# Bounded training block (resume needs a matching checkpoint directory with replay).
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/anchor-diagnostic-s5d-20261103.yaml train-dqn --steps 999996 --max-seconds 3600
# Frozen greedy evaluation from the canonical start, no learning.
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/anchor-diagnostic-s5d-20261103.yaml record-dqn <checkpoint-dir> --seeds 1 2 3
# The Center of the Universe arm that reached 26% of the route: a two-hour block resumed from a checkpoint, and a video.
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/center-altbtr-novelty-20261113.yaml train-dqn --steps 1000000 --max-seconds 7200 --resume <checkpoint-dir>
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/center-altbtr-novelty-20261113.yaml record-dqn <checkpoint-dir> --seeds 11 --max-steps 500
# Assisted frozen evaluation from a verified anchor.
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/anchor-diagnostic-s5d-20261103.yaml record-dqn <checkpoint-dir> --assisted-start --anchor human-waypoint-23.state
```

A checkpoint must match its game, model, action, reward and replay signatures.
Trained checkpoints are local and not distributed. The [CLI reference](docs/cli.md)
documents commands and their side effects, and the
[training guide](docs/training.md) covers longer runs. To record a human
demonstration, see the [human recording guide](docs/human-recording.md).

TensorBoard event files are written under `runs/model-free/tensorboard`:

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir runs/model-free/tensorboard --host 127.0.0.1 --port 6006
```

## Earlier experiments

Before the human-anchor curriculum, full-start training produced no stars. In the
October 27 audit, an SMG1 IQN/IMPALA allocation stopped after about 2 hours 18
minutes because the disk filled. Its last valid checkpoint contains 749,006 real
decisions and 12,125 completed attempts; see the
[failure audit](docs/iqn-four-hours-20261027.md). Genetic search over visual
policies and goal distance ([genetic results](docs/genetic-visual.md),
[goal-distance trial](docs/genetic-goal-distance-20261027.md)), expanding routes,
reward-scale comparisons and world models were also tested without a full-start
star. The [experiment history](docs/experiment-history.md) records these findings,
negative results and comparison limits.

## Documentation, tests and license

- [Documentation index](docs/README.md) and [exploration study](docs/exploration-study.md) and [related projects](docs/related-projects.md)
- [Center training results](docs/center-training-20261114.md), [Luigi's Purple Coins](docs/luigi-purple-coins.md), [Training guide](docs/training.md), [CLI](docs/cli.md) and [reproducibility](docs/reproducibility.md)
- [World model](docs/world-model.md) and [experiment protocol](docs/experiments.md)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Portable tests do not boot Dolphin or prove that a policy solved a mission.
Original project material uses the [MIT license](LICENSE). Games, Dolphin and
dependencies have separate rights, as described in
[third-party notices](THIRD_PARTY_NOTICES.md).
