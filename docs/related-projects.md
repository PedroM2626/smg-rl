# Related projects

This repository (smg-rl) is one of several reinforcement-learning projects by the
same author, plus the external work it is compared with. This page says what each
project is, what it contributed here and what it does not show. The descriptions of
the sibling projects come from their own READMEs, read on November 6, 2026, without
running or changing them. Their numbers are their authors' reports for their own
games and setups; none of them transfers to Super Mario Galaxy.

## smg-rl (this repository)

[PedroM2626/smg-rl](https://github.com/PedroM2626/smg-rl) trains agents on
Super Mario Galaxy (Wii) and Super Mario Galaxy 2 through a native Dolphin build.
The agent sees 84x84 frames and acts with eight motor labels
(`motor8-longjump-v1`, 12 emulated frames per decision). Progress, damage, death and
the star are read from emulator RAM and never patched. The main learner is an IQN
agent with an IMPALA encoder, noisy dueling heads, prioritised 3-step replay and an
optional DQfD margin loss for scripted demonstrations. Work so far centres on the
Sweet Sweet Galaxy stage: a backward curriculum over human-recorded anchors,
demonstrations found by route-aware beam search, and a frozen greedy evaluation from
the canonical start as the final gate. See the
[exploration study](exploration-study.md) for the experiment record and
[current status](current-status.md) for dated claims. Licence: MIT.

## alternative BTR: mario64ds-rl

[PedroM2626/mario64ds-rl](https://github.com/PedroM2626/mario64ds-rl) is a deep
reinforcement-learning project for Super Mario 64 DS on a Nintendo DS emulator:
three downhill slide courses (`ds1` Cool, Cool Mountain; `ds2` The Princess's Secret
Slide; `ds3` Tall, Tall Mountain) and Tick Tock Clock ascents (`ds8`, `ds9`).

It contains the **alternative BTR** referred to in this repository
(`src/btr.py`, trained by `scripts/train_btr.py`). Its own README states that it is an
independent implementation of the Beyond-the-Rainbow recipe, not the authors' code:
IMPALA CNN on 84x84 grayscale frames with a 4-frame stack, an IQN head with dueling
noisy layers, 3-step returns, a Munchausen term, prioritised replay, soft target
updates (0.005), AdamW and a discount of 0.99, acting with ten buttons.

What its README reports: one network trained on all three slides at once with a coin
reward from RAM and a +20 bonus at the finish region, no human demonstrations, 3.5M
environment steps in two stages (about 5.5 hours of wall clock, five environments per
slide, fifteen in parallel). The final checkpoint's greedy policy finished `ds1` in
20 of 20 episodes, `ds2` in 20 of 20 and `ds3` in 18 of 20, and with up to 15 random
decisions before the agent takes over it finished 20, 20 and 17 of 20. Its README
notes that 20 episodes leave a wide interval and that the game is not deterministic.

How it is used here: its parameters define the **alternative BTR** arm that may join
the algorithm benchmark (see the [study notes](exploration-study.md#planned-algorithm-arms-and-what-the-official-btr-does)).
It differs from the official BTR in several ways that are listed there, among them
the discount (0.99 against 0.997), soft against hard target updates, AdamW and the
lack of spectral norm and adaptive max-pooling. Its game, reward and episode rules
differ from Sweet Sweet's, so its results show that the implementation can learn in
that setting, not what it would do here. Licence: MIT; the ROM and savestates are not
in the repository and remain Nintendo property.

## nsmbds-rl

[PedroM2626/nsmbds-rl](https://github.com/PedroM2626/nsmbds-rl) is a research
pipeline for New Super Mario Bros. on the DeSmuME Nintendo DS core. It explores two
routes to progress tracking: computer vision (dense optical flow and frame
differencing, avoiding volatile memory pointers) and direct RAM and ROM reverse
engineering (telemetry, entity graphs, tile geometry). Its README lists many learning
families: model-free policy gradients (PPO, recurrent PPO, an IMPALA encoder),
self-supervised representations (autoencoders, CURL, SPR, DrQ-v2), curiosity (ICM),
sequence models, world models (DreamerV3, a variant called NE-Dreamer), genetic
search and sep-CMA-ES, and graph networks.

What its README reports: a World 1-1 clear by a 1,032-parameter neuroevolved policy
(206 decisions, 4,020 of 4,032 pixels) on a fixed savestate; specialised controllers
and measured routes clearing seven benchmark courses; a physics-informed world model
(PINN) planning experiment that cleared 0 of 27 episodes over nine courses; and a
re-benchmark of every learned architecture under one reward and action set in which,
under stochastic sampling, no gradient-trained arm cleared World 1-1 in more than
3 of 30 episodes. Its README also warns that replays use fixed savestates and do not
establish generalisation.

How it relates here: it is the earlier, 2D testbed for the same questions this
repository asks in 3D: shaped potential rewards, neuroevolution against gradient
methods, and whether world models help. The world model sits last in this repository's plan by the
owner's decision; its negative planning result there is consistent with that order but
is a different game and setup. Licence: MIT.

## rl-multi_marl-competition

[PedroM2626/rl-multi_marl-competition](https://github.com/PedroM2626/rl-multi_marl-competition)
is a multi-agent study, not a Mario project: a 3D battle arena with nine agents in
three teams of three. Each team is trained with PPO in JAX, a Rust module does the
batched movement and collision checks, and the architectures compete inside one
engine so that the match outcome is itself the measurement. Its README describes a
replicated study (10 seeds, 500,000 steps each, 150 held-out greedy matches per
replicate), documentation covering the formal model, results with confidence
intervals and power analysis, threats to validity and a 56-item code audit. Licence:
MIT.

Why it is listed: it shows the evaluation discipline this repository's benchmark
should aim for (replicated seeds, intervals, an explicit list of what the results
cannot support, an audit before trusting a ranking), which the one-seed algorithm
benchmark here does not yet meet. It shares no code or checkpoints with smg-rl, and
nothing in it is trained or evaluated on Mario Galaxy.

## External work

- **Beyond The Rainbow (BTR)**, Clark, Towers, Evers and Hare, ICML 2025:
  [VIPTankz/BTR](https://github.com/VIPTankz/BTR) (read at commit `093ca77`) and
  [VIPTankz/Wii-RL](https://github.com/VIPTankz/Wii-RL). The official defaults,
  and how this repository's BTR arm follows them, are in the
  [BTR review](btr-research-review.md) and the
  [exploration study](exploration-study.md#planned-algorithm-arms-and-what-the-official-btr-does).
  The public code does not contain Super Mario Galaxy addresses.
- **Bigger, Better, Faster (BBF)**: planned as the BBF-lite arm (higher replay ratio,
  periodic resets, annealed n-step and discount, weight decay, a wider encoder,
  without the self-predictive loss). Its details are to be checked against the paper
  before implementation, so no claim about it is made here yet.

## What works and what does not in mario64ds-rl (read November 8, 2026)

Its README separates two results. The three slide courses (ds1 to ds3) are finished by the alternative BTR with a
coin reward (+1 per RAM coin, +20 at the finish region, no death penalty, no step cost, five environments per slide,
3.5M environment steps; the 500,000-step checkpoint already finished ds1 and ds2 and died on ds3 in its stored
evaluation). Tick Tock Clock is not solved: its README states that no agent has climbed it, that the v1 to v3 agents
stay on the start platform, and that v4, trained with a human-route reward, a curriculum of 89 saved states, 1,732
demonstration transitions and a larger action set, falls at decision 128 from the full start (+268.5 height there, below v3's +378). That is not zero progress: from 13 human checkpoints along the route, v4 gains a median 336.7 route units with 11 of 13 ending in a fall, against 8.2 units and 2 of 13 for doing nothing. Its own diagnosis for the
clock lists the same causes met here: standing still is paid (the gears move, so idling earns optical flow), the straight
distance to the goal is almost all height so a distance ratchet pays nothing along the winding path, random exploration
falls within about 37 decisions, and the action set could not express a running jump.

So there is no working waypoint system there to compare with. What works there is a dense collectible reward on courses
where gravity carries the character forward (idling is not possible on a slide), which is a different task from a stage
that must be crossed by chained long jumps from a platform where idling is stable.

## Licences and what is run of the official BTR

Checked on November 8, 2026 from the repositories' pages: VIPTankz/BTR carries the MIT licence (copyright 2024
VIPTankz), so its agent code may be reused with its notice kept. VIPTankz/Wii-RL has no licence file, so its code
cannot be copied without the author's permission. Wii-RL has its own Mario Kart Wii environment (Luigi Circuit,
four Dolphin instances through Felk's scripting fork) and its own copy of the BTR agent in `BTR.py`; its README
mentions no other game, and the Super Mario Galaxy experiment the paper reports (Appendix J.1, the final fight
of The Center of the Universe) has no public environment code, as the BTR review in this repository records.

What the benchmark runs today is this project's own implementation configured with the official repository's
default settings (arm BTR, and BTR with projection credit), not the official code itself. The alternative BTR
(mario64ds-rl) is a second independent implementation. Running the official agent unchanged would mean
vendoring the MIT BTR repository's `Agent.py`, `networks.py` and `PER.py` with their licence and writing an adapter
to this project's multi-process Dolphin collection; nothing of that is built yet.

## Newer Tick Tock Clock results in mario64ds-rl (commits of November 6 to 8, 2026)

After the README section quoted above, that project added a mode of `train_btr.py` for the clock: the BTR learner with
+1 for each of the 371 waypoints of the recorded human ascent passed for the first time and +100 for the star, no coin
term, and the credit computed by projecting the position onto the route (see `src/route.py`). Its commit messages
report about 26 waypoints after a 2-hour run from scratch and 58 to 63 after a second 2-hour stage, and its result
tables (`results/btr_ttc_wp_train.csv`, `btr_ttc_wp2_train.csv`) show the mean route arc length of recent episodes at
about 3,860 and 8,900 units of a 53.2k-unit route; arc-length and no-coin variants reached less in the same tables.
These are the figures reported by that project, read here from its commits and CSV files; they are progress along the
route, not a completed climb, and its README had not yet been updated for them when read.
