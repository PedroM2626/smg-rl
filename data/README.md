# Local game assets

Place your own legally obtained game images here. The Dolphin savestates below are ignored by Git and never committed; they are published as the `smg-rl-states-core.tar` bundle (the Luigi states below as `smg-rl-states-luigi-coins.tar`) of the [`states-v1` release](https://github.com/PedroM2626/smg-rl/releases/tag/states-v1) and installed with `scripts/manage_states.py unpack` (see the [main README](../README.md#savestates)). Game images are not distributed.

| File | Purpose |
| --- | --- |
| `SB4E01.s01` | Spinning and Spinning and Spinning: episode start |
| `SB4E01.s02` | First variation: near-star validation only |
| `SB4E01.s03` | Turning Turning Double Time: episode start |
| `SB4E01.s04` | Second variation: near-star validation only |
| `RMGE01.s01` | Galaxy 1 Sweet Sweet Galaxy: training start and detector validation |
| `RMGE01.s02` | Galaxy 1: near-star completion detector validation only |
| `RMGE01.s03` | Center of the Universe: full-start recording reference |
| `RMGE01.s04` | Center of the Universe: owner-selected pre-boss location, not a star |

Luigi's Purple Coins states live in `data/luigi-coins/` (a separate folder because the slot names would clash with the table above). Their hashes are pinned in the Luigi configs.

| File | Purpose |
| --- | --- |
| `data/luigi-coins/SB4E01.s02` | Galaxy 2 Mario Squared Galaxy: mission start, 0 of 100 purple coins, life 3 (training start) |
| `data/luigi-coins/SB4E01.s03` | Galaxy 2: 100 purple coins, life 2, one jump from the star (detector validation only) |
| `data/luigi-coins/RMGE01.s01` | Galaxy 1 Toy Time Galaxy: mission start, 0 purple coins (zero-shot evaluation start) |
| `data/luigi-coins/RMGE01.s02` | Galaxy 1: 102 purple coins, star in reach (detector validation only) |

The Galaxy 1 ROM is `Super Mario Galaxy (USA) (En,Fr,Es).nkit.iso`; Galaxy 2 is `Super Mario Galaxy 2 (USA) (En,Fr,Es).nkit.iso`. Use the matching explicit configuration.

The near-star states and the Center of the Universe goal-location fixture are not completed states. Completion must be observed after collecting the star.
