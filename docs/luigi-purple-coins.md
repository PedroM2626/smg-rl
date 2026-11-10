# Luigi's Purple Coins: RAM signals (November 10, 2026)

Two owner missions, both "collect the purple coins, then jump to the star": Galaxy 2 **Mario Squared Galaxy**,
"Luigi's Purple Coins Chaos" (internal stage `MarioOnMarioGalaxy`, scenario 2), and Galaxy 1 **Toy Time Galaxy**,
"Luigi's Purple Coins" (`FactoryGalaxy`, scenario 5). Only the Galaxy 2 mission is planned for training; the same agent
is to be tested on the Galaxy 1 mission as a similar stage. This page records what was found and how it was checked.
No reward weights and no training are defined here; the owner will specify the reward for the Galaxy 2 stage.

## Savestates and how they are handled

The owner's states are in `D:\luigi-coins` (`SB4E01.s02` and `.s03`, `RMGE01.s01` and `.s02`). They are read in place:
the native backend copies a state into its own disposable profile before loading it, nothing is written back, and
their SHA-256 hashes were recorded before any work (`.local/luigi-coins/original-hashes.json`, local only) and
checked again after every verification run (unchanged). The project's own `data/` states were not touched.
The Dolphin native bridge reports error 2 when these states are loaded (a captured frame it does not accept); the
load itself succeeds, which the stage name, position and a frame confirm, so `scripts/luigi_dump.py` tolerates that error
and no other.

| State | Content (read from the frame and RAM) |
| --- | --- |
| `SB4E01.s02` | Galaxy 2 mission intro, 0 of 100 purple coins, life 3 |
| `SB4E01.s03` | 100 of 100 purple coins, life 2, one jump from the star |
| `RMGE01.s01` | Galaxy 1 mission intro, 0 purple coins, health 3 |
| `RMGE01.s02` | 102 purple coins (the HUD counts past 100), star in reach |

## Signals

Defined in [`config/luigi-coins-signals.yaml`](../config/luigi-coins-signals.yaml), read through `GameMemory(signals=...)`
and exposed as `Telemetry.purple_coins` and, for Galaxy 2, `health`/`max_health`. Evidence:
[`docs/assets/performance/luigi-coins-signals-20261110.json`](assets/performance/luigi-coins-signals-20261110.json).

| Signal | Galaxy 2 Mario Squared | Galaxy 1 Toy Time |
| --- | --- | --- |
| Purple coins | `0x80A75D08` (live counter), mirrors `0x81321034` and `0x81321038` | `0x810550DC`, mirrors `0x812F3DF4` and `0x812F3DF8` |
| Energy | Mario actor + `0x6B0`, maximum 3 (3 at the start, 2 in `s03`) | already read by the SMG1 profile (actor + `0x380`) |
| Star | existing completion check, scene nerve `0x807D5318` | existing completion check, scene nerve `0x806A1BD8` |

How they were found: the words that are 0 in the start state and equal to the HUD count in the end state (4 candidates
for Galaxy 2, 3 for Galaxy 1) were tracked while a forward-biased random policy collected coins from the start state;
three words rose by exactly one per coin and returned to 0 on respawn, and the HUD read `002/100` when they were 2
(Galaxy 2) and `001/100` when they were 1 (Galaxy 1). The lone word falls to 0 on respawn one step before its two mirrors.
For the Galaxy 2 energy, the words that were 3 at the start and 2 in `s03` were filtered by tracking: five varied freely
and were rejected, two stayed at 3, and both went 2 to 3 on respawn (the actor-relative one is used). The star: stepping
a jump from each end state set the existing completion flag within one to three decisions (the screen showed "You got
a Star!"), with the coin count unchanged.

## What is and is not established

- The counters and the energy value reproduced in every fresh emulator process that loaded these states (more than 25
  loads). Their addresses are heap addresses and are valid for these saved states; a different route into the stage
  (a fresh boot, another savestate) was not tried and may move them.
- **A live damage event (3 to 2) was not observed for Galaxy 2.** Random and idle play (about eight runs of 150 to 250 decisions) was not
  hit. The energy value is supported by the states (3 at the start and after respawn, 2 where the HUD shows life 2).
  It should be confirmed with a deliberate hit before it is used as a damage penalty.
- No coin-collection ground truth beyond the HUD frames I read (counts of 1 and 2) exists; the counters are
  consistent with them, and mirrors agree.
- The stage and scenario names were added to the memory profiles (`extra_stages`, `extra_scenarios`); training on these
  missions would also need start states supported by the environment (it loads canonical start slots today), which is
  not built.

## Scripts

`scripts/luigi_probe.py` (stage, position, a full frame), `scripts/luigi_dump.py` (MEM1 dumps; MEM2 is not located by
the reader), `scripts/luigi_explore.py` (a forward-biased random policy that logs chosen words and frames) and
`scripts/verify_luigi_signals.py` (the checked evidence above).
