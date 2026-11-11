# Luigi's Purple Coins: RAM signals (November 10, 2026)

Two owner missions, both "collect the purple coins, then jump to the star": Galaxy 2 **Mario Squared Galaxy**,
"Luigi's Purple Coins Chaos" (internal stage `MarioOnMarioGalaxy`, scenario 2), and Galaxy 1 **Toy Time Galaxy**,
"Luigi's Purple Coins" (`FactoryGalaxy`, scenario 5). Only the Galaxy 2 mission is planned for training; the same agent
is to be tested on the Galaxy 1 mission as a similar stage. This page records what was found and how it was checked.
The reward the owner specified for the Galaxy 2 stage is implemented (section "Reward and environment"); no training has been run on it.

## Savestates and how they are handled

The owner's states are kept in the project at `data/luigi-coins/` (`SB4E01.s02` and `.s03`, `RMGE01.s01` and `.s02`;
byte-identical copies of the originals, hashes below). They are ignored by Git like every savestate, so they are not in
a clone: anyone reproducing this work needs the matching bundle (`smg-rl-states-luigi-coins`, defined in
`scripts/manage_states.py` and not yet uploaded to the release) or their own saves of the same moments. The configs pin
each file's SHA-256 and refuse a different one. They are read only: the native backend copies a state into its own
disposable profile before loading it and nothing is written back. Hashes: `RMGE01.s01` 9af10580…, `RMGE01.s02` fc0933dc…,
`SB4E01.s02` eb1eb06c…, `SB4E01.s03` d4ecf4d1…; they were recorded before any work and checked after every
verification run (unchanged). The project's other `data/` states were not touched.
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
- **A live damage event (3 to 2) is still not observed for Galaxy 2.** On November 11 `scripts/luigi_hit_test.py`
  played 18 attempts from the owner start state with a forward-biased random policy (8 attempts, then 10 with a
  more forward policy): every attempt fell off the sky platforms within 15 to 80 decisions (a fall kills at energy 3,
  with no damage step) or hit a transient pointer fault after the fall, and none met an enemy. The owner state hash was
  unchanged. The energy value stays supported only by the two states (3 at the start and after respawn, 2 where the
  HUD shows life 2, see the `s03` frame) and the filtering by tracking; a savestate placed just before a hazard or
  enemy of this stage would settle it, and until then the -1 damage penalty rests on that evidence.
- No coin-collection ground truth beyond the HUD frames I read (counts of 1 and 2) exists; the counters are
  consistent with them, and mirrors agree.
- The stage and scenario names were added to the memory profiles (`extra_stages`, `extra_scenarios`).

## Reward and environment (implemented November 11, 2026)

Config `config/luigi-mario-squared-altbtr-20261110.yaml` (learner: the alternative BTR of the benchmark, the project's
single model for every stage; calibration `config/luigi-smg2-calibration.json`):

| Event | Reward |
| --- | --- |
| purple coin gained | +1 each (the counter rises by one per coin) |
| damage | -1 per point of energy lost (`rewards.damage`); not paid on the decision that kills |
| death | 0 |
| count reaches 100 | switches on the approach credit |
| approach to the star | 10 in total, paid only when the 3D distance to the star position falls below its previous best, in proportion to the gain over the distance at the moment 100 was reached; retreating or regaining distance pays nothing |
| star (RAM star, scene nerve) | +50 |

Notes and limits:
- The star position is the owner's `star_ready` state (Mario one jump below the star); it is an approximation of the star,
  and only 47 units from the start point, so the approach credit is measured from wherever the count reaches 100.
  Success is the RAM star, never this point.
- Idle (no coin and no approach) ends an attempt after 400 decisions; an attempt has at most 9000 decisions. A counter
  jump of more than 25 coins in one decision is treated as a memory fault.
- A variation may set `start_state` plus its `start_state_sha256`: the owner file is hashed (once per process), checked
  against the pin and copied by the emulator profile into its own disposable slot; the original is only read. A bridge
  "error 2" at load is tolerated with one idle decision. The emulator also boots from that state, so the stage check
  at start-up passes.
- Checked on the real emulator with one worker (not a training run): the reset reads 0 coins and energy 3 from `s02`,
  100 coins and energy 2 from `s03`, a jump from `s03` ends the attempt with the star and a reward of +50, and the owner
  hashes are unchanged afterwards (`.local/luigi-coins/env_check.py`).
- **Toy Time zero-shot test (SMG1):** `config/luigi-toy-time-zero-shot-20261110.yaml` (calibration
  `config/luigi-smg1-calibration.json`) uses the same reward and learner settings, the coin counter at `0x810550DC`
  (the energy comes from the SMG1 profile) and the owner state `RMGE01.s01` (sha256 pinned); it has training
  disabled and a `zero_shot.source_config` pointing at the Mario Squared config. `record-dqn` then accepts a
  checkpoint trained under the source config (`checkpoint_signature` demands identical pixels, crop, frame skip,
  action set and execution and scenario count) and runs it frozen, reporting the best coin count and the star.
  Checked on the real emulator with one worker: `s01` reads 0 coins and energy 3, `s02` reads 102 coins (the count
  passes 100, so the approach credit is armed at reset) and a jump ends the attempt with the star and +50; the owner
  hashes were unchanged. No agent has been evaluated on it.

## Scripts

`scripts/luigi_probe.py` (stage, position, a full frame), `scripts/luigi_dump.py` (MEM1 dumps; MEM2 is not located by
the reader), `scripts/luigi_explore.py` (a forward-biased random policy that logs chosen words and frames) and
`scripts/verify_luigi_signals.py` (the checked evidence above) and `scripts/luigi_hit_test.py` (the damage search).
