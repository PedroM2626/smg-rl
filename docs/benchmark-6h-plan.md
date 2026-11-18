# Planned 6 hour Sweet Sweet benchmark (prepared, not run)

Agreed on November 18, 2026. Every arm trains on SMG1 Sweet Sweet Galaxy (Rocky Road) from the canonical start
only, with no demonstration, one seed and 4 workers, for one continuous 6 hour block (about 730,000 real
decisions at the 34 decisions/second measured on the Center run; the world-model arm is slower because the
model also trains). A single block keeps the position-novelty visit counts, which are per worker and reset at
every resume. Nothing here has been trained: it needs the emulator, so it only starts after the Center run
ends and only on an explicit go from the owner.

| # | Arm | Status | Config | Command |
|---|-----|--------|--------|---------|
| 1 | alternative BTR + position novelty (reference) | ready | `benchmark6h-altbtr-novelty-20261118.yaml` | `train-dqn --steps 1000000 --max-seconds 21600` |
| 2 | alternative BTR without novelty (control) | ready | `benchmark6h-altbtr-plain-20261118.yaml` | `train-dqn ...` |
| 3 | BBF-lite + position novelty | ready | `benchmark6h-bbf-novelty-20261118.yaml` | `train-dqn ...` |
| 4 | official BTR + position novelty | ready | `benchmark6h-officialbtr-novelty-20261118.yaml` | `train-official-btr --max-seconds 21600` |
| 5 | Go-Explore without human data | not implemented | | |
| 6 | evolutionary search (GA or ES) | GA scripts exist with their own goal-distance fitness; no ES; not on this reward | | |
| 7 | curriculum without a recorded trajectory | not implemented (needs a self-built savestate archive, shared with arm 5) | | |
| 8 | Dreamer-style world model | ready | `benchmark6h-worldmodel-20261118.yaml` | `train-world-model --steps 1000000 --max-seconds 21600` |

## What is the same and what differs

- Arms 1 to 4 keep the learner block, reward (0.1 per waypoint, 27.8 in total, star 50, death, damage and
  timeout 0) and environment of the earlier 1 hour projection-free arms; only the paths, the study block and the
  position-novelty bonus change. The bonus is the one used on the Center run: 0.02 per new 200x250x200 cell per
  episode, decaying with the visit count, reward-only (never a policy input).
- Arm 2 is the control for arm 1: the configs differ only in the novelty block, the paths and the study block
  (checked in `tests/test_benchmark6h_configs.py`).
- Arm 8 is **not** the alternative BTR with a model: it is the existing guarded Gaussian RSSM with an imagined
  actor/critic (`world_training.py`) on the same reward and action set. It explores with its own latent
  disagreement instead of position novelty. Every held-out quality gate is identical to
  `config/smg1-world-model.yaml`, the actor learns only after the gates pass, and real and imagined transitions
  keep separate counters; imagined transitions never count as a mission completion. A Dyna-style variant that feeds
  imagined data to the alternative BTR would be new code and is not part of this preparation.
- `train-world-model` gained `--max-seconds` (status reports `stop_reason`) so this arm has the same wall-clock
  budget as the others.

## Reading the results

One seed per arm cannot rank arms whose difference is a few points: the Center run lost about 20 points of
median route advance for 300 attempts inside a single block and recovered. Compare arms on collection attempts
and on a frozen greedy check at the end, report real decisions next to wall-clock time, and repeat only the best
two or three arms with a second seed.
