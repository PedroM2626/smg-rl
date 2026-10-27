# Frozen IQN checkpoint evaluation and goal-distance candidate

On October 27, 2026, the owner requested frozen-policy evaluation of the latest
valid SMG1 checkpoint and an assessment of an objective based on proximity to
the goal. This is an evaluation and offline metric test; no learning or new
training data were produced.

## Policy and integrity

The checkpoint `runs/model-free/smg1-six-two-hour/dqn-20261027-114555-611503/checkpoint-final`
contains 749,006 cumulative real decisions and 186,739 optimizer updates.
Its `agent.pt` and `replay.npz` matched the manifest SHA-256 before and after
the evaluations. `agent.pt` contains the saved optimizer state as well as
policy weights; no optimization was called. Both recordings report zero added
training decisions and zero imagined transitions.

The IQN uses a trainable IMPALA-style residual visual encoder with a stack of
four grayscale 84 × 84 images and a scenario one-hot vector. It has no
recurrent state. In the video, the inset shows the **newest frame** of the
four-frame policy input; it is not a visualization of all four stacked frames.
The overlay shows raw expected Q values and applied controls, not probabilities.

## Results

Each episode started from the canonical SMG1 slot 1 with a maximum of 600
decisions. The deterministic mode used epsilon 0, fixed midpoint IQN quantiles
and no parameter noise. The stochastic mode used noisy layers, random quantiles
and the saved policy's training epsilon of 0.1. Seed labels 11, 23 and 47 do not
change the restored Wii savestate or establish independent game RNG.

| Mode | Label | Decisions | Outcome | Grounded projection | Closest-distance reduction | Opening fraction |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Greedy | 11 | 58 | Death | 4.6902% | 4.5536% | 100% |
| Greedy | 23 | 58 | Death | 4.6902% | 4.5536% | 100% |
| Greedy | 47 | 58 | Death | 4.6902% | 4.5536% | 100% |
| Noisy + epsilon | 11 | 56 | Death | 4.6902% | 4.5536% | 100% |
| Noisy + epsilon | 23 | 27 | Death | 2.8391% | 2.8127% | 49.40% |
| Noisy + epsilon | 47 | 30 | Death | 3.9585% | 3.8377% | 93.61% |

All six episodes ended in death, with **zero RAM-confirmed stars**. The three
greedy action sequences and trajectories were identical. Their actions per
episode were forward 24, right 5, left 5, spin 21 and jump 3; they reached the
end of the mapped opening at decision 29, then reversed course, spun and died.
Each greedy episode received +1.0 regional shaping and -2.0 death, return -1.0.
The stochastic episodes used 8, 3 and 1 epsilon-random decisions respectively.
Three stochastic episodes are not enough to establish comparative success rates.

The two percentages above are **different geometric diagnostics**. Projection
is the maximum alive-grounded X/Z projection along the initial-to-goal axis.
Distance reduction is the minimum alive-grounded X/Z Euclidean distance to the
near-star goal, compared with the initial distance. Neither is native mission
progress or a measured fraction of the winding course. The 100% opening
fraction covers the short mapped prefix only.

The actual local recordings and trajectories are retained in:

- `runs/model-free/smg1-six-two-hour/eval-dqn-video-20261027-175202-073857/`
- `runs/model-free/smg1-six-two-hour/eval-dqn-video-20261027-175242-481141/`

Each contains `policy.mp4`, `recording.json` and episode CSVs. The first video
has 1,044 encoded frames and the second 678, matching the recorded six frames
per 12-frame decision. The local runs and videos are ignored by Git. See the
[aggregated evidence](assets/performance/iqn-frozen-goal-distance-20261027.json)
and [top-down paths](assets/performance/iqn-frozen-trajectories-20261027.png).

## Existing reward, genetic fitness and candidate

The active IQN reward pays newly confirmed advance along a predefined,
grounded opening route; the goal projection is used for diagnostics, with a
zero weight in that IQN configuration. It is therefore not a direct reward for
getting closer to the star. The earlier genetic fitness is the best grounded
projection onto the start-to-goal axis. It does not require a recorded route,
but projection is not Euclidean distance to the goal and can increase for
off-course motion. Trajectory CSVs are retained for verification and graphs,
not required by the online genetic tracker to assign its fitness.

At the time of this evaluation, `GroundedGoalDistance` in
`src/smg_rl/goal_distance.py` was an **offline-only prototype** for the user's
proposed alternative. It credits each new closest
alive-grounded X/Z distance once, relative to the actual episode reset, with a
finite-position/discontinuity guard. Its scale is normalized to the canonical
start-to-goal distance; assisted starts do not receive credit merely for being
closer at reset. It needs no predefined course route. The evaluation replayed
the six recorded traces through this candidate without changing their behavior;
these values do **not** establish that a policy trained with it would improve.
A later [genetic experiment](genetic-goal-distance-20261027.md) explicitly uses
full XYZ distance for its separate online fitness; IQN reward/replay are unchanged.

Geometric distance can reward an unreachable shortcut and can punish a
necessary detour or a route passing above/below the goal; X/Z ignores height.
An observed closer airborne or falling state is deliberately not credited by
the prototype. Before adopting it in training, compare a separately signed
configuration and fresh replay against the existing reward under matched
budgets and canonical frozen evaluation. Preserve the current checkpoint,
signatures and failure penalties. Do not combine this hypothetical change with
other algorithm or control changes when testing its effect.
