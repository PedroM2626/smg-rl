# Training and experiment guide

**Current learner (October26):** [model-free Rainbow with optional IQN](model-free.md)
replaces the active RSSM experiment. Use its explicit profiles and TensorBoard;
older PPO/world-model workflows below remain historical references.

**Current status (October 25, 2026):** staged development superseded the October 23
hold below. The latest [regional reward/curriculum/control intervention](regional-intervention.md)
is complete, with all processes stopped at finite boundaries. Use the new regional12
profiles for future isolated pilots; frame4 remains experimental. Historical
checkpoint reward heads/replay must not be resumed under the new signatures.

**Status note (2026-10-23):** the supervised PPO jobs described here are stopped;
their final pairs are preserved. The separate world-model experiment recorded in
[the live world-model record](world-model-live.md) was also stopped at the
owner's request on October 23, 2026; its final checkpoints are preserved and
that dated hold was superseded by the later staged authorization above.

**Native backend update (2026-10-22):** the local learners now use
`config/native.yaml` and `config/smg1-native.yaml`, with shared-memory RGBA
capture and bounded frame stepping. The stock hotkey/PNG backend remains a
fallback. See [native controls, profiling, benchmarks, and migration](native-backend.md).
The older measurements below retain their original backend/timing context.

## Operating workflow

Use the [README quick start](../README.md) for installation and hidden supervised
training, the [CLI reference](cli.md) for exact arguments, and
[concurrent operation](concurrent-training.md) for two independent game policies.
This guide explains control/reward choices and experimental practice.

## Observation and action model

The 16:9 viewport is cropped to remove most HUD content, converted to grayscale,
resized to 84 x 84, and stacked over four decisions. Galaxy 2 has a two-element
scenario identity; the independent Galaxy 1 policy has one element. No coordinate,
actor pointer, scene counter, health value, terminal flag, or computed progress
enters the observation.

The 54 discrete actions combine nine movement directions, including idle, with
six modes: neutral, jump, spin, crouch, crouch+jump, and jump+spin. Hidden workers
send controller state through private DSU servers. The desktop fallback injects
Shift before jump in combined actions and releases inputs around F10. Held-input
priority can otherwise suppress the shortcut.

Both backends have asynchronous timing. Hidden workers target 12 scene frames
at 2x emulation speed; the desktop interval is 0.25 seconds. Pause/input/capture
overhead makes achieved duration variable, so neither claims exact action repeat.
Logs retain the scene counter and nerve: differences across nerve changes are
not elapsed-frame measurements. Very short desktop pulses can be canceled by
neutral frames during shortcut delivery.

Workers pause for observation processing, learning, and episode output. Pause
checks use RAM with bounded retries. A missed acknowledgment aborts the attempt.
The current hidden probe requires two distinct counter/nerve changes to classify
running, and pause pulses last 120 ms with a 35 ms release interval. Hidden
workers close on shutdown; the desktop adapter leaves its window paused. Desktop
focus loss aborts/releases inputs. Hidden inputs/capture need no foreground focus.

## Novelty and reward

Each worker/scenario has independent RND target/predictor, optimizer, running
statistics, and episodic memory. A visual inverse-dynamics encoder predicts the
action from consecutive frames. An episode-frozen copy provides kNN embeddings;
it is refreshed at reset so existing episodic embeddings remain comparable.

Episodic novelty uses normalized squared neighbor distances and a similarity
kernel. RND error is normalized using previous running statistics and clipped to
a multiplier from 1 to `rnd_clip`. The components are multiplied. Repeated nearly
identical embeddings eventually receive zero episodic bonus. Terminal frames,
frames below the reward height, and Galaxy 1 sampled HP-loss transitions cannot
collect novelty. Rotating backgrounds can still produce unhelpful visual novelty;
inverse-dynamics features do not guarantee its elimination.

```text
reward = progress_weight * new_valid_high_water_progress
       + bounded_and_stall_gated_intrinsic_reward
       + completion_bonus OR death_penalty
       + nonfatal_damage_penalty_if_applicable
       + step_cost
       + timeout_penalty_if_truncated
```

Recommended native settings are progress 5, intrinsic 0.005 with an episode cap
of 0.5, completion +20, death -2, step -0.001, and timeout -2. Novelty fades
linearly between 30 and 120 consecutive decisions without valid progress; an
episode truncates after 180 such decisions or 3000 total decisions. The
[optimization audit](optimization-v2.md) explains this reward revision and the
separate curiosity optimizer's batch of eight. Legacy configurations retain
intrinsic 0.02, step -0.0001, and their previous limits.
Galaxy 1 additionally charges -0.25 per nonfatal HP lost, once per
sampled decrease. Healing/reset do not earn rewards, and a fatal transition uses
only the death penalty. This is a net-health detector and can miss a hit followed
by healing between samples. Galaxy 2 retains death-only penalties for the two
selected missions. See [live health evidence](damage-and-recovery.md).

Progress is a gated high-water geometric X/Z projection onto a separately
calibrated start-to-near-star axis, not a native percentage. Its approximation
is especially limited on Sweet Sweet Galaxy's curved route. Full completion
requires the game-specific RAM event. Episodes terminate on death/completion and
truncate at decision/no-progress limits. Galaxy 2 trains/evaluates only from
slots 1/3, reserving 2/4 for detector checks; Galaxy 1 uses start 1 and fixture 2.

## Longer experiments and evaluation

Exercise collection, reset, both Galaxy 2 variations, and stop/resume in a short
pilot before a long experiment. A local configuration with `max_steps: 32` and
`idle_limit: 32` can test truncation/reset paths; restore full limits afterward.
Use a fresh run and preserve configuration/source provenance per experiment.

Inspect per-worker `episodes.jsonl`, episode CSVs, `live.csv`, and top-down plots.
Current `status` progress resets per episode; `report` audits available historical
maxima. High novelty with little valid progress suggests inspecting the crop,
embedding, reward balance, and repeated routes before scaling the budget.

The default recommendation is `train-supervised`: it recovers infrastructure
failures from paired checkpoints while preserving the target. `train-parallel`
is a single attempt. Observe retry logs; repeated startup failures can indicate
bad assets or profiles that retry cannot repair. Follow
[checkpoint semantics](reproducibility.md), rather than treating policy-only
emergency snapshots as full resumable pairs.

Evaluate substantial checkpoints with deterministic actions and intrinsic reward
and curiosity updates disabled. The current `evaluate` command uses the visible
adapter, so shut down parallel learners before running it. Use the checkpoint's
game configuration and at least ten episodes per variation for a pilot; rigorous
comparisons need the multi-seed protocol below. A continuation after reward/timing
changes is not an independent seed. No converged solution is claimed.

## Troubleshooting

| Symptom / backend | Action |
| --- | --- |
| Invalid executable or stage / either | Match the verified USA game profile and configured start states; revalidate replacements. |
| Savestate mismatch / either | Use the compatible Dolphin build and matching game/state files. |
| Verified RAM missing on boot / hidden | Inspect attempt stderr and owned Dolphin log; allow the 90-second boot window and five-second Galaxy 2 worker spacing. |
| Pause acknowledgment or frame stall / hidden | Inspect worker logs and controller configuration; supervised retry preserves completed pairs, but repeated faults need investigation. |
| Stale/black PNG / hidden | Inspect the owned renderer/capture logs; desktop visibility is irrelevant to this backend. |
| Focus error or black desktop capture / visible | Keep the render window foreground and unobscured, then resume from a pair. |
| Pause shortcut failure / visible | Check F10/hotkey device and release physical modifiers; held inputs can suppress the shortcut. |
| Profile copy failure / either | Close the source Dolphin process before copying and inspect the owned-profile marker; preserve the original profile. |
| Wrong learner selected by status/stop | Select the game configuration explicitly and, if needed, pass its `parallel-*` attempt directory. |
| Repeated supervised attempts with no progress | Inspect `attempt-*.stderr.log`, validate assets/configuration, and stop the owning supervisor while fixing the cause. |
| Workflow publishing rejected | Keep the template in `ci/` until a publishing credential has workflow permission. |

## Mathematical specification and research protocol

[Methodology](methodology.md) specifies equations, network shapes, constants,
reward bounds, and the information boundary. The [experimental protocol](experiments.md)
specifies independent seeds, ablations, evaluation, uncertainty, and throughput
measurement. [Reproducibility](reproducibility.md) specifies modules, outputs,
configuration, and recovery limits. These planned research procedures are
separate from completed engineering validation; neither algorithm superiority
nor full-course learning success follows from smoke tests.
