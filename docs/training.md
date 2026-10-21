# Training and experiment guide

## Observation and action model

The centered 16:9 Dolphin viewport is cropped to remove most HUD content, converted to grayscale, resized to 84 by 84, and stacked over four decisions. A two-element scenario identity lets one recurrent visual policy represent both variants. No coordinate, actor pointer, frame counter, death flag, progress fraction, or success flag enters the observation.

The 54 discrete actions combine nine movement directions (including idle) with six modes: no modifier, jump, spin, crouch, crouch+jump, and jump+spin. Shift is injected before jump in combined actions. All controller inputs are released around F10 because held-input priority can suppress the emulator shortcut. Neutral game frames occur during the pause handshake. The actual elapsed game frames are logged; this desktop adapter does not claim exact action repeat.

The environment is paused between decisions, during CNN/RND/PPO updates, while writing episode plots, and on normal shutdown. Pause is checked against the scene's RAM counter, with bounded retries. A missed pause produces an error instead of silently continuing training on stale observations. Focus loss also aborts and releases inputs.

State loading also takes place while paused. This preserves the supplied near-star snapshots before a moving platform can carry Mario away. The production held-action interval is 0.25 seconds; early smoke runs used 0.10 seconds. Very short movement pulses can be mostly canceled by neutral frames in the desktop shortcut handshake.

## Novelty

Each scenario owns an independent RND target, predictor, optimizer, and running error statistics, as well as its own episodic memory. A visual inverse-dynamics encoder predicts the selected action from consecutive frames. Its frozen copy supplies kNN embeddings for the current episode; the snapshot is refreshed only at reset, so embeddings already in memory stay comparable.

Episodic novelty uses normalized squared neighbor distances and a similarity kernel, following the structure of NGU's episodic bonus. Repeated nearly identical embeddings eventually receive zero bonus. RND predictor error is normalized using previous error statistics and clipped into a lifelong multiplier from 1 to `rnd_clip`. These components are multiplied. Terminal cinematics and observations below the reward height threshold cannot collect intrinsic reward.

The encoder is learned from images, not coordinates. Rotating backgrounds can still create visual novelty; inverse-dynamics features reduce that problem but do not eliminate it. Compare the RND and episodic metrics against actual progress and independent evaluation success.

## Reward

```text
reward = progress_weight * new_valid_high_water_progress
       + intrinsic_weight * episodic_novelty * lifelong_novelty
       + completion_bonus OR death_penalty
       + step_cost
```

The weights are explicit in `config/default.yaml`. Progress has an episode high-water mark and its endpoint is calibrated separately per variation. Success requires the RAM state transition, never merely reaching a coordinate or seeing a bright object.

Episodes end on death or completion and truncate at the decision or no-progress limits. Only start states 1 and 3 are sampled during learning and evaluation. States 2 and 4 are reserved for detector calibration. This prevents a trivial near-star reset from being reported as full-course learning.

## Running longer experiments

Start with a short test using a local configuration with `max_steps: 32` and `idle_limit: 32` so both variations and reset paths are exercised. Then restore the default timeouts for real training. Use a fresh run directory for each experiment and preserve its config snapshot.

Watch `episodes.jsonl` for `best_progress`, `outcome`, and `variation`. PPO console output includes policy/value losses and separate curiosity metrics. The CSV and top-down plot make it possible to inspect repeated loops, sideways wandering, falls, and false progress. A high novelty return with no forward progress is a reason to lower the intrinsic coefficient or change the visual crop/embedding settings.

Evaluate every substantial checkpoint on both variations with intrinsic reward disabled. Ten or more episodes per variation are more informative than a single successful attempt. The two courses share their visual policy, but their novelty models and progress references remain separate. The initial scenario distribution is uniform; a curriculum can be added once the easier course is consistently solved.

The defaults are a starting configuration, not a demonstrated convergent solution. This repository has no claim of sample efficiency or full-course success. The desktop controller supports a single visible instance. The separate [background backend](background-training.md) supports isolated parallel environments and GPU learning.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Invalid executable signature | Use the verified SB4E01 executable or derive and test a new profile. |
| Unexpected scene / stage | Load a Twisty Trials starting state before attaching. |
| No render window | Boot the game with the same Dolphin build that created the states. |
| Focus error | Bring the Dolphin render window to the foreground and restart/resume. |
| Pause acknowledgment error | Check F10 and the hotkey device; release physical modifier keys. Prefer the isolated profile. |
| Savestate mismatch | Use a compatible Dolphin version and your matching state files. |
| Profile copy failure | Close the source Dolphin process and rerun `prepare`; incomplete owned copies are recoverable. |
| Game moves but observation is black | Keep Dolphin visible and unobscured; screen capture requires a usable desktop. |
| GitHub workflow publishing rejected | Keep the template in `ci/` or use a credential with workflow permission. |

## Mathematical specification and research protocol

The implementation-level equations, network shapes, constants, reward bounds and information boundary are in [methodology](methodology.md). The [experiment protocol](experiments.md) specifies independent seeds, ablations, evaluation intervals, uncertainty and throughput measurement. [Reproducibility](reproducibility.md) documents modules, file formats, configuration, checkpoint limits and failure recovery. These research procedures are documented separately from completed engineering validation; no ablation or converged full-course result is claimed. Galaxy 1 has a [separately authorized single-worker learner](concurrent-training.md); its validation configuration remains disabled for learning.
