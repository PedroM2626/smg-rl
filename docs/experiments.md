# Experimental protocol and evidence register

## Research questions

1. Does visual episodic novelty plus RND improve full-course success compared
   with the same recurrent PPO learner using progress and terminal rewards alone?
2. Does the shared visual policy learn both Twisty Trials variations, rather
   than concentrating on the easier one?
3. How much wall-clock acceleration remains after accounting for instance
   startup, screenshot synchronization, learning, failures and checkpoint I/O?

These are proposed experiments. The short engineering tests reported below do
not answer the learning questions.

## Task definition and leakage prevention

Galaxy 2 scenario 1 starts exclusively from `SB4E01.s01`, and scenario 2 from
`SB4E01.s03`. The episode ends at the first RAM-confirmed collection or death;
the near-star states are never included in training or evaluation reset sampling.
The evaluation policy receives the same image/scenario observation as training.
Intrinsic rewards and curiosity updates are disabled in evaluation.

The initial test states may have been recorded during an opening animation.
Keep the same state bytes across compared methods. Changing a state or waiting
different amounts before control begins changes the task distribution.
Full-course success requires collection after a start-state reset. Calibration
success after a near-star reset is a separate detector test.

Galaxy 1 is a separately authorized exploratory training pilot. Its start state
is used for training and its near-star state exclusively for detector validation.
Do not pool its results with the controlled two-scenario Galaxy 2 comparison.

## Controlled comparisons

Use at least five independently initialized training seeds (for example
42, 43, 44, 45, 46) per condition, identical assets, emulator, crop, action set,
reward-height gates, episode limits and compute budget. Save a copy of every
condition's YAML. Fix both aggregate decision budget and report actual elapsed
time. The following are planned conditions, not completed results:

| Condition | Change relative to the current implementation |
| --- | --- |
| Extrinsic baseline | Set intrinsic coefficient to zero; optionally disable curiosity computation in a separate measured implementation |
| Episodic only | Hold the lifelong multiplier at one; requires an explicit ablation switch/code change |
| RND only | Use normalized RND novelty without episodic multiplication; requires a separate implementation |
| Current combination | Episodic kNN times clipped RND multiplier |
| No inverse features | Fixed random episodic features; requires a separate implementation |
| Sparse terminal baseline | Zero progress and intrinsic coefficients |

Do not silently replace methods mid-run. Record any additional implementation
changes as a new condition. Setting intrinsic reward weight to zero currently
does not skip novelty computation; its timing is not a valid optimized baseline.
Use separate directories and clean initial model states for seed comparisons.
Resuming the exploratory smoke checkpoint is useful operationally but is not an
independent seed from random initialization.

## Evaluation schedule and metrics

Evaluate every 50,000 aggregate decisions, and at the final checkpoint. These
intervals are a proposed research schedule; the learner does not automatically
launch evaluation. Use at least 30 episodes per scenario per checkpoint and
keep deterministic and stochastic evaluation results separate. The supplied
`evaluate` command uses deterministic actions on the desktop backend.

```powershell
.\.venv\Scripts\smg-rl.exe --config config/default.yaml evaluate runs/parallel-EXAMPLE/model-final.zip --episodes 30
```

The desktop evaluation launcher expects no other Dolphin instance. Stop the
parallel learner gracefully and use a saved checkpoint before this command;
otherwise use an isolated future evaluation backend with explicit PID scoping.
Checkpoint loading should select `device: cpu` for desktop evaluation unless a
matching CUDA configuration is desired.

Report results separately for each scenario, then report a balanced mean over
scenarios. Log success count/attempt count, death rate, timeout rate, episode
length, best valid projection, final projection, extrinsic return, intrinsic
return, time to first full-course success, total decisions, and wall-clock time.
The current evaluation JSON contains success and best progress; other metrics
must be derived from the corresponding trajectory CSVs. There is no built-in
automatic significance test or bootstrap evaluator.

Success is a binary collection event. For $k$ successes in $n$ evaluation
episodes, report $\hat p=k/n$ with a Wilson 95% interval, using $z=1.96$:

$$
c=\frac{\hat p+z^2/(2n)}{1+z^2/n},\quad
w=\frac{z}{1+z^2/n}
\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)},\quad [c-w,c+w].
$$

Repeated deterministic resets from an identical state are correlated. Their
interval is descriptive and does not replace independent training seeds or a
broader held-out initial-state distribution. For method comparisons, bootstrap
over training seeds, preserving each seed's pair of scenario results, rather
than treating every episode as an independent trained agent. Report all seeds,
including failures; avoid selecting only the highest-return checkpoint after
seeing test performance. See
[Agarwal et al.](https://arxiv.org/abs/2108.13264) for motivation for uncertainty
reporting in deep RL.

## Throughput protocol

For each worker count 1, 2 and 4, use the same backend revision and policy state.
Warm up shader compilation separately, then measure a fixed additional decision
budget over at least ten minutes when practical. Include checkpointing and
optimization in the operational rate. Report startup separately and distinguish
aggregate decisions/second from per-worker rate and rendered FPS. Repeat at
least three times; give mean, spread, hardware and thermal/power mode.

Increasing `frame_skip` changes the control problem and cannot be treated as a
free throughput optimization in a sample-efficiency comparison. Increasing
`emulation_speed` can alter overshoot from asynchronous command delivery.
Re-run detector calibration and compare achieved action durations after either
change. A future native bridge/shared framebuffer can reduce polling and PNG
overhead but is not implemented here. No distributed multi-machine trainer or
renderer-free Dolphin build is included.

## Evidence register as of 2026-10-21

| Experiment | Evidence | Established result | Does not establish |
| --- | --- | --- | --- |
| Galaxy 2 detector calibration | `config/calibration.json`; `runs/background-validation-20261021-175431-396382/` | Both stars, falls, movement and pause passed | Learned full-course success |
| Original desktop smoke | `docs/assets/*-smoke.csv` and PNGs | Real visual RL and measured trajectories | A converged policy |
| Two-worker CUDA smoke | Local run/checkpoint logs | Roughly 4.6 aggregate decisions/s in the short test | Long-run scaling |
| Original four-worker CUDA smoke | `runs/parallel-20261021-175102-110584/` | 512 new decisions in 64.94 s; model reached 768 total | Sustained 7.9 decisions/s after later changes |
| Graceful stop/resume | `runs/parallel-20261021-175601-308500/` | Stopped at 1140 total; policy and curiosity saved | Exact deterministic continuation |
| Initial long run | `runs/parallel-20261021-175922-132631/` | Failed at 1944 on a pause acknowledgment error | Reliable unattended training |
| Intermediate pause implementation | Unit regression tests and the subsequent live run | Longer stability window; measured-state reconciliation | Elimination of all possible OS stalls |
| Current pause/recovery revision | `docs/damage-and-recovery.md`; portable regression tests | Two counter changes required, longer pause pulses, paired-checkpoint supervised continuation | Failure-free operation or exact replay |
| Galaxy 1 health | `docs/assets/smg1/damage-validation.json` and HUD screenshot | Live 3-to-2 HP transition and one-time damage penalty on reward replay | Learned damage avoidance or detection of every between-sample hit |
| Galaxy 1 validation | `config/smg1-calibration.json`; `docs/assets/smg1/` | Independent star/death/movement/pause detection passed | Training or a solved course policy |

Local run directories are ignored by Git. Small measured CSV/PNG detector
fixtures and calibration JSON are retained to make the engineering evidence
reviewable without redistributing game assets. No full-course learned success
is claimed by this report.

## Threats to validity

Novelty return can reward animation or camera changes; a high intrinsic return
need not imply progress. The straight-line projection can reward shortcuts in
coordinate space and undercount curved legal routes. Death detection covers
live-tested abyss states plus explicitly identified symbol-derived states;
untested death modes should not be described as empirically verified.

Host scheduling changes effective action durations. A fixed per-decision
discount does not normalize those durations. Parallel novelty is local to each
worker, so visits in one do not reduce another worker's lifelong bonus. More
workers therefore change both throughput and exploration dynamics.

Savestate resets constrain the initial distribution and can conceal failures on
normal game loading. Frame crops may omit useful hazards/HUD information.
Independent test stages and additional start states are required before claiming
generalization. The small engineering measurements and one policy seed per game
(with checkpoint continuations) do not justify claims of algorithm superiority
or publication-grade performance.
