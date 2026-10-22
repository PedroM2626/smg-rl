# Methodology and mathematical specification

## Abstract

This project implements visual, recurrent on-policy reinforcement learning for
two Twisty Trials scenarios in Super Mario Galaxy 2. It combines episodic visual
novelty and Random Network Distillation (RND) with coordinate-derived progress,
death penalties and a RAM-verified star-collection bonus. The policy observes
pixels and a scenario identity. Privileged RAM telemetry supplies supervision
and diagnostics only. This is an engineering research implementation; calibration
and throughput measurements do not establish learned full-course success.
Super Mario Galaxy 1 uses a separately authorized single-scenario learner; its
independent configuration and initial concurrent measurements are documented in
[concurrent training](concurrent-training.md). The equations below describe the
shared algorithm, while the hyperparameter table specifies the Galaxy 2 run.

## 1. Problem definition and information boundary

Let the hidden game state be $s_t$, action $a_t$, scenario $v$, observation
$o_t$, and recurrent policy state $h_t$. The task is partially observable:

$$
o_t=(I_{t-3:t},e_v),\quad
(a_t,h_{t+1})\sim\pi_\theta(\cdot\mid o_t,h_t),\quad
J(\theta)=\mathbb E[\sum_{t=0}^{T-1}\gamma^t r_t].
$$

$I$ is a cropped, grayscale 84 by 84 image; four consecutive decision images
form a channels-first uint8 tensor. $e_v$ is a two-dimensional one-hot scenario
vector for Galaxy 2. Coordinates, velocities, death/completion flags, object
pointers, state-machine values and computed progress are excluded from $o_t$.
Galaxy 1 uses a one-element scenario vector; current/max HP are also excluded.
The code enforcing this boundary is `GalaxyEnv._observation` in `env.py`.

This is **visual control with privileged reward supervision**, rather than
purely unsupervised learning or a pixels-only reward system. Its outcomes should
not be compared directly with an intrinsic-only benchmark without describing
the additional geometric and terminal supervision.

Galaxy 2 training starts are slots 1 and 3. Slots 2 and 4 are calibration fixtures only.
Galaxy 1 trains from slot 1 and reserves slot 2 for detector checks.
Fixed worker assignments alternate scenarios, so four workers allocate two to
each variation. An odd worker count creates a sampling imbalance; episodes per
scenario can also differ because episode lengths differ.

## 2. Control and temporal abstraction

There are 54 discrete actions: nine movement directions, including idle,
combined with six modes: neutral, jump, spin, crouch, crouch+jump, jump+spin.
The action index is `9 * modifier_index + direction_index`; `actions.py` defines
the exact order. Simultaneous controls do not encode an entire timed long-jump
macro, so the agent must learn useful action sequences.

The recommended backend sends controller state through an isolated loopback DSU
server. Each worker has a separate process, user directory, DSU port and RAM
reader. The main policy uses CUDA; each worker's small novelty networks use CPU.
The desktop backend instead uses the configured W/A/S/D, Shift and mouse inputs.

The hidden backend aims for 12 scene frames per action at 2x emulation speed.
Pause polling, hotkey delivery, input release and screenshot flushing add
variable frames. Four stacked observations span **decisions**, not exactly four
video frames. The process is therefore closer to a variable-duration control
process; the implementation nevertheless uses a fixed discount per decision.
It does not implement time-adjusted semi-Markov discounting.

The scene counter measures the current nerve's lifetime and resets at state
transitions. Differences across different scene nerves are not valid elapsed
frame counts. CSVs retain both fields so this can be inspected. Screenshot
requests are accepted only for newly written complete PNGs; they come from the
renderer, not an obscured desktop window. A hidden rendering surface is still
required; this is not a renderer-free headless simulator.

## 3. Progress and extrinsic reward

Let $p_t=(x_t,z_t)$, calibrated start $b_v$ and near-star point $g_v$. Define:

$$
q_t=\operatorname{clip}_{[0,1]}
\frac{(p_t-b_v)^\top(g_v-b_v)}{\lVert g_v-b_v\rVert_2^2}.
$$

Let $G_t$ require finite coordinates, $y_t\geq-1800$, no death, a previous
sample, and a 3D displacement no greater than 2500 world units. The production
Galaxy 2 configuration sets these thresholds. Then:

$$
\Delta_t=\mathbf1[G_t]\max(0,q_t-H_{t-1}),\qquad
H_t=\begin{cases}\max(H_{t-1},q_t)&G_t\\H_{t-1}&\text{otherwise}.\end{cases}
$$

On reset $H_0=q_0$. Thus revisiting an already rewarded projection produces
no further progress reward within that episode. This is a geometric proxy,
not a discovered native percentage. It is not guaranteed to be potential-based
shaping or to preserve the optimal policy of a sparse-reward task.

The Galaxy 2 scalar reward is below. Galaxy 1 adds the nonfatal HP-loss term
specified in the final section, and suppresses novelty on HP-loss transitions:

$$
r_t=5\Delta_t+0.02\,m_t n_t+
20\mathbf1[C_t]-2\mathbf1[D_t\land\neg C_t]-0.0001,
$$

where $C_t$ and $D_t$ are the game-specific completion and death predicates,
$n_t$ is intrinsic novelty, and $m_t$ is zero on death, completion, or below the
configured reward height, otherwise one. Completion takes precedence if both
terminal flags were ever true. Novelty and progress have distinct validity gates:
the progress teleport gate does not automatically gate intrinsic reward.

For an episode starting at zero progress, total progress reward is at most 5.
Intrinsic novelty is bounded above by 5 per decision, giving at most 0.1 reward
per decision. At the 3000-decision limit the loose intrinsic upper bound is 300,
larger than the 20-point success bonus. Actual bonuses decay, but this reveals a
real incentive-design risk: novelty can dominate completion. Weight tuning and
evaluation success must decide whether the combination is useful.

An episode terminates on death or star collection. It truncates at 3000
decisions or 400 consecutive decisions without a progress increment above
$10^{-5}$. Savestate resets restart the high-water mark and episodic memory.
These limits are experimental choices, not game rules.

## 4. Intrinsic reward implementation

The design follows the structure of episodic novelty in
[Never Give Up](https://arxiv.org/abs/2002.06038) and lifelong prediction error in
[RND](https://arxiv.org/abs/1810.12894). It does not reproduce NGU's distributed
Q-learning, UVFA exploration-policy family or meta-controller. The base learner
here is recurrent PPO with one fixed intrinsic coefficient.

### 4.1 Visual features and inverse dynamics

Each scenario/worker owns an encoder $f_\phi$, a linear inverse classifier
$d_\psi$, RND target $F_\xi$ and predictor $\widehat F_\eta$. Encoders have
Conv(1,16,k8,s4), ReLU, Conv(16,32,k4,s2), ReLU, adaptive average pooling to
4 by 4, flattening, Linear(512,64) and tanh. Inputs are single grayscale images
divided by 255, rather than policy frame stacks.

The action-prediction objective is:

$$
L_{\rm inv}=-\log d_\psi(a_t\mid[f_\phi(I_t),f_\phi(I_{t+1})]).
$$

A copy $\bar f$ is frozen at episode reset. All episodic embeddings in that
episode use this copy; the trainable encoder continues to change. This prevents
stored vectors from becoming incomparable merely because the encoder changed.
There is no pretraining dataset or coordinate input.

### 4.2 Episodic nearest-neighbor bonus

Let $z=\bar f(I_{t+1})$ and let $d_i$ be the up to ten smallest squared Euclidean
distances to episode memory. Let $\bar d$ be their mean. A distance scale is
initialized with $\max(\bar d,10^{-6})$, then updated by
$s\leftarrow0.99s+0.01\bar d$. With $k\leq10$:

$$
u_i=\max(d_i/\max(s,10^{-6})-0.008,0),\quad
K_i=\frac{10^{-4}}{u_i+10^{-4}},\quad
S=\sqrt{\sum_{i=1}^{k}K_i}+0.001.
$$

$$
b_{\rm epi}=\begin{cases}0&S>3\\\min(1,S^{-1})&S\leq3.\end{cases}
$$

An empty memory returns 1. Every scored embedding is appended; the oldest is
removed at capacity 4000. Reset inserts the initial observation. The bonus is a
kernel similarity heuristic, not a probability density estimate with calibrated
units. Repeated frames eventually get zero bonus. Memory and distance scale
restart on every episode.

### 4.3 Lifelong RND multiplier and learning

The random target $F_\xi$ never receives gradient updates. The scalar error is:

$$
e_t=\operatorname{mean}[(\widehat F_\eta(I_{t+1})-F_\xi(I_{t+1}))^2],
\quad
b_{\rm life}=\operatorname{clip}_{[1,5]}
\left(1+\frac{e_t-\mu_{t-1}}{\max(\sigma_{t-1},10^{-4})}\right),
\quad n_t=b_{\rm epi}b_{\rm life}.
$$

Running moments use Welford's algorithm and sample variance from past errors;
the estimator is updated after computing the multiplier. The minimum multiplier
is one, so learned familiarity cannot make it suppress episodic novelty below
its baseline. A single Adam optimizer updates encoder, classifier and predictor
on $L_{\rm inv}+e_t$, with learning rate $10^{-4}$ and gradient norm clipped to 1.
Updates use one transition at a time. There is no replay buffer or shared global
novelty model across parallel workers.

Inverse dynamics biases features toward action-related information, but does
not prove invariance to rotating scenery, camera motion or animation. Action
prediction can also be ambiguous on blocked or uncontrollable frames. RND error
measures predictor difficulty, not progress or semantic novelty.

## 5. Recurrent PPO

The implementation uses `sb3-contrib` 2.6.0's `MultiInputLstmPolicy`:
the default combined image/vector extractor, 128-unit LSTM state, and separate
128-unit actor/value MLP heads. Policy image normalization is handled by SB3.
The tested library defaults include PPO clipping 0.2, value coefficient 0.5,
maximum gradient norm 0.5 and normalized advantages. Explicit project settings
are in `config/parallel.yaml`.

For the policy probability ratio $\rho_t=\pi_\theta(a_t\mid o_t,h_t)/
\pi_{\theta_{old}}(a_t\mid o_t,h_t)$, PPO maximizes:

$$
L^{\rm clip}=\mathbb E[\min(\rho_t\widehat A_t,
\operatorname{clip}(\rho_t,1-0.2,1+0.2)\widehat A_t)].
$$

The implementation adds entropy regularization and the value loss. Generalized
advantage estimation uses temporal-difference residuals
$\delta_t=r_t+\gamma V_{t+1}-V_t$ with terminal masking and
$\widehat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$.
Timeouts use SB3's truncation handling. Recurrent state is masked at episode
boundaries. See [PPO](https://arxiv.org/abs/1707.06347),
[GAE](https://arxiv.org/abs/1506.02438), and the
[versioned implementation documentation](https://sb3-contrib.readthedocs.io/en/v2.6.0/modules/ppo_recurrent.html).

| Parameter | Hidden parallel configuration |
| --- | ---: |
| Workers | 4 |
| Rollout decisions per worker | 128 |
| Aggregate rollout size | 512 |
| Minibatch size | 256 |
| Optimization epochs | 4 |
| Learning rate | 0.0001 |
| Discount / GAE lambda | 0.997 / 0.95 |
| Entropy coefficient | 0.01 |
| Training seed | 42 |
| Policy device / worker Torch threads | CUDA / 2 |
| Requested new decisions | 1,000,000 |
| Periodic checkpoint interval | 1024 aggregate decisions |

Rollouts can round the request upward. Resuming counts `--steps` as additional
decisions. Mid-rollout checkpoints contain policy/optimizer state and curiosity
state but not an exactly replayable partially collected rollout.

## 6. Measurement, reliability and interpretation

Terminal detection follows game-specific object graphs and nerve pointers;
it does not classify death from black frames or completion from star color.
Exact layouts and evidence levels are documented in
[Galaxy 2 memory](memory-profile.md) and [Galaxy 1 validation](smg1.md).
The reader requests no game-memory write permissions.

Sampling multiple RAM fields is not an atomic emulator snapshot. Pausing before
normal reward sampling reduces inconsistency. Read-time ID, code, pointer,
stage and scene checks reject unsupported states, but an eight-byte signature is
a layout guard, not a cryptographic guarantee of executable equivalence.

The initial four-worker long run failed at timestep 1944 after a pause
acknowledgment error. A 45 ms frame-stability check could misclassify a brief
render stall. The current detector samples six times at 30 ms intervals and
requires two distinct changes in the scene counter/nerve tuple before returning
early as running. This rejects a single queued screenshot step as evidence of
sustained advancement. It reconciles measured state before toggling and allows
six attempts with 120 ms pause pulses and 35 ms release intervals.
This trades some throughput for reliability; it does not establish immunity to
OS stalls. Failures remain explicit, with owned child cleanup and periodic
paired checkpoints. The optional recovery supervisor retries from completed
pairs and maintains the original target; it does not repair invalid game
profiles or eliminate control failures. No synthetic transition is substituted
for a failed step.

The historical 7.9 decisions/second benchmark predates this stability change.
It must not be presented as the revised backend's measured long-run speed.
FPS, aggregate policy decisions, per-worker decisions and wall-clock time are
different quantities. Report the measured status values for each experiment.

The following are not established: optimal novelty coefficients, superiority
over RND alone, a solved full course, generalization to unseen stages, native
route completion percentage, exact deterministic replay, or a scientifically
meaningful success rate from the calibration fixtures. The
[experiment protocol](experiments.md) defines how to test these claims.

## Galaxy 1 damage reward revision

For the separately authorized Galaxy 1 learner, the scalar reward additionally
includes $-0.25\max(0,h_{t-1}-h_t)$ on nonfatal, non-completion transitions.
Terminal death uses only the existing death penalty. Health is reset as a tracker
baseline after each savestate load, and healing gives no positive bonus. Novelty
is suppressed on sampled HP-loss transitions. Current/max HP are privileged
reward/logging telemetry, not policy features. This is a new reward condition;
previous checkpoint continuation is not an independent training seed. See
[measured damage evidence and sampling limits](damage-and-recovery.md).
