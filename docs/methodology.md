# Methodology and mathematical specification

**Latest revision (October 26, 2026):** the active
[simple-reward IQN/IMPALA protocol](simple-reward-impala.md) specifies the
undiscounted incremental high-water reward, residual visual encoder and
attempt/decision metrics. Those equations apply only to the new signed profiles;
the earlier equations below retain their historical meaning and checkpoints.

**Regional revision (October 25, 2026):** the regional profiles
use [ordered grounded potential shaping, curated resets and frame-based termination](regional-intervention.md).
Those equations supersede the historical novelty/projection objective below only
for the new signed configurations. The finite intervention is complete; no extra
training was launched. The October 23 hold below was superseded by the owner's
later authorization for reviewed bounded experiments.

**Status note (2026-10-23):** the supervised PPO jobs described by these
equations are stopped; their final pairs are preserved. The separate visual
world-model experiment specified in [the world-model implementation](world-model.md)
and recorded in [the live world-model record](world-model-live.md) was also
stopped at the owner's request on October 23, 2026; its final checkpoints are
preserved and all training remains stopped until the owner explicitly authorizes
resuming.

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

The native backend samples a shared-memory input mask inside Dolphin's input
update. The legacy hidden backend sends controller state through an isolated
loopback DSU server. Each worker has a separate process, game-namespaced user
directory, DSU port and RAM reader. The main policy uses CUDA; each worker's
small novelty networks use CPU.
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

### Native frame-control revision (2026-10-22)

The native backend retains the action and observation spaces but replaces the
asynchronous hotkey/PNG loop with a versioned shared-memory bridge in a pinned
Dolphin 2609a build. It advances 12 newly presented frame-step boundaries per
decision and acknowledges only while paused. Direct renderer RGBA readback is
performed for the final frame; images are cropped/resized as before. Scene-nerve
counters remain diagnostic and may reset on transitions. In the live final
benchmarks, all native acknowledgments were 12 and same-nerve scene deltas matched.

This changes the temporal abstraction relative to legacy checkpoints; it is not
an exact continuation of the old control-duration distribution. No inference of
sample-efficiency superiority follows from the faster wall-clock rate. The
[protocol, benchmarks, and limitations](native-backend.md) define this backend
condition. Earlier asynchronous timing evidence in this report remains historical.

Protocol v3 replaces polling with private Windows event wakeups, moves
crop/grayscale/area preprocessing into Dolphin, and instruments each worker's
synchronous rollout barrier. Native CPU images matched 144 OpenCV reference
images exactly in the checked fixtures; experimental GPU-linear images did not.
The measured collector preserves original environment ordering, automatic
resets, terminal observations and recurrent PPO barriers. These engineering
changes do not add coordinate observations or change the reward equations.
See the [current pipeline audit](optimization-v3.md) for configuration, timing,
image-equivalence bounds and remaining worker waiting.

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

The **legacy** Galaxy 2 scalar reward is below. Galaxy 1 adds the nonfatal HP-loss term
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
larger than the 20-point success bonus. The recorded native runs later showed
near-constant bonuses on over 92% of sampled transitions. Novelty could dominate
completion; decay cannot be assumed. The current revision below addresses this
observed incentive risk.

An episode terminates on death or star collection. It truncates at 3000
decisions or 400 consecutive decisions without a progress increment above
$10^{-5}$. Savestate resets restart the high-water mark and episodic memory.
These limits are experimental choices, not game rules.

### Bounded exploration revision

The recommended native configurations now use `bounded-exploration-v2`. Let
$j_t$ count consecutive decisions without a valid progress increment exceeding
$10^{-5}$. Define $g_t=\operatorname{clip}_{[0,1]}((120-j_t)/90)$ and
$B_0=0.5$. The paid intrinsic reward is
$i_t=\min(B_{t-1},0.005g_tm_tn_t)$, with $B_t=B_{t-1}-i_t$.
The reward becomes

$$
r_t=5\Delta_t+i_t+20\mathbf1[C_t]-2\mathbf1[D_t\land\neg C_t]
-0.001-2\mathbf1[T_t].
$$

$T_t$ denotes truncation at 180 stalled decisions or 3000 total decisions,
excluding death/completion. Galaxy 1 retains its nonfatal damage term. The
intrinsic budget and stall counter reset per episode. Progress clears the stall
counter but cannot replenish the budget. Thus total paid novelty is at most
0.5 per episode; it cannot accumulate indefinitely by rotating, jumping in
place, or waiting. Timeout is an environment penalty, not a fabricated RAM
death. These gates use RAM progress only for reward calculation; the observation
remains pixels and scenario. They may discourage temporarily unproductive but
necessary detours, especially on curved routes, so full-start evaluation remains
essential. The [audit](optimization-v2.md) reports counterfactual reward replay,
not proof of improved policy learning.

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
Legacy settings update on one transition at a time. Native settings score each
eligible transition immediately under no-grad, enqueue the two uint8 images,
action and fixed RND target, then average both losses over eight transitions
before one optimizer step. A partial batch flushes before the next episode's
frozen encoder is refreshed. Pending samples and optimizer/update counters are
checkpointed; old checkpoints default to an empty queue. No samples are dropped.
This reduces optimizer frequency rather than reproducing eight sequential Adam
steps, and is a changed learning condition. There is no replay sampling or
shared global novelty model across workers. A preallocated circular array
retains the same bounded kNN memory rule without rebuilding a Python-list stack
on every decision.

Inverse dynamics biases features toward action-related information, but does
not prove invariance to rotating scenery, camera motion or animation. Action
prediction can also be ambiguous on blocked or uncontrollable frames. RND error
measures predictor difficulty, not progress or semantic novelty.

## 5. Recurrent PPO

The current implementation uses `sb3-contrib` 2.9.0's `MultiInputLstmPolicy`
(the earlier protocol-v3 evidence used 2.6.0):
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
[versioned implementation documentation](https://sb3-contrib.readthedocs.io/en/v2.9.0/modules/ppo_recurrent.html).

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
