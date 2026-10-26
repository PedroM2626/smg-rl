# Model-free Rainbow, IQN and TensorBoard

**Comparison completed:** Rainbow and fresh IQN each reached 8192 real decisions
per mission and completed all nine frozen evaluations. Neither collected a star
or demonstrated useful frozen-policy navigation. All training/evaluation processes
are stopped, with no successor allocated. TensorBoard remains available for
history. The CLI default retains the IQN profile for explicitly requested future
work; it does not imply a running job or demonstrated superiority.

On October 26, 2026 the owner selected model-free distributional Q-learning after
the finite RSSM/world-model pilot. PPO was already a historical alternative; the
actual replaced learner was a Gaussian RSSM with imagined actor/critic updates.
Neither is constructed, fitted or used for action selection by the new pipeline.
There are no imagined frames, continuation/reward predictors, model-quality gates
or synthetic transitions in this learner. Original checkpoints and gate definitions
remain retained and unchanged for historical reproduction.

## Scientific choice and scope

[Rainbow (Hessel et al.)](https://arxiv.org/abs/1710.02298) combines six DQN
extensions and reports strong data efficiency on Atari. Its replay can reuse
expensive real Dolphin experience; it does not make emulation itself faster or
guarantee success in these 3D Mario tasks. This project selected Rainbow as the
first online baseline. [IQN (Dabney et al.)](https://proceedings.mlr.press/v80/dabney18a.html)
learns a quantile function rather than a fixed categorical return support and is
available as a separate configured alternative.

This is an original PyTorch implementation, not a drop-in port or reproduction
of Atari benchmark results. The [official DQN Zoo reference](https://github.com/google-deepmind/dqn_zoo)
was consulted as context; no Apache/JAX source was copied into this MIT code.
These variants have explicit Mario-specific observation, reward, warmup and
update schedules. Published Atari rankings do not establish a ranking against
PPO or the prior RSSM in this project.

## Algorithms and exact targets

`src/smg_rl/distributional.py` and `dqn.py` implement:

| Component | Rainbow | IQN alternative |
| --- | --- | --- |
| Double Q | Online network selects bootstrap action; target network evaluates it | Same |
| Dueling heads | Per-atom value plus advantage minus mean action advantage | Per-quantile value/advantage |
| Prioritized replay | Proportional sum/min tree; alpha 0.5 | Same |
| Importance correction | Global minimum-probability normalized weights; beta 0.4 to 1 | Same |
| Multi-step returns |3 real decisions, boundary-aware shortened suffixes | Same |
| Parameter exploration | Factorized NoisyLinear, resampled for acting and optimization | Same |
| Distribution |51-atom C51, support[-10,25] |32 sampled quantiles, 64-dimensional cosine embedding |

Thus the IQN option is **IQN with Rainbow-style double/dueling/PER/n-step/noisy
additions**, not a replication of unmodified original IQN. Both are model-free.
Neither uses recurrence: four stacked frames supply short temporal context;
longer camera/platform history can still be partially observed.

For an emitted k-step transition:

```text
R = sum(gamma**i * real_reward[i], i=0..k-1)
D = 0 on absorbing termination; otherwise gamma**k
a* = argmax_a E[Z_online(s_next,a)]
```

Rainbow projects `R + D * support` from the target distribution for `a*` onto the
fixed support. Exact integer bins retain their full mass. Its weighted cross-
entropy loss uses PER importance weights; priorities use distributional KL plus
epsilon, not entropy-containing cross entropy. IQN uses pairwise quantile Huber
loss, summed over prediction quantiles and averaged over target samples; its PER
priority is mean absolute pairwise TD error plus epsilon. IQN's action selection
averages uniform quantile samples for risk-neutral expected returns. Evaluation
uses fixed midpoint quantiles and disables NoisyLinear parameter noise.

The categorical support is an explicit experimental range that includes the +20
star and observed negative outcomes, not Atari's clipped-reward convention or a
guarantee that every possible return is bounded there. Rewards are **not clipped**.
`learning/support_clipped_mass` exposes Bellman mass outside the support. Nonzero
mass requires review before changing the support and its checkpoint identity.
IQN avoids a fixed categorical support.

Common initial settings: gamma 0.997 per 12 physical frames; Adam learning rate
0.0000625 with epsilon 0.00015; batch 32;512-transition warmup; one optimizer update
per 4 aggregate real decisions; target synchronization every 500 optimizer updates;
gradient norm clipped at 10. Beta reaches 1 at 100000 real decisions. Collection is
uniform random during warmup; afterward NoisyNet exploration is mixed with an
explicit epsilon 0.01. No old RSSM/PPO weights are transferred. Settings are saved
with checkpoints rather than silently inferred.

## Observation, reward and boundaries

The four 84x84 grayscale gameplay frames and scenario one-hot are the only neural
inputs. No coordinates, HP, region IDs, start type, episode counter or RAM terminal
pointer enters Q prediction. SMG1 has one scenario; SMG2 has two. The first pilot
keeps eight exclusive physical actions and the validated twelve-frame 8-on/4-off
jump/spin control. Changing frame interval is a separate experiment.

The signed [regional potential, HP/death and curated starts](regional-intervention.md)
are held fixed across learners. Curiosity and old projection reward are zero.
The `world_model` configuration section is absent; regional gamma validation uses
`dqn.gamma`. `novelty.enabled=false` prevents construction of curiosity modules.
The maps still cover only verified openings, not entire missions. Switching the
learner cannot fill missing route geometry, supply unseen successes or eliminate
partial observability.

There is a remaining reward-design hypothesis to examine separately: with
gamma 0.997, step cost -0.001 and a -2 failure penalty, an otherwise stationary
failure at decision 180 has return about -1.307, versus about -1.937 for dying at 14.
Potential shaping closes to zero discounted total from canonical failed starts,
so it does not remove this preference for delaying failure. An early stationary
health policy is therefore not evidence that novelty still exists or that a
different Q algorithm will automatically solve exploration. Keep reward fixed
for this first learner comparison; if final evaluations still stall, test the
time/failure objective and more informative validated starts in isolated finite
interventions instead of assuming that a longer run must fix them.

Per-worker n-step queues flush at a real episode boundary, never across workers
or resets. The actual `terminal_observation` supplies the final pixels after
automatic reset. RAM death/completion and the declared regional idle/task budget
are absorbing and remove bootstrap. An ordinary time-limit truncation retains
bootstrap. Collector/checkpoint cuts shorten pending n-step targets but keep
bootstrap; they do not invent deaths. Portable tests exercise each case.

## Replay, useful episodes and persistence

`PrioritizedFrames` stores each new grayscale frame once and reconstructs stacks
through generation-checked previous-frame links. Resets pad with their own first
frame; history cannot cross episodes. The frame ring reserves enough space for
two observations per decision plus active stacks/n-step prefixes. A stale frame
ID fails closed. The initial capacity is 50000 transitions per game, about 0.71GB
of maximum allocated frame storage per learner, rather than duplicated state and
next-state four-frame stacks. Actual resident use initially grows with touched
frames. Capacity is checked against configured and current RAM budgets.

PER emphasizes experiences with larger prediction error. It is **not an elite
buffer that keeps only high-return episodes**: deaths and surprising bad outcomes
are useful learning samples too. Capacity remains bounded and oldest entries are
eventually overwritten. Episode CSV/trajectory files, per-action collector ledger
and immutable block checkpoints retain auditable history. Each current 8192-per-
mission pilot fits entirely in its own retained final replay checkpoint; a FIFO
priority mechanism must not be described as permanent best-play retention.

Checkpoints are new directories containing `agent.pt`, compressed `replay.npz`
and a SHA256 manifest. They include online/target weights, optimizer, CPU/CUDA
and action/replay RNG states, frame generations, priorities and real/update counts.
Publication uses an incomplete marker; incomplete, corrupted, incompatible or
overwriting restores fail. Replay contains only real labeled transitions.
Checkpoints are local and ignored by Git. Restoring starts new environment
episodes rather than pretending to restore an interrupted emulator's runtime.

## TensorBoard and measured stages

[PyTorch SummaryWriter](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
writes event files below `runs/model-free/tensorboard/ALGORITHM/GAME/RUN`.
The local-only server is at **http://127.0.0.1:6006/**. It requires no TensorFlow
training runtime. The server's optional-TensorFlow warning is expected.

Logged data include episode return/length and RAM stars split by mission and
`full_start`/`curriculum`; optimizer/target update counts; loss, gradient norm,
Q values, PER priorities/beta/effective sample size; support clipping; replay
size/allocated bytes; deaths, HP loss, actions, available RAM; real decisions and
frames per second. Timings separate inference, emulator collection, Q optimization
and checkpoint work. Actual input-frame stacks are logged as images.

Axes use real decisions, never optimizer batches or imagined steps. IQN offline
smoke logs use their own clearly named optimizer-update axis. TensorBoard exposes
metrics; it does not certify mission completion or backfill old PPO/RSSM runs.
The initial neural throughput alone cannot predict end-to-end Mario learning time.

Each collector action ledger stores the real count, optimizer age, actions,
worker episode and episode step. This permits later separation of warmup/prior
collection from behavior after Q updates. A best collection trace is still not
a frozen-policy evaluation.

## Actual validation and staged dispatch

All mathematical/collector tests passed before the longer online dispatch.
[Actual Rainbow health evidence](assets/performance/model-free-health-20261026.json)
records six simultaneous collectors,512 decisions each: 1024 SMG1 decisions with
128 optimizer updates;2048 SMG2 decisions with 383 updates. Replay reconciled
exactly, losses/gradients were finite, no C51 support clipping was observed, and
TensorBoard scalar/image events were read back successfully.

Three frozen health-policy video fixtures then ran from canonical starts, each
with seed 11 and at most 96 decisions. Sweet Sweet stayed at its starting position
and reached that evaluation budget. Spinning/Double Time died after 14/18 decisions.
No star was collected. These are successful **implementation checks**, not
navigation or superiority results. Videos show the exact grayscale actor input,
raw expected Q returns (not probabilities), selected action and applied controls
every two frames. Encoder frame counts, stable input digests and 8-on/4-off button
timing passed sidecar checks. Full-RGBA recording uses the established OpenCV
gray/crop/resize path; collection uses native CPU capture, so evaluation records
that capture-path distinction rather than assuming byte-identical preprocessing.
The movie inset displays the latest actual grayscale frame from the four-frame
observation, not the entire temporal stack. The network still receives all four;
TensorBoard's stack image shows all four collection frames together. Sidecar
input digests refer to the displayed latest frame.

[IQN optimizer validation](assets/performance/model-free-iqn-smoke-20261026.json)
performed 64 finite updates per game on **copies** of the real Rainbow health
replay. It added zero physical decisions and zero imagined transitions, modified
no source checkpoint, and produced finite losses/gradients and IQN checkpoints.
This reused-data smoke is neither unseen validation nor an online Rainbow-vs-IQN
comparison; IQN was not silently given six additional emulators.

After reviewing health, the selected online pilot resumes those compatible
Rainbow checkpoints with two Sweet Sweet workers and four SMG2 workers, two per
variant. The predeclared cumulative training caps are 8192 new real decisions per
mission:8192 aggregate SMG1 and 16384 aggregate SMG2, including health. Additional
budgets are 7168 and 14336 respectively. No old world-model job is resumed. Local
authority, exact run/controller identities, logs and source/target counts are in
`.local/model-free-state.json`. At most six owned project emulators may overlap.

After both finite blocks finish, review before any successor. The declared final
evaluation is frozen canonical-start video, seeds 11/23/47, at most 300 decisions
per mission per seed, with all failures retained and no optimizer calls. Training
and evaluation counts stay separate. No additional online block or offline fit
is automatically allocated. The hourly follow-up `revisar-piloto-rainbow` reports meaningful findings,
failures or completion in Portuguese, stays quiet while healthy, and retires
after the allocated pilot and evaluations. Original source states and near-star
detector fixtures never establish full-start learning success.

## Commands

Both finite algorithm pilots are completed. Inspect state/status before executing
commands; do not duplicate active workers. IQN occupies the six collectors after Rainbow stopped, rather than adding
another six concurrently. Both trial allocations remain finite.

These statements describe the completed dispatch order; no collector remains active.

```powershell
# Reference for a NEW bounded Rainbow block:
.venv\Scripts\python.exe -m smg_rl.cli --config config/rainbow-smg1.yaml train-dqn --steps 1024
# Replace the config with iqn-smg1.yaml or the corresponding smg2 profile for a deliberate alternative.
# Resume only a compatible model-free checkpoint DIRECTORY:
.venv\Scripts\python.exe -m smg_rl.cli --config config/rainbow-smg1.yaml train-dqn --steps 1024 --resume runs/PATH/checkpoint-final
# Frozen full-start evaluation, including failures and budget cuts:
.venv\Scripts\python.exe -m smg_rl.cli --config config/rainbow-smg1.yaml record-dqn runs/PATH/checkpoint-final --variation 0 --seeds 11 23 47 --max-steps 300
# Dashboard, only if the existing server is not already listening:
.venv\Scripts\python.exe -m tensorboard.main --logdir runs/model-free/tensorboard --host 127.0.0.1 --port 6006
```

`status` and graceful `stop` now recognize `dqn-*` runs in the selected profile's
run directory. `report` includes only `model_free_run=online` collections and
excludes assisted starts from full-start success. Offline smoke and video roots
are excluded. RGB observations, episode metadata, best paths and Q return values
must not be reported as native RAM mission percentages or learned stars.


## Completed Rainbow result and reviewed IQN comparison

[The audited Rainbow result](assets/performance/model-free-final-20261026.json)
contains exact checkpoint/replay hashes, scenario counts, collection outcomes,
TensorBoard tags and all nine frozen-policy recordings. Real counts were 8192
SMG1 and 16384 SMG2 (8192 per scenario). The networks performed 1920 and 3967
optimizer updates respectively, with zero imagined transitions and no runtime
errors. C51 clipping was zero in all logged observations. No training or
frozen-evaluation episode collected a star.

Main-block aggregate throughput was 10.08 decisions/s for SMG1 and 18.68 for SMG2,
including resume/startup and final checkpoint work. Earlier RSSM blocks measured
7.60/10.73. These are descriptive wall-clock observations under different
initialization/visitation/update workloads, not controlled learning-efficiency
ratios. Q updates averaged about 52.3/50.4ms, while emulator collection barriers
averaged 155.1/143.6ms per vector step. The environment is still a major cost.
Optimizer-update counts and replayed sample counts are never real-frame savings.

The final Rainbow frozen policies completed seven idle timeouts and two
300-decision evaluation cuts. Sweet Sweet had zero grounded opening progress
in every seed and used spin for 521/540 decisions. Spinning/Double Time had zero
opening progress for seeds 11/23 and 0.848% of the short mapped opening for seed 47;
this is not mission completion percentage. Their spin counts were 432/660 and
650/660 respectively. The behavior occurred with zero visual novelty and no
world model, so novelty is not a necessary cause of spinning. The delayed-failure
return preference, exploration and Q calibration remain plausible contributors;
this pilot does not identify one sole cause.

A collection episode during Rainbow health reached the end of the **short
Spinning opening map**, but the frozen final policy did not reproduce that
behavior. Collection maxima must not be substituted for evaluation or reported
as learned star completion. All failures and movies are retained.

After reviewing those outputs, a second **fresh online IQN** pilot was manually
allocated within the owner's Rainbow-and/or-IQN request and continuation. It
uses the same per-game training seed 142/42,512-decision warmup, reward, controls,
starts, health budget 512 per collector and cumulative 8192 real decisions per
mission. The first health counts are 1024/2048 and the reviewed remainders are
7168/14336. It does not resume the earlier reused-data IQN smoke checkpoint or
inherit Rainbow replay, parameters or readiness. Thus each algorithm receives
its own equal 8192-per-mission allocation; both together total 49152 new training
decisions across three missions, plus separately counted frozen evaluations.

Only one algorithm's six collectors may be active. IQN health must pass the
same numerical, frame-stack, checkpoint, counter and TensorBoard checks before
its remaining block starts. After both IQN blocks complete, only its declared
three-seed frozen evaluations remain; no indefinite successor is allocated.
The hourly follow-up reads the current algorithm and actual controllers from
state and must never rerun the completed Rainbow blocks/evaluations. One paired
training seed per game is an exploratory comparison, not three independently
trained seeds; three evaluation seeds do not establish statistical superiority.


## Final IQN result and comparison closure

[IQN completion evidence](assets/performance/model-free-iqn-final-20261026.json)
reconciles 8192 SMG1 and 16384 SMG2 real decisions, 1920/3967 optimizer updates,
3/7 target synchronizations and zero imagined transitions. Replay scenario counts,
episode decisions and per-action ledgers match exactly. No runtime training
error, RAM star or nonfatal HP loss occurred. All 302 terminal failures passed
potential closure (maximum error about 1.17e-15), with 12 nonterminal collector
cuts retained separately. No reward, model parameter or optimizer was changed
during frozen evaluation.

IQN's nine final episodes produced six idle timeouts and three 300-decision
budget cuts, with zero RAM stars. Sweet Sweet used spin 656/660 times; each SMG2
mission used none 657/660 times. Every mission had zero grounded opening advance
for evaluation seeds 11/23. Seed 47 reached 1.139% of the short Sweet Sweet opening
and 0.848% of each SMG2 opening. These tiny excursions occurred in the explicit
1% epsilon evaluation and do not show useful navigation, a full-course percentage
or independently replicated learning. Videos, exact image digests, raw Q values,
button timing and encoded frame counts were checked and retained.

The [paired summary](assets/performance/model-free-comparison-20261026.json)
and [world-path/action-occupancy figure](assets/performance/model-free-comparison-20261026.png)
compare all three frozen seeds for each mission. Main-block observed throughput:

| Game | Rainbow real decisions/s | IQN real decisions/s |
| --- | ---: | ---: |
| SMG1 |10.08|13.44|
| SMG2, aggregate across both scenarios |18.68|23.02|

IQN's Q update means were 21.14/20.08ms versus Rainbow 52.28/50.40ms in this
implementation/machine. Workloads, visits and runtime conditions differ: these
observations support lower measured overhead here, not a general algorithm
ranking or proven sample-efficiency benefit. Neither final policy solved a
mission; faster counters did not translate into demonstrated navigation.

One parallel Spinning evaluation failed before emulator creation because two
`new_run` calls received the same Windows clock timestamp. No episode or game
decision was executed by that failed allocation. Both completed other missions
and the original stderr were retained. Run allocation now atomically falls back
to a random suffix on name collision, with a frozen-clock regression test.
Only the three previously unexecuted Spinning seeds were retried, with the same
300-decision cap; the successful retry is recorded in local state. No completed
training/evaluation budget was duplicated or extended.

The useful next diagnosis is the time/failure objective, exploration and Q
calibration rather than an automatic longer run. The explicit return calculation
above shows why delaying failure can be preferable among failed behaviors.
The observed spin/idle policies in both algorithms, with zero curiosity and no
world model, make visual novelty an insufficient sole explanation. They do not
prove a single root cause or that either learner cannot succeed after a better
validated task design and larger controlled study.

This finite comparison is closed. No additional fitting, training or algorithm
trial was started. The hourly follow-up was removed after the final report;
TensorBoard was deliberately left available for inspecting retained history.
The regression suite passed 182 tests and Ruff. All original checkpoints, ROMs,
states and profiles remain preserved; commits contain only source/docs/config
and numeric or project-authored figure evidence.
