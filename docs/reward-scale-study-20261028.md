# Bounded SMG1 reward-scale comparison

Owner-authorized October 28, 2026. The study compares the most recent
expanding-route IQN checkpoint with its 749,006-decision source through frozen
canonical-start recording, then tests one reward-normalization hypothesis.
There is no world model, curiosity or imagined transition. All original assets,
source checkpoints, optimizer state and replay remain untouched.

## Frozen-policy baseline

Record both policies with exactly the same eight-action control, initial game
state, 600-decision cap and video instrumentation. A single greedy episode is
followed by six stochastic episodes with action RNG seeds 11, 23, 47, 101, 149
and 197, NoisyNet, random quantiles and an explicit matched epsilon of 0.1.
Seed labels do not vary the restored Wii state. Training and evaluation decision
counts are separate. These small samples are exploratory and do not establish
generalization or causal attribution.

Both recordings completed with matching epsilon and unchanged source agent
hashes. The greedy source reached 100% of the mapped opening then died; the
greedy expanding-route checkpoint reached 1.84% before timing out. Across six
stochastic action draws, mean opening advance was 61.47% for the source and
13.62% for the expanding-route checkpoint, with six deaths and six timeouts
respectively; neither collected a RAM star. The corresponding mean grounded
goal-axis projections were 3.21% and 1.60%. These are not native mission
completion percentages. The expanding-route branch changed its reward scale,
start mixture, replay and optimizer, so this comparison does not isolate the
cause. [Per-episode results](assets/performance/scale-source-frozen-comparison-20261028.json)
and local videos preserve every trial.

## Reward audit and matched learning study

The signed expanding-route objective uses `normalization_distance=50000`, which
makes the 2186.142-unit opening worth about 0.0437 reward; death or timeout is
-2. The candidate changes only the denominator to 5000 and pays ten times as
much for a fixed route displacement, at most about 0.4372 for the current
opening. Death, timeout, damage, completion, route, controls, model, starts and
replay settings are unchanged. The candidate may saturate at 5000 units: this
study fixes the current opening and does not promote route extensions. It is
not a complete-course solution or policy-invariant shaping guarantee.

The [offline audit](assets/performance/reward-scale-audit-20261028.json)
covered 666 completed and excluded 18 partial attempts. Mean returns were
-1.992 for 491 deaths and -1.995 for 175 timeouts; no credited transition had
identical sampled XYZ, was airborne or was dead. A tenfold counterfactual on
the same traces gives mean returns -1.918 and -1.947 respectively, without
assuming that an agent trained under the alternative would choose the same
actions. Other incentive failures and unobserved transitions remain possible.

## Intermediate-state check and assisted probe

All three existing curriculum states passed the old alive/grounded/HP check
for 60 idle frames on October 28. The separate 250-unit frontier displacement
screen reveals displacements of 3.11, 452.00 and 477.69 world units for right
approach, first pink and first chocolate, respectively. The latter two drift
on conveyors; the measurement does not prove they are unsafe, but they cannot
be described as spatially stationary. The [drift evidence](assets/performance/curriculum-idle-drift-20261028.json)
retains the initial and final coordinates. The existing bank and old acceptance
criteria remain unchanged; the validator now reports drift separately.

A bounded [assisted-start probe](assets/performance/assisted-frontier-probe-20261028.json)
restored first chocolate and ran one greedy and three epsilon-0.1 noisy frozen
source-policy attempts, each capped at 200 decisions. All four died after
24–44 decisions, with no recorded route-prefix exit or star. They start from
the assisted snapshot, not the canonical full start, and do not demonstrate
learned course completion. These attempts do not establish that all actions or
policies fail at this anchor. No extra training or replay was used.

Control and candidate each start from the same frozen source weights with a
fresh optimizer, target, RNG and separately signed replay. Two paired training
seeds 142 and 143 each receive 6144 real decisions per arm, at most 900 seconds,
with six canonical-start workers. One block at a time is dispatched and must
pass checkpoint/replay/reward/attempt accounting before its frozen evaluations
and before the next dispatch. Every frozen evaluation has one greedy seed 11 and
three stochastic action seeds 11/23/47 at matched epsilon 0.1, up to 600
decisions each. All deaths, cuts and evaluation actions remain recorded.

For this exploratory screening, the candidate must exceed its same-seed
control's stochastic mean opening advance by at least five percentage points
for both independently initialized training seeds, without fewer canonical RAM
stars in either pair. This heuristic gate is not a significance test or a
promise of full-level improvement. Compare goal-axis projection, deaths, idle
and action occupancy as secondary outcomes. Random action seeds are not
independent Wii states. Run no automatic longer training after this screen; a
review must first consider sampled outcomes, candidate regressions, source
integrity and validation limits.

`scripts/run_reward_scale_ab.py` creates distinct immutable local profiles and
replays, records pending commands before dispatch, and stops on any failure or
partial budget without retry. The output ledger is
`.local/reward-scale-ab-20261028/state.json`. Each block owns at most six
emulators; tests, fixes and independent prior data must not be mixed into the
study results.

## Completed matched comparison

All four blocks completed and passed the recorded accounting and reward audits.
Each collected 6,144 real decisions and performed 1,022 optimizer updates; in
total there were 24,576 training decisions and 184 complete training attempts.
No world-model or imagined transitions were used. All four signed checkpoint
and replay pairs, plus the original source pair, passed SHA-256 checks. The
sixteen frozen evaluation episodes used an additional 577 real decisions; all
started from the canonical state, ended in death and recorded zero RAM stars.
The agent-file hashes used for evaluation matched the final checkpoint hashes.

| Training seed | Control stochastic opening mean | Candidate stochastic opening mean | Candidate minus control |
| --- | ---: | ---: | ---: |
| 142 | 76.72% | 80.83% | +4.11 percentage points |
| 143 | 73.20% | 10.52% | -62.68 percentage points |

The candidate did not meet the preregistered five-percentage-point threshold
in either pair; seed 143 also produced a substantial regression. The gate
therefore failed and **no larger training block was started**. In the greedy
episodes, both controls and candidate seed 142 reached the opening end and
died; candidate seed 143 reached 10.52% and died. The reported percentages are
of the mapped opening, not the full level. Three noisy action draws from the
same restored game state are not independent environments, and two training
seeds and one short budget do not establish a general conclusion about reward
scales. The results are a reason not to extend this particular candidate as
tested, not proof that a stronger progress reward can never help.

The [machine-readable audit](assets/performance/reward-scale-ab-results-20261028.json)
contains per-episode outcomes, signatures, hashes, locally retained videos and
separate training/evaluation counts. The state ledger and runs remain local;
the ROM, savestates, replays, checkpoints and MP4 files are not committed.
