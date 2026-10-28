# Human-route reward audit and matched IQN comparison

This owner-authorized experiment compares the existing opening reward with a
landing-audited route derived from the owner's complete Sweet Sweet run. The
metric is a fraction of that one human route, not a native completion
percentage. Native full-start star events are identified separately from RAM.

## Reward audit and revision

Schema 4 allowed an airborne waypoint visit to be credited after landing back
at the start. A synthetic two-waypoint reproduction produced 0.5 route credit
without maintaining that advance. Schema 5 retains schema 4 unchanged for
historical signatures, but commits pending crossings only after a grounded
landing near the last crossed waypoint: the horizontal tolerance is the
smaller of 450 units and 45% of that segment's length, with the existing height
tolerance. A failed landing discards uncommitted crossings. Death and
completion never pay route credit, large position discontinuities invalidate the
episode, and committed credit is a non-repeating high-water mark. Health loss
remains a separate -0.25 penalty per HP; the route can still credit grounded
advance when damage occurs. These local tests cannot rule out all shortcuts,
moving-platform drift or adverse reward incentives.

The [version 2 specification](../config/routes/smg1-human-landing-v2.json) and
[experimental profile](../config/human-route-smg1-landing-v2.yaml) are distinct
from all original profiles and replays. Re-scoring the recorded human path at
all six 12-frame phases gives 98.4% to 100% of this mapped path. This is reused
source data and not an independent test. The full-route normalization caps
progress reward at +1; death or timeout is -2, a nonfatal HP is -0.25, and a
native star is +20. The reward per metre is materially smaller than the
opening-only objective. Route end is not completion.

## Frozen baseline and matched study

The source is the audited `749006`-real-decision IQN/IMPALA checkpoint. A new
greedy and three epsilon-0.1 stochastic frozen episodes from the canonical start
all ended in death, with zero RAM stars. The schema-5 full-route score was zero
for all four: they did move, but never passed the first waypoint under this
strict metric. The action RNG labels restore the same Wii savestate. The
original checkpoint and replay were retained, and these are evaluation
decisions, not training.

The preregistered screen uses seeds 142 and 143, control and candidate arms,
6,144 real decisions per arm with six collectors and a 900-second wall cap per
block. Both arms copy the same online weights, synchronize targets and use
fresh optimizer and separately signed fresh replay; both use canonical starts.
Every block is reviewed before the next starts. Each checkpoint is evaluated
with one greedy and three stochastic full-start episodes, at most 600 decisions
per episode, epsilon 0.1. All results are scored using the same schema-5 route,
and native stars, deaths, HP and videos are recorded separately.

The exploratory gate requires the candidate's mean stochastic route fraction
to exceed its paired control by at least 0.005 in **both** training seeds,
without fewer RAM stars in either pair and without operational or integrity
failures. It is not a statistical significance test. If the gate fails or is
inconclusive, do not start a longer block. A passing gate requires a separate
review before the owner-authorized, single bounded longer block.

The first launch failed before collection because profiles were created one
directory too deep, so `load_config` resolved `data` beneath `.local`. Its
retained failed run records zero real decisions and zero updates. The repaired
supervisor uses profiles directly under `.local`, verifies the workspace root
and state file before dispatch, and retains the failed artifacts separately.
No training data or results were reused from the failed launch.

Study state: `.local/human-route-ab-fixed-20261028/state.json`.
The local controller owns four sequential finite blocks; never duplicate an
active/completed block or retry a partial failure blindly. ROMs, states,
recordings, replays and checkpoints are not committed.

## Post-hoc re-scoring of previous frozen evaluations

The retained reward-scale and chocolate-demonstration studies contributed 32
reviewed canonical frozen episodes in total, with four paired cases in each
study. All 32 received zero under the schema-5 human-route metric, and none had
a native RAM star. This does not mean the policies stayed at the start: the
highest separately recorded goal-axis projection was approximately 7.45%, a
geometric projection rather than a native mission percentage or a distance
along the route; this old generic maximum is not restricted to grounded
samples. The first human waypoint requires a sufficiently close ordered
visit and landing; these past policies moved without satisfying that condition.

This floor effect means the route metric alone may be too coarse to distinguish
early policies, and a zero-versus-zero current A/B is inconclusive rather than
evidence that either reward is equally effective. Historical re-scoring uses
existing trajectories and is not an unseen test. It does not change the
preregistered current gate or retroactively validate an old candidate. The
[episode-level evidence](assets/performance/human-route-historical-frozen-20261028.json)
contains source state and trajectory hashes, goals and failures. Future studies
should preregister additional interpretable early-progress measurements and
their geometric limitations before looking at their results.

## Completed comparison

All four branches completed their 6,144-decision budgets and passed checkpoint,
replay, reward and episode accounting. Each performed 1,022 optimizer updates.
Across branches there were 24,576 training decisions, 191 completed attempts,
163 deaths, no nonfatal HP loss and zero full-start RAM stars. The failed
pre-dispatch attempt described above collected zero decisions and is excluded.

The 16 frozen evaluations required 796 additional real decisions: 15 ended
in death and one in timeout, with no HP loss or RAM stars. All scored zero on
the shared schema-5 route, including both stochastic means in each seed pair.
The preregistered gate failed. This zero-versus-zero comparison is inconclusive
because the metric has a severe floor effect; it does not establish that the
candidate reward is ineffective or that the policies did not move. No larger
training block was dispatched.

For a secondary, post-hoc diagnostic, the best alive-grounded X/Z projection
toward the near-star point was 5.73% in a candidate-142 stochastic evaluation.
This straight-line projection is neither native mission completion nor a
distance along the winding route. Source and final checkpoint hashes, reward
signatures and evaluation agent hashes were checked. The
[episode-level result](assets/performance/human-route-ab-results-20261028.json)
contains separate training/evaluation counts, videos, trajectories, HP and
hashes. All action seeds restore the same Wii state. Two training seeds, a short
budget, the coupled change of reward scale and route, and post-hoc secondary
metrics limit causal and generalization claims.
