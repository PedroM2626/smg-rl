# Motor8 reward review, October 25, 2026

**Subsequent closure:** [all six remaining SMG2 evaluations](motor8-evaluation-results.md)
are complete, with zero stars. The finite pilot has closed and its hourly
follow-up is retired. Statements about pending evaluations below describe the
earlier reward-review boundary.

The owner changed experiment reviews from every ten minutes to once per hour.
Finite subprocess budgets and reviews before successors remain unchanged.
A short block may finish and wait almost an hour; hourly scheduling reduces
review overhead, not emulator time or sample needs. Do not increase a block
simply to keep the PC busy between reviews.

## Actual reward evidence

Both current eight-action world-model profiles disable legacy visual curiosity
(`use_legacy_curiosity: false`) and latent disagreement reward
(`disagreement_scale: 0`). The legacy intrinsic coefficient in YAML is inactive
for this learner. Every SMG1 bootstrap CSV records zero intrinsic payout.
Animation cannot be blamed on an active curiosity reward in this branch.

The read-only [reward audit](assets/performance/motor8-reward-audit-20261025.json)
reproduces historical route labels from all 4096 actual SMG1 bootstrap decisions:
52 full-start episodes, 48 ending in death. Nine deaths have positive
undiscounted returns; eleven have positive discounted returns at gamma 0.997.
Episode 12 in `world-model-20261025-132157-406146` earned 2.860475 for partial
prefix advancement, lost 2 for death and 0.057 in step costs, and finished at
+0.803475 without a safe initial exit or star. Prefix shaping can pay up to 5
per reset while death costs only 2.

Positive failed returns alone do not prove an inferior optimal policy or
learned exploitation: exploration may legitimately earn intermediate credit.
They expose a tradeoff needing safe-exit evaluation: partial advance followed
by a fall can be profitable. This bootstrap used a fixed prior with zero
actor/model updates; it cannot establish learned reward exploitation.

There is also a concrete route bookkeeping bug. Schema 1 advances segments
when Mario is merely within 180 units of an endpoint. The next observation
counts the full previous segment even if Mario has not moved. At XYZ
(146, 45, -295), the first update reaches 0.241680 of the prefix; an identical
second sample grants another 0.103577, or +0.517885 at weight 5. This is bounded
per waypoint, not unlimited stationary reward. It can inflate prefix reports
and reset the no-progress timer without physical advancement. Historical
58.89% prefix reports require this qualification; they were never full-course
percentages or learned successes.

## Isolated correction and remaining design work

`config/routes/smg1-initial-exit-v2-candidate.json` uses endpoint-plane crossing
instead of proximity-based advancement. Qualifying segment changes are resolved
inside one update before computing credit. An unchanged position earns zero
additional reward. Final exit requires reaching its endpoint plane and two
valid grounded observations. Tests cover stationary samples, dense traversal,
backtracking, corner shortcuts, contact, health and distinct replay signatures.
Synthetic regression fixtures are not live validation or learned completion.

The original schema-1 file, hash, profiles, replay and checkpoints are preserved.
The candidate is **not active in training**. Offline replay of actual positions
with schema 2 and unchanged weight still produces seven positive undiscounted
death returns and nine positive discounted death returns. Bookkeeping alone
does not fix reward scale, sparse successful experience or model calibration.
Counterfactual labels are not a trained A/B result: policies, future visits and
no-progress stopping times could change online.

A new reward revision should value safe route advancement and RAM-confirmed
stars while reducing reliance on this short prefix. Making death very negative
may suppress exploration. A principled shaping candidate is

\[
F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t).
\]

With consistent terminal handling and zero terminal potential, discounted
shaping telescopes to a policy-independent offset for a fixed initial state.
See [Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
The current discounted positive-high-water increments do not implement that
formula. A potential still needs suitable route state, including history if
used. Geometry cannot certify safe traversal of unvisited platforms.

Before activating such a revision, specify timeout semantics. Currently death
and completion are terminated; stalls/time limits are truncated. Model
continuation targets use terminated only. Setting zero terminal potential at
every truncation while bootstrapping across it would be inconsistent. Define
true task failures versus external collection cuts, then align shaping,
continuation targets and evaluation. No retained labels were changed here.

The current prefix also suppresses credit while HP is below its reset value.
That may suit a damage-free control fixture but suppress surviving advancement
on later hazards. There was zero HP loss in this bootstrap, so this restriction
did not cause these observed failures. The RAM HP detector and net-loss penalty
are separate from route eligibility.

SMG2 still uses a straight start-to-goal XZ projection with weight 5 rather than
an ordered, ground-validated route. It can reward movement toward the goal axis
without proving a safe landing. No validated SMG2 contact/route signal exists
in this branch: do not copy SMG1 offsets or call geometric projection a native
mission percentage. Audit SMG2 geometry separately before extending training.

## Completed blocks and decisions

The [final-block review](assets/performance/motor8-final-block-review-20261025.json)
was made after both controller PIDs exited; both error logs were empty. No
successor training was launched during this review.

| Branch | New real decisions | Model updates | Actor updates | Imagined transitions |
| --- | ---: | ---: | ---: | ---: |
| SMG1 | 4096 | 1024 | 0 | 0 |
| SMG2, both missions combined | 6096 | 636 | 134 | 129552 |

SMG1 exhausted its single permitted extra 128-cycle offline fit. Final sampled
terminal reward MAE was 1.1813 (limit 0.5); terminal continuation Brier was
0.5255 (limit 0.2). Unchanged gates reject imagination; no actor learning
occurred. These are model prediction failures, not evidence of a failed RAM
death detector. SMG1 continuation is stopped for reward/model diagnosis, with
no further fitting allocation under this pilot.

SMG2 reached its 6096 aggregate real-decision cap and passed its final sampled
prediction gates. The last 2000-decision block contained 29 deaths and zero
RAM-confirmed stars. Gates and actor updates do not establish navigation gains.
Only declared evaluations remain: seeds 11/23/47 for each mission, at most 600
decisions per attempt, all failures retained, actual policy image and applied
controls shown. No further real training is allocated. Retire the hourly
follow-up after these evaluations and report the incomplete SMG1 branch.

Reproduce the audit without an emulator or optimizer, using a new output file:

```powershell
.venv\Scripts\python.exe scripts/audit_motor8_reward.py --output NEW_AUDIT.json
```

The script preserves previous audits and rejects incomplete/duplicate bootstrap
coverage or historical route labels it cannot reproduce.

Validation for this revision: 144 portable tests passed and Ruff passed. The
protected historical route specification still matches its original SHA-256.
