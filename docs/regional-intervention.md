# Regional rewards, curated starts and separate temporal control experiment

**Subsequent authorization:** the owner then requested [six concurrent regional
collectors](regional-six-training.md). That separate finite run supersedes the
two-emulator limit below for its own dispatch only. This completed validation
record and its no-training accounting remain unchanged.

The owner authorized these three changes on October 25, 2026. They form a new
bounded intervention after the previous pilot closed. Original checkpoints,
states, ROMs, reward identities and quality gates remain retained. The local
authority is `.local/regional-intervention-state.json`.

1. Validate game-specific contact and ordered opening regions; use frame-discounted
   potential shaping with zero absorbing-terminal potential. HP loss remains a
   separate nonfatal penalty. Declared stall/time budgets are true task failures,
   while collector interruptions remain nonterminal cuts.
2. Create reachable intermediate snapshots from original starting slots 1/3.
   Accept only undamaged, grounded states that pass two 60-frame idle probes and
   exact restoration. Near-star slots 2/4 remain detector fixtures. A training
   reset has at least 50% canonical-start probability; full-start evaluations force canonical
   starts independently of the training mixture.
3. Only after reviewing those results, compare temporal control under fixed
   emulated-frame budgets. Retain eight logical labels. Carried jump direction
   and full button hold are separate declared hypotheses before comparing 12
   versus 4 frames with matching semantics. Legacy 12-frame 8-on/4-off semantics
   remain available unchanged.

Preparation is limited to 6000 physical frames per mission. Timing attempts use
seeds 11/23/47, at most 1200 physical frames per attempt and predeclared conditions.
At most two project emulators may overlap. No indefinite training or automatic
extension is granted. Scripted, assisted and untrained-policy outcomes never
establish learned full-start completion. Coverage of an opening route is not a
native mission percentage or a claim that the full course has been mapped.

## Completed outcome and deployment boundary

All three stages are complete. No learner fitting, actor updates, imagined
transitions or online training decisions were added. The new local banks contain
three SMG1 starts, four SMG2 Spinning starts and two SMG2 Double Time starts.
All 36 frozen-policy attempts completed their finite limits. None collected a star.
The optional 4-frame policy inference advanced less than matched 12-frame inference;
the twelve-frame baseline remains unchanged. Eight encoded videos passed input/
button-timing checks. [Top-down paths](assets/performance/regional-control-20261025.png)
show the best of the three seeds per condition, including failed trajectories.

The CLI defaults to `config/regional-smg2-12.yaml`; the independent Galaxy 1 profile
is `config/regional-smg1-12.yaml`. Both retain the old eight physically exclusive
controls and 12-frame 8-on/4-off jump/spin timing. New action timing is available
only through explicit experimental profiles/overrides. Historical `motor8-*`
profiles retain their original reward and checkpoint identities.

These are **opening-route** specifications, not complete-course reward maps.
SMG1 covers the initial platform through the first chocolate approach. SMG2
covers the initial platform and first obstacle approach. The accepted state
originally named `beyond-obstacle` did not clear that obstacle; its published
bank name is `obstacle-jump-approach`, with the original provenance retained.
No full mission percentage or obstacle-clearance result is inferred from these
fixtures. Extend and validate subsequent regions before claiming dense reward
coverage of the full course.

## Reward definition and failure correction

Only verified RAM coordinates, contact, HP and terminal states affect reward.
Pixels and scenario identity remain the only policy inputs. Visual curiosity,
world-model disagreement novelty and the old straight-line progress coefficient
are zero in the new profiles. Moving scenery, repeated jumping and spinning
therefore cannot earn visual novelty.

Schema3 resolves endpoint crossings in the same update. Mere proximity to a
waypoint does not advance the region and produce credit on a later stationary
frame. Updates require the ordered segment's horizontal corridor, interpolated
height tolerance and RAM-confirmed ground contact. Dead/complete/airborne samples
cannot increase route progress. Best confirmed arc distance is monotonic; revisits
cannot pay again. Nonfatal damage does not permanently disable valid later
progress: HP loss is a separate penalty.

Let `d` be confirmed opening arc distance in world units, including the correct
initial region for an assisted start. The profiles use:

```text
Phi = 5 * min(1, d / 50000)
gamma_frame = 0.997 ** (1/12) = 0.9997496555897012
gamma_decision = gamma_frame ** actual_advanced_frames
F = gamma_decision * Phi_next - Phi_previous
reward = F + step_cost + completion_or_failure + nonfatal_damage
```

The 50000-unit normalizer is an explicit engineering scale, not an identified
RAM percentage or measured course length. It prevents a short prefix from paying
as though it were an entire mission. Extending the mapped route does not rescale
previous arc-distance targets. Shape is bounded by5 across that distance, rather
than awarding5 for the short initial platform exit.

Completion pays +20; RAM death and frame-budget failure pay -2. SMG1 nonfatal HP
loss pays -0.25 per point; a fatal transition receives the death penalty without a
second damage penalty. The time cost is -0.001 per12 emulated frames, proportionally
scaled to -0.0003333333 per4. Gamma is 0.997 at 12 frames and 0.99899899833 at 4;
both PPO and world-model settings use that physical-time discount. The environment
rejects mismatched model gamma or unverified frame counts.

For RAM completion/death **and** declared idle/episode failures, `Phi_next=0`.
Idle failure is 2160 physical frames without new grounded route distance; maximum
episode length is 36000 physical frames. These are absorbing task failures, marked
`terminated=True`, not bootstrap truncations. Collector cuts remain nonterminal:
closing a process or saving a partial chunk must not invent a failure label.
The 1200-frame control comparison temporarily uses its smaller explicit limit.

With `D_t` the cumulative frame discount, potential rewards telescope:

```text
sum(D_t * F_t) = -Phi_start + D_terminal * Phi_terminal
```

A closed failed episode has zero discounted shaping from a canonical start,
or the constant `-Phi_start` from an intermediate start. A partial advance followed
by death cannot retain a profitable shaping balance. This removes a concrete
reward defect; it does not by itself prove faster learning or select a policy.

The [retained-episode counterfactual](assets/performance/regional-reward-counterfactual-20261025.json)
replayed 49 real terminal failures from the old audit. Twelve old discounted returns
were positive; none is positive under the new definition. No source episode or
replay chunk was rewritten. Interrupted collectors were excluded, and absent RAM
contact labels are rejected rather than invented. These reused episodes are
development evidence, not unseen validation. Separate actual-game traces verify
grounded advancement, airborne suppression, stationary/revisit behavior and exact
closure; portable tests cover timeout absorption, signatures, normalization and
progress after nonfatal damage.

## Curated intermediate starts

The original starting slots 1/3 generated all candidates through explicit control
schedules. Slots 2/4 were used only to verify the RAM star detector and never entered
the curriculum. Each accepted state was alive, grounded and undamaged; it passed
two 60-frame idle probes and exact scene-frame/XYZ/HP restoration. The new environment
then independently restored each accepted state, ran another 60-frame probe and
confirmed that `full_start=True` overrides curriculum selection.

| Mission | Accepted / candidates | Formal preparation frames | Independent reset-probe frames |
| --- | --- | ---: | ---: |
| SMG1 Sweet Sweet / Rocky Road |3/4|1764|180|
| SMG2 Spinning and Spinning and Spinning |4/4|1396|240|
| SMG2 Turning Turning Double Time |2/4|1284|120|

Rejected candidates remain in the preparation evidence. The deeper SMG1 attempt
fell; SMG2 Double Time's obstacle approach failed idle safety, and its deeper
candidate was not an alive grounded safe start. Passing a 60-frame test is bounded
stability evidence, not an indefinite safety guarantee on a rotating platform.

Local `.local/regional-curriculum/{smg1,smg2}/bank.json` manifests bind emulator,
route and individual state SHA256 digests, origin slot, scenario, region, health,
scene frame and position. A start must remain at least 2000 world units from the
near-star calibration point. Paths are restricted to local simple `.state` names;
unexpected traversal, corruption, goal fixtures or unvalidated entries fail closed.
The bank and snapshots are ignored by Git. Public evidence contains only metadata
and hashes, not ROM/state bytes.

The mixture has 50% canonical reset probability and 50% intermediate reset
probability within each scenario; finite realized counts may differ. Eligible
anchors are sampled uniformly. Reset initializes `Phi` to that anchor's existing
distance and pays zero for teleporting there. Episode CSV, summaries and replay
provenance identify `full_start` versus `curriculum`, anchor ID and region.
Sequence replay reports separate transition counts by start kind. Assisted stars,
if eventually observed, cannot establish full-start mission success.

## Separate temporal-control comparison

The same frozen source checkpoint, seeds 11/23/47, canonical starts and at most 1200
physical frames per attempt were used in each condition, independently per mission.
Only seed 11 adds full-RGBA video capture; seeds 23/47 use the normal CPU capture.
All failures and timeouts are retained.

| Condition | Policy interval | Jump/spin press/release | Jump direction |
| --- | ---: | --- | --- |
| legacy12 |12|8/4|Existing inertia; no extra directional button |
| held12_inertia |12|12/0|Existing inertia |
| held12_carry |12|12/0|Most recent cardinal button held during jump |
| held4_carry |4|4/0|Same carried direction semantics |

Eight logical labels remain `none,right,left,forward,backward,jump,spin,crouch`.
The optional carried jump applies two physical controls, direction+jump. `none`
and crouch clear stored direction; reset clears it too. This is an explicit action
semantics hypothesis, not an undeclared ninth action. Comparing legacy12 to
held12_inertia tests pulse duration; comparing the two held12 conditions tests
carried direction; comparing held12_carry to held4_carry tests decision interval
with matching hold/carry semantics.

Source SMG1: `runs/m8-smg1-boundary/boundary-heads_strong-20261025-162736-258709/actor-smoke.pt`
(`c42a1b168b7c2bfe9e3857b4f49082ac91576be2a0f07a966dbbf1df14f1f1fe`),
with one actor update. Source SMG2:
`runs/m8-smg2/world-model-20261025-133915-837384/world-final.pt`
(`81d9c99ab880759f744a29b7094e3317869c0098c30fa001b3681c6a61b245d7`),
with 134 actor updates. Both are weak policies. Loads use original signed settings;
adaptation is explicitly inference-only, with no converted/resumable checkpoint,
no optimizer calls and no replay. Counters and source digests remain unchanged.

| Condition | SMG1 mean confirmed distance | SMG2 Spinning mean | SMG2 Double Time mean |
| --- | ---: | ---: | ---: |
| legacy12 |437.8|203.4|211.6|
| held12_inertia |333.3|279.9|285.8|
| held12_carry |343.8|331.1|323.8|
| held4_carry |124.3|52.5|71.3|

Distances are grounded opening arc units, not native mission percentages. All 27
twelve-frame attempts ended in RAM death. All 9 four-frame attempts reached the
1200-frame limit alive but with less navigation. Zero attempts collected a star;
zero nonfatal HP losses were observed in this comparison. Held12 carry versus
held12 inertia has a small positive SMG2 signal (+18.3%/+13.3% mean distance), with
inconsistent SMG1 results. Three seeds provide no robust significance claim.
Four-frame recurrent inference receives three times as many updates per unit game
time; its world dynamics were not trained or gated at that interval. Its poor
result therefore cannot establish that a newly trained frame4 agent would fail.
It also does not justify adopting frame4 for speed: extra decisions may increase
decisions/second without reducing emulated time needed to solve the task.

An independent scripted equivalence check reproduced 17 legacy decisions using
three 4-frame commands each: directions repeat three times; jump/spin use
press,press,none to reproduce 8-on/4-off. Every mission matched all 17 RAM snapshots,
including scene frame, ground/HP/death flags, with zero XYZ error and identical
final pixel arrays. Thus frame4 retains physical control reachability. This is a
control fixture, not a learned result. The first equivalence controller stopped
before the third mission because it retained the previous timing value; the fix
reset timing explicitly and retried only the unexecuted mission. Both reports
and the failure remain retained, without duplicating completed budgets.

Videos contain the exact grayscale actor frame, action probabilities, selected
logical action and actual physical buttons sampled every two frames. Digests are
stable within each decision; legacy releases and optional carried direction are
checked against the sidecars. Encoded frame counts match all eight sidecars.
Videos are labelled `FROZEN POLICY TIMING TEST`, include source actor counts, and
are not presented as new trained completions. Videos and real episode CSV/PNG
remain under the exact runs recorded in [control evidence](assets/performance/regional-control-20261025.json).

## Reproduction, evidence and next training boundary

Use the project virtual environment. Preparation and publication are separate:
publication creates a new bank directory and refuses to overwrite an existing one.
The committed manifests in configuration refer to this machine's verified local
banks; another machine must prepare its own compatible states and explicitly
review/update those hashes. Do not bypass integrity checks to reuse another bank.

```powershell
.venv\Scripts\python.exe scripts/prepare_regional_curriculum.py --game smg1
.venv\Scripts\python.exe scripts/prepare_regional_curriculum.py --game smg2
# Substitute the reported preparation path and a NEW output folder:
.venv\Scripts\python.exe scripts/build_regional_curriculum.py --preparation RUN/result.json --route config/routes/smg1-opening-regional-v3.json --output NEW_BANK_FOLDER
.venv\Scripts\python.exe scripts/validate_regional_intervention.py --game smg1 --preparation RUN/result.json
.venv\Scripts\python.exe scripts/audit_regional_reward.py --output runs/regional-counterfactual.json
.venv\Scripts\python.exe scripts/compare_regional_control.py --game smg1 --checkpoint SOURCE.pt
.venv\Scripts\python.exe scripts/verify_frame4_equivalence.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
```

[Preparation/reset/equivalence evidence](assets/performance/regional-preparation-20261025.json)
retains exact reports, frame counts, state/route digests and rejected candidates.
Formal preparation used 4444 physical frames, independent reset probes 540, and
matched-control equivalence 1224. Earlier finite geometry surveys remain separate
engineering fixtures under `runs/regional-intervention/geometry-*`, without
training replay or completion accounting. The local state records all actual
preparation, validation and control runs; no background follow-up is active.

The next learning experiment should isolate new reward/curriculum at 12 frames,
starting with a finite health block and fresh signed replay, followed by review
of real grounded exits, damage/deaths and full-start behavior. Historical reward
heads and held-out prediction scores do not become valid under a new reward/start
distribution or frame interval. Preserve original checkpoints and unchanged gate
thresholds; collect compatible data and establish fresh prediction gates before
claiming valid imagination. No extra training/fitting block was launched by this
environment/control intervention. There is no evidence supporting six simultaneous
learners or unattended million-step extension from these checks.
