# Owner-authorized six-instance regional training

**Completed and reviewed:** both blocks reached their exact caps. All six owned
emulators and both controllers exited, with no process errors. No successor or
extra fit was dispatched. The hourly follow-up is retired after this review.
The [completion audit](assets/performance/regional-six-completion-20261025.json)
and [top-down trajectories](assets/performance/regional-six-completion-20261025.png)
are authoritative dated evidence; startup statements below retain their original
context.

On October 25, 2026 the owner explicitly requested six concurrent instances,
two for each mission, after the regional environment/control validation. This
supersedes the preceding intervention's two-emulator limit for this new run.
The previous finite pilot stays closed. No original checkpoint, replay root,
ROM, savestate or profile is overwritten. Current local authority and exact
controller identities are in `.local/regional-six-state.json`.

## Workers, policies and initialization

| Game policy | Profile | Workers | Worker-to-mission mapping |
| --- | --- | ---: | --- |
| SMG1 | `config/regional-six-smg1.yaml` | 2 | Both Sweet Sweet / Rocky Road |
| SMG2 | `config/regional-six-smg2.yaml` | 4 | 0/2 Spinning; 1/3 Double Time |

These are **six real emulator collectors and two independent game policies**,
not six separately trained policies. Each worker has a private Dolphin profile,
RAM reader and input bridge. Workers share only their own game's learner/replay.
No shared novelty reward is active. Controls remain eight exclusive actions,
twelve-frame decisions with 8-on/4-off jump/spin pulses. The signed regional
potential, damage/death rules and curated bank checks remain unchanged.

`smg_rl.regional_transfer` provides a deliberate same-control warm start. It
checks matching game, scenario ordering, eight action identities/execution,
crop, image size, frame interval and tensor shapes. It retains encoder,
recurrent latent dynamics, decoder and ensemble. Reward/support, continuation,
actor, critics, all optimizers and replay RNG start fresh. Branch counters start
at zero; original ancestor counts and SHA256 are retained in provenance. The
initial actor has an explicit prior: 80% walking/idle, 15% jump, 2.5% spin, 2.5%
crouch. That prior is not learned navigation.

Sources are the repaired SMG1 `actor-smoke.pt` and selected SMG2 `world-final.pt`
recorded in [the regional validation](regional-intervention.md). The new immutable
initial files and provenance remain local at `.local/regional-six/*-initial.*`.
No source prediction quality/readiness is inherited. All original quality-gate
thresholds remain unchanged and must pass on fresh compatible real replay before
any imagined policy updates. This avoids reusing old reward heads under changed
reward and absorbing-timeout semantics without throwing away trained vision.

## Finite stages and actual dispatch

The startup health budget was 256 real decisions per emulator. Both jobs finished
without errors; all six workers contributed exactly 256 decisions and every replay
chunk was accepted. All sampled contact values were known; no RAM star or nonfatal
HP loss occurred. Fresh source counters were zero before collection. No actor or
model update occurred below the unchanged minimum replay threshold.

| Game | Health run | Health real count | Reviewed successor additional count | Cumulative cap |
| --- | --- | ---: | ---: | ---: |
| SMG1 | `runs/regional-six/smg1/world-model-20261025-214517-915763` |512|7680|8192|
| SMG2 | `runs/regional-six/smg2/world-model-20261025-214518-089752` |1024|15360|16384|

Actual main runs:

- SMG1: `runs/regional-six/smg1/world-model-20261025-214721-733881`.
- SMG2: `runs/regional-six/smg2/world-model-20261025-214757-927434`.

The main jobs resume their **compatible new-branch health checkpoint**, not the
historical source directly. Both main blocks end at their fixed boundary and
save a final checkpoint. Overall allocation is 8192 new real decisions per mission,
24576 across the three, including health collection. No automatic successor or
additional offline fit is allocated. Do not restart a completed budget as though
it were unexecuted; account for all partial collection on a failure.

Only this reviewed main dispatch is authorized by this record. Future successors
require review of completed outputs and a separate finite allocation. Hourly
follow-up `revisar-treino-mario-com-seis-inst-ncias` inspects actual controllers,
timestamps, logs, gates, real/model/actor/imagined counts, replay rejection, HP,
deaths, trajectories and resources. It stays quiet unless something meaningful
changes, then reports in Portuguese. Once both blocks finish it reports and
pauses; it cannot dispatch an indefinite extension. A completed block may wait
for the hourly review.

## Startup resource measurement and interpretation

A snapshot with all six owned Dolphin processes active measured about 6.0GiB
available RAM (81.1% used), 83.3% CPU utilization and 1300MiB GPU memory used. GPU
utilization was 75% and temperature 61C. This is a startup snapshot on the owner's
machine, not a promise of thermal behavior, steady-state speed or model-fitting
memory. The test suite also passed under concurrent collection: 168 portable
tests and Ruff. Live status, process ancestry and resource snapshots are recorded
locally rather than presenting these dated values as a dashboard.

Do not call an assisted reset's existing opening position new learned progress.
Use start-relative grounded advancement and keep `full_start` and `curriculum`
outcomes separate. Only real RAM-confirmed star collection from the canonical
start establishes full-start success. Geometric projection, opening-route
completion, scripted fixtures and imagined rollouts do not. The route maps still
cover only the verified openings, as documented in the previous intervention.

The [dated startup evidence](assets/performance/regional-six-startup-20261025.json)
contains exact controllers, ancestry, reviewed health data and a six-process
snapshot. It must not be read as current status after its recorded timestamp.

## Completed-block review

The October 26 UTC review inspected both actual controller PID/creation-time
identities, owned processes, stderr, terminal status, final checkpoint digests
and serialized counters, all compatible disk replay and all 495 episode CSVs
from health plus main collection. The jobs completed on October 25 local time
(SMG1 at 22:04, SMG2 at 22:11 America/Sao_Paulo). Both final checkpoints are retained
at their main run's `world-final.pt`.

| Policy | New real decisions | Model updates | Actor updates | Imagined transitions | Final gate |
| --- | ---: | ---: | ---: | ---: | --- |
| SMG1 |8192|248|3|3072|Blocked: reward MAE and continuation Brier |
| SMG2 |16384|384|32|30960|Passed for both scenarios |

Disk replay contains exactly 8192 SMG1 and 16384 SMG2 transitions, with zero rejected
chunks. SMG2 contains exactly 8192 decisions per scenario. Final `status.json`
snapshots show 8066/16273 replay transitions because they were published before
worker shutdown flushed partial chunks. The post-close disk audit and episode
sums reconcile to the full counters; this is a stale diagnostic snapshot, not
missing real data.

| Mission | Canonical episodes | Canonical deaths / cuts | Best grounded arc distance from canonical start | Fraction of mapped opening | Full-start stars |
| --- | ---: | --- | ---: | ---: | ---: |
| Sweet Sweet |72|68/4|899.6|41.15%|0|
| Spinning |93|91/2|1477.0|75.32%|0|
| Double Time |86|84/2|1395.3|71.15%|0|

These fractions cover only the short validated opening maps, **not the full
mission**. Whole-course straight-line geometric projection maxima from canonical
starts were 4.916%,4.226%,2.741%, respectively; those are also geometric proxies,
not native game percentages. The best grounded and best projected values can
come from different episodes. All maxima are collection observations, including
initial prior exploration. Per-episode actor-update age was not recorded, so they
must not be attributed to a trained final policy or compared to smaller-budget
preview maxima as evidence of improved learning. No frozen final-policy evaluation
was allocated or launched in this review.

Assisted episodes: SMG1 had 66 (65 deaths, 1 timeout), Spinning 83 (81 deaths, 2 cuts),
Double Time 95 (93 deaths, 2 cuts). Spinning had 11 confirmed opening exits from
intermediate starts, and no opening exit from the canonical start. Neither other
mission had a confirmed opening exit. All assisted and full-start episodes had
zero RAM stars. None recorded nonfatal HP loss, so no nonfatal damage penalty
was exercised by this collection; the separate damage detector evidence remains
unchanged. There were no unknown-contact samples. All 483 terminal failures had
negative discounted returns and correctly closed potential (maximum numerical
closure error 1.11e-15); the 12 collector cuts were excluded from terminal closure.

The useful positive result is that SMG2 fitted the new reward/start distribution
well enough to pass all unchanged final prediction gates and enable 32 actor
updates. That is model-quality evidence, not proof of learned navigation or
mission completion. SMG1 briefly enabled 3 updates but finished with reward
MAE 0.654>0.5 and continuation Brier 0.210>0.1. Terminal reward MAE remained 0.083
and terminal Brier 0.000017, so successful terminal prediction alone did not mean
good predictions throughout live play. Its final dynamics KL was 30.9 and its
reported pre-clipping gradient norm 297.5. A future isolated diagnosis should test
whether carrying the previous 40x terminal loss weighting from a head-only repair
into full-world optimization overweights terminal prediction or destabilizes
latent dynamics. These symptoms support that hypothesis but do not establish
causality; no objective, gate, checkpoint or optimizer was changed to test it here.

The finite allocation is exhausted. New online blocks or offline fits were not
started. The hourly follow-up was removed after this report rather than polling
the completed jobs indefinitely. No unrelated job or automation was changed.
