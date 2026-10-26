# Owner-authorized six-instance regional training

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
