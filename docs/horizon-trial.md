# Eight versus sixteen imagined decisions

October 25, 2026: a bounded follow-up tested the retained Galaxy 2 terminal-weight-5
model and Galaxy 1 curiosity-control model. Their original checkpoints were
preserved. The default imagination horizon remains eight decisions.

## Paired protocol

For each game, horizons 8 and 16 were evaluated on the same eight frozen
held-out batches with Torch seeds 11, 23 and 47. Every existing quality threshold
was retained, including future-image MSE <= 0.04, beating image persistence,
terminal reward MAE <= 0.5 and terminal Brier <= 0.2. Both horizons passed all
three audits for each game. The horizon-16 check actually rolls sixteen steps;
it does not reuse an eight-step prediction score.

Only after the joint audit passed did both arms receive 64 offline actor/critic
updates per seed, starting from identical original actor/model/optimizer states.
The world model stayed bitwise unchanged, with zero new world-model updates or
real training decisions. Training posterior seeds came only from training replay;
held-out images were used only for audits. The same replay draws and per-update
seed schedule were used across horizons, though longer rollouts consume more
random draws within an update. Source reward support bounds and all objective
coefficients were preserved. Candidate checkpoints have separate settings and
integrity manifests.

Across the two games and six paired actor fits per game, this consumed 768 new
actor updates and 1160880 imagined transitions, separately accounted. The
[offline record](assets/performance/horizon-offline-audit-20261025.json) retains
raw audits, timing and counters. Horizon 16 does more imagined work per update;
equal update counts are not equal compute budgets.

Each resulting actor received one fresh full-start attempt at its matching seed,
for each supported scenario, with a 600-decision cap. This totals eighteen
attempts and 2378 real evaluation decisions, with no online optimizer updates.
Full visual recordings and CSV/RAM traces were retained. Evaluation uses the
same full-capture path described in the recording documentation.

## Observed outcomes

| Scenario | Horizon 8 mean projection | Horizon 16 mean projection | Outcomes at both horizons |
| --- | ---: | ---: | --- |
| Galaxy 2 Spinning | 8.37% | 8.77% | All three attempts died |
| Galaxy 2 Double Time | 2.93% | 2.64% | All three attempts died |
| Galaxy 1 Sweet Sweet | 2.42% | 2.48% | All three attempts timed out |

These are geometric high-water projections, not verified platform traversal or
native mission percentages. No star was collected. Paired seed effects are
mixed and small; three short attempts per scenario cannot establish a robust
benefit, harm, convergence result or time-to-solution improvement. The world
model was warm-started from one retained checkpoint per game, rather than
independently trained three times. The reused validation partition is not an
unseen external test.

Decision: keep horizon 8 as the default and retain horizon-16 actors as
experimental artifacts. Sixteen-step prediction is viable on visited states,
but this bounded actor adaptation does not demonstrate better real navigation.
The [complete evaluation record](assets/performance/horizon-8-vs-16-20261025.json)
contains each attempt, provenance, evaluation cost and RAM outcome.

```powershell
.venv\Scripts\python.exe -m smg_rl.cli --config config/staged-smg2-terminal_reward_weight.yaml test-world-horizon --checkpoint runs/staged-smg2/terminal_reward_weight/world-model-20261025-010019-701413/world-final.pt --updates 64
```

The command audits horizons first and skips both actor arms if any joint audit
fails. A run-local `STOP` marker stops fitting between updates. It never launches
Dolphin; real evaluation is a separate, bounded recording command. New fitting
requires an enabled training configuration and respects the local training hold.
