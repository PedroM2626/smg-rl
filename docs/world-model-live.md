# Authorized online world-model experiment

## Scope and authorization

On October 22, 2026 the owner explicitly authorized continuing the Dreamer-inspired
workflow, including collection and training. The earlier local hold was archived
with that authorization before either emulator job was launched. The preserved
PPO jobs remain stopped. No previous PPO decisions are added to these new agents.

Two independent learners started at approximately 20:01 America/Sao_Paulo:

| Game | Configuration | Online run | Workers | New real-decision budget |
| --- | --- | --- | ---: | ---: |
| Galaxy 2, both Twisty Trials variations | `config/world-model.yaml` | `runs/world-model-20261022-200121-619365` | 4 | 1000000 |
| Galaxy 1, Sweet Sweet Galaxy | `config/smg1-world-model.yaml` | `runs/smg1/world-model-20261022-200121-657138` | 1 | 1000000 |

These are compact original Gaussian RSSM agents inspired by Dreamer, not a
DreamerV3 reproduction. See the [implementation specification](world-model.md)
for equations, architecture, loss functions, gates and limitations.

## Stages and measurement

Collection starts from the configured initial savestates and records actual
84 x 84 grayscale observations. Warm-up actions are random. Model optimizer
updates start only after at least 4096 compatible replay transitions and the
presence of both training and held-out episodes. An episode belongs wholly to
one split. Unflushed transitions can make the replay counter temporarily lower
than the real-decision counter.

Every 32 vector decisions, the learner refreshes replay, performs four model
updates when eligible, then checks held-out prediction quality. This corresponds
to 128 new real decisions per cycle for Galaxy 2 and 32 for Galaxy 1. It is a
different replay-to-interaction ratio across games, not a controlled comparison.
Policy updates in imagination require every scenario's gate to pass. A running
model-fitting job may therefore still have zero actor updates; this is an
explicit safety condition on model quality, not proof of policy learning.

Use the run's `status.json` for the current `real_decisions`, `model_updates`,
`actor_updates`, `imagined_transitions`, replay split, losses and per-scenario
validation metrics. The speed shown there is cumulative real decisions per
wall-clock second including startup and fitting, rather than imagined steps
per second. It is not a measured improvement in time to solve the mission.

The dated initial observation is retained separately in
[the live evidence record](assets/performance/world-model-live-20261022.json).
The earlier [synthetic validation record](assets/performance/world-model-validation.json)
remains evidence about software behavior only.

At approximately 20:09 local time, both learners had begun fitting real visual
data. These are initial snapshots, not continuously updated counters:

| Game | Real decisions | Model updates | Actor updates | Imagined transitions |
| --- | ---: | ---: | ---: | ---: |
| Galaxy 2 | 6912 | 80 | 0 | 0 |
| Galaxy 1 | 4176 | 4 | 0 | 0 |

Both statuses were `running`; learner PID/create-time identity and all five owned
native worker processes were verified. All recorded losses and validation
metrics were finite, neither replay reported rejected archives, neither run
contained `error.txt`, and both periodic checkpoint SHA-256 manifests matched.
The policy gates remained closed because early image and terminal predictions
were insufficient. Fitting the model is working; successful imagined policy
learning has not yet been demonstrated. The evidence JSON retains the complete
per-scenario reasons and contemporaneous GPU resource observation.

## Runtime checks and recovery

```powershell
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/world-model.yaml status
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/smg1-world-model.yaml status
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/world-model.yaml inspect-replay
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/smg1-world-model.yaml inspect-replay
```

Both learners use CUDA and isolated hidden Dolphin profiles. No keyboard focus
is needed. ROMs, replay, checkpoints, runtime profiles and logs remain local and
ignored by Git. Five native emulator workers are expected while both jobs run.

Periodic checkpoints are saved every 4096 new real decisions; the latest three
are retained. A normal stop or caught exception saves `world-final.pt` with its
checksum/signature JSON manifest. The world-model workflow does not use the PPO
auto-supervisor. If a job fails, inspect `error.txt`, `status.json` and worker
logs, fix the cause, and explicitly resume its latest valid checkpoint:

```powershell
# Replace both placeholders; --steps is the additional real-decision budget.
.\.venv\Scripts\python.exe -m smg_rl.cli --config config/world-model.yaml train-world-model --steps REMAINING --resume CHECKPOINT.pt
```

Use `config/smg1-world-model.yaml` for the Galaxy 1 equivalent. Never launch a
duplicate job against a still-running learner. Keep replay and manifests with
checkpoints; weights alone cannot restore the dataset. Recovery resets the game
to the initial savestate rather than restoring an exact in-flight emulator state.

## Interpretation limits

Only a real full-start trajectory with the game's RAM star-completion predicate
establishes success. Neither fitting a model, an imagined success, nor a near-star
fixture counts. Actual RAM positions remain the basis of top-down trajectories
and calibrated geometric progress. This percentage is not native mission
completion from the game. No sample-efficiency advantage, time-to-solution gain,
or full-course learned success has yet been established for this experiment.
