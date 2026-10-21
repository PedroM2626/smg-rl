# Local validation report

The newer [background training report](background-training.md) records four-worker CUDA validation, approximately 8x aggregate throughput, hidden star/death checks, and graceful stopping. The report below retains the original desktop baseline for comparison.

Validation date: **2026-10-21**, America/Sao_Paulo. Emulator: **Dolphin 2609a**, Windows. Game: **SB4E01**. Python: **3.11**. All tests used the supplied local states; no game asset was uploaded.

## Memory and control validation

All four savestates were loaded. In both scenarios:

- The starting state has Mario near `(0, -1150, 500)` and scene nerve `0x807D5310`.
- The near-star state is initially incomplete.
- A real jump collects the star and changes the scene nerve to `0x807D5318`.
- Walking backward off the starting platform causes actor nerve `0x807D4E3C` and scene nerve `0x807D5320`.
- Forward movement changes the three position floats without claiming death or completion.
- The RAM scene-frame counter stays stable while paused.

The structured evidence is in [`config/calibration.json`](../config/calibration.json). Full initial calibration traces are retained under `runs/validation-20261021-170616-414513/`.

The final configuration was also tested using the **isolated copied profile**: Dolphin was booted through the project's `--user`, `--exec`, and `--save_state` launcher, and all four-state checks passed again. Final evidence is retained under `runs/validation-20261021-172440-122183/`. This run used the 0.25-second action interval and paused state restoration, and left Mario paused at the first starting state.

The star detector uses the current collection sequence rather than total stars in the save file. This also handles replaying a star already collected in the user's save.

## Live training smoke test

Command:

```powershell
.\.venv\Scripts\smg-rl.exe --config .local/smoke-config.yaml train --attach --steps 128 --rollout-steps 32
```

The temporary smoke config uses a 0.10-second held-action interval, a 32-decision episode limit, and a 32-decision idle limit. Production defaults remain in `config/default.yaml` with a 0.25-second held-action interval and longer episode limits. The run took about 129 seconds for 128 decisions, including four PPO rollouts and updates, at roughly one decision per second on CPU.

| Episode | Variation | Decisions | Outcome | Farthest valid progress |
| --- | --- | ---: | --- | ---: |
| 1 | Spinning | 32 | Time limit | 0.414% |
| 2 | Double Time | 32 | Time limit | 0.452% |
| 3 | Double Time | 32 | Time limit | 0.271% |
| 4 | Spinning | 32 | Death | 0.132% |

No full course was completed. These results validate live image observations, nonzero novelty rewards, progress signals, death termination, scenario resets, gradient updates, and checkpoint creation. They do not establish that the agent has learned to beat either course.

Local output: `runs/train-20261021-171440-784733/`. Both `model-final.zip` and the matching `model-final.intrinsic.pt` are retained locally and excluded from Git.

Measured sample trajectories are committed as small, reproducible evidence artifacts:

![Spinning smoke trajectory](assets/spinning-smoke.png)

[Spinning CSV](assets/spinning-smoke.csv)

![Double Time smoke trajectory](assets/double-time-smoke.png)

[Double Time CSV](assets/double-time-smoke.csv)

The progress plot zooms to the observed percentage range and explicitly labels full-course progress as 100%. The top-down view uses equal-scale world X/Z coordinates.

## Evaluation with intrinsic reward disabled

The smoke checkpoint was evaluated deterministically for one 32-decision episode per variation, with intrinsic rewards and curiosity updates disabled:

| Variation | Successes | Farthest valid progress |
| --- | ---: | ---: |
| Spinning | 0 / 1 | 0.958% |
| Double Time | 0 / 1 | 0.601% |

Local evaluation output: `runs/eval-20261021-171702-636139/`. The tiny sample and short timeout are only an integration check, not a reliable policy comparison.

## Resume validation

The 128-decision checkpoint and its matching curiosity state were loaded into a new run, trained for another 32 decisions, and saved successfully at **160 total decisions**. This verifies actual serialization and resumption of PPO and both curiosity models, beyond the unit-test roundtrip.

Latest local smoke checkpoint: `runs/train-20261021-171922-286912/model-final.zip`, with matching `model-final.intrinsic.pt`. This remains a pipeline smoke checkpoint, not a solved-course policy.

A final fresh 32-decision PPO run also passed using the production 0.25-second action interval, the paused reset implementation, and the isolated profile. It reached 0.781% valid progress without completing the course and saved its checkpoint in `runs/train-20261021-172631-112036/`. This is the latest fresh-run checkpoint; the 160-decision checkpoint above is the resume-test checkpoint.

## Automated checks

The test suite covers dynamic pointer relocation, endianness, both scenario IDs, pending nerve precedence, completion/death separation, invalid memory rejection, progress farming prevention, fall/teleport suppression, frozen episodic embeddings, RND checkpoint state, small-distance normalization, policy observation boundaries, episode logging/plots, input release around pause shortcuts, bounded pause retries, and source-profile preservation.

`python -m pytest -q`: **27 passed**. `python -m ruff check src tests`: **passed**. `python -m pip check`: **no broken requirements**.

## Subsequent integration and reliability checks (2026-10-21)

The canonical private repository is `PedroM2626/smg-rl`; its history was transferred from the initially configured account. The expanded portable suite has 43 passing tests, including independent RMGE01 telemetry, cross-game rejection, both CLI training guards, two-slot profile preparation and pause-stall regression. Ruff and dependency consistency checks passed locally.

Galaxy 1 Sweet Sweet Galaxy passed live star collection from slot 2, backward-fall death from slot 1, coordinate movement and pause checks. The full pointer map, fixture telemetry and measured top-down plots are in [the Galaxy 1 report](smg1.md). No Galaxy 1 training was started.

The first Galaxy 2 long run stopped at 1944 aggregate timesteps on a pause acknowledgment failure. It was resumed from the intact 1140-step paired checkpoint after extending frame-stability detection and reconciling measured pause state. The revised four-worker CUDA run is `runs/parallel-20261021-183228-379160/`. During the subsequent live observation it exceeded 2700 aggregate timesteps, completed further PPO updates and wrote a fresh policy/curiosity pair at 2164. Its measured collection/optimization rate was approximately 4.6 new aggregate decisions per second. This is an active training observation, not a claim of an unattended completed experiment or learned full-course success.

A final Galaxy 2 hidden calibration using the revised pause logic passed both star collections, both backward-fall deaths, both movement checks and pause stability. Its source evidence is `runs/background-validation-20261021-183949-902972/`; the calibration JSON was refreshed and portable measured traces were copied to `docs/assets/smg2-calibration/`. These detector checks ran in an isolated instance while the four-worker Galaxy 2 learner continued.
