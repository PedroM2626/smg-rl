# Hidden parallel training on Windows

**Status note (2026-10-23):** the supervised PPO jobs described here are stopped;
their final pairs are preserved. The separate world-model experiment recorded in
[the live world-model record](world-model-live.md) was also stopped at the
owner's request on October 23, 2026; its final checkpoints are preserved and
all training remains stopped until the owner explicitly authorizes resuming.

**Native backend update (2026-10-22):** the local learners now use
`config/native.yaml` and `config/smg1-native.yaml`, with shared-memory RGBA
capture and bounded frame stepping. The stock hotkey/PNG backend remains a
fallback. See [native controls, profiling, benchmarks, and migration](native-backend.md).
The older measurements below retain their original backend/timing context.

The background backend runs multiple independent Dolphin processes with no visible game windows and no foreground input requirement. Within one learner, a recurrent visual PPO policy learns from all workers. `config/parallel.yaml` uses four workers balanced between the two Galaxy 2 scenarios; `config/smg1-training.yaml` runs an independent single-worker Galaxy 1 policy. Both use CUDA optimization. See [concurrent operation](concurrent-training.md) for running the two learners together.

This uses Dolphin batch mode with a hidden Win32 rendering surface. The GPU still renders game images for computer vision. It is not a renderer-free `DolphinNoGUI` executable. The existing Dolphin 2609a and original savestate format are preserved.

## Isolation and observations

Every worker owns its Dolphin user directory, Wii storage copy, savestate copies, loopback UDP port, process handle, and screenshot directory. A local DSU server supplies Nunchuk movement, jump, crouch, spin, and emulator commands. `BackgroundInput=True` and `HotkeysRequireFocus=False` are essential. Windows keyboard/mouse injection is not used.

RAM is read from an explicit worker PID using read-only process permissions. The game ID and executable signature are verified before accepting a mapped region. The process-global `dolphin-memory-engine` hook is not used by parallel workers, preventing one instance from supplying another's coordinates or terminal state.

Images come from Dolphin's internal renderer, not the desktop. A frame step flushes the pending screenshot while paused. Only a newly written, complete PNG is accepted. Controller inputs remain held during the pause handshake, and the RAM frame counter acknowledges pause transitions. The frame budget is approximate because commands are polled asynchronously and screenshot flushing can add frames. CSV logs retain the actual scene counter.

The parent runs CNN/LSTM inference and PPO updates on CUDA. Every worker trains its small inverse-dynamics/RND models on CPU, with independent novelty statistics per scenario and worker. Policy learning aggregates all workers; curiosity models are not globally synchronized. CPU thread counts are capped.

## Tested CUDA environment

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

`requirements-gpu.lock.txt` records the full tested environment and CUDA wheel index. The older `requirements-windows.lock.txt` records the original desktop CPU environment. No NVIDIA driver or system security settings were changed.

## Start and resume

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml train-supervised --workers 4 --steps 1000000
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml train-supervised --workers 4 --steps 100000 --resume runs/parallel-EXAMPLE/model-final.zip
```

Both the policy and matching curiosity checkpoint are required. These are alternative fresh/resume examples; do not start duplicate learners unintentionally. If the worker count changes, saved novelty states are mapped cyclically to new workers. Episodes, episodic memory, LSTM state, RNG state, and incomplete PPO rollouts restart. `--steps` counts additional aggregate decisions, including on resume, and can round up to a full rollout. The supervisor preserves this initial target across retries. `train-parallel` remains available for a single attempt without restart; see [recovery semantics](damage-and-recovery.md).

The original desktop Dolphin session is independent of these workers. It can remain paused or be closed separately. The dedicated training controller closes its own child processes when the run ends.

## Inspect and stop

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml status
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml stop
```

These select the newest parallel run in the selected configuration's runs root unless an explicit learner attempt directory is supplied. `stop` creates that run's `STOP` marker and also signals its owning supervisor. A healthy learner finishes the current vector step, saves the final policy and curiosity states, and closes the workers. A worker failure can prevent a final save. Ctrl+C also requests shutdown in a foreground terminal. Select `config/smg1-training.yaml` to control Galaxy 1 separately.

Each run has `controller.json`, `status.json`, periodic checkpoint pairs (every 1024 aggregate decisions, newest three retained), and a final checkpoint pair on normal exit. An error may leave a policy-only emergency snapshot; resume from a complete periodic pair. Trajectories live under `worker-XX/<variation>/episode-NNNNNN.csv`. CSVs are saved for every episode. PNGs are generated every 20 episodes per worker, on every success, and when saving an interrupted episode; any CSV can also be plotted later with the `plot` command.

Runtime profiles are stored under `.local/parallel-profiles/<run-name>/worker-XX`. Each run uses a separate namespace, so restarting training does not reuse another run's controller files or UDP port.

## Measured acceleration

On the local i9-14900HX / RTX 4070 laptop:

| Backend | Aggregate decisions per second |
| --- | ---: |
| Original desktop controller | about 1.0 |
| 2 hidden workers + CUDA | about 4.6 |
| 4 hidden workers + CUDA | about 7.9 |

These historical measurements preceded the longer pause-stability check. An intermediate revision subsequently measured approximately 4.6 aggregate decisions/second. The later 120 ms pause pulse and two-counter-advance check add further overhead; none of these historical rates is a current throughput guarantee. Measure each revision and concurrent workload separately using aggregate **new** decisions over elapsed collection/update time, and record startup/failure downtime separately.

The four-worker test collected 512 additional decisions in about 65 seconds, resumed from 256 to 768 total decisions, completed PPO updates, and saved all four curiosity states. These are decisions, not rendered frames per second. Startup and shader compilation are separate from steady-state collection time.

Improvements include removing desktop input/focus handshakes, collecting in parallel, optimizing on the GPU, running Dolphin at 2x speed, rendering at native internal resolution with VSync disabled, and plotting less often. DSU polling and PNG capture remain bottlenecks. More workers do not guarantee linear scaling; CPU, RAM, GPU rendering, encoding, and storage can become limiting factors.

Higher `emulation_speed` or `frame_skip` can improve throughput but makes control coarser and may skip brief terminal transitions. Revalidate after changing them. A future native frame-synchronous bridge could remove PNG encoding and polling; it is not part of this backend.

## Validation

```powershell
.\.venv\Scripts\smg-rl.exe --config config/parallel.yaml calibrate-background
```

The local `runs/background-validation-20261021-175431-396382/` test passed star collection, death, movement, and pause checks in both variations using a hidden worker. Movement checks allow the initial spawn animation to release control.

The resume/stop integration test at `runs/parallel-20261021-175601-308500/` stopped gracefully at 1140 total decisions, saved both final checkpoint files, and closed its workers. The expanded suite covers both games and the pause-stall regression. No full course has been solved by these short tests; the long run must be judged by independent success rates.

## Sources

- [Dolphin DSU client and input names](https://github.com/dolphin-emu/dolphin/blob/master/Source/Core/InputCommon/ControllerInterface/DualShockUDPClient/DualShockUDPClient.cpp)
- [Dolphin DSU protocol](https://github.com/dolphin-emu/dolphin/blob/master/Source/Core/InputCommon/ControllerInterface/DualShockUDPClient/DualShockUDPProto.h)
- [Dolphin hotkey focus setting](https://github.com/dolphin-emu/dolphin/blob/master/Source/Core/Core/Config/UISettings.cpp)
- [Dolphin internal screenshot pipeline](https://github.com/dolphin-emu/dolphin/blob/master/Source/Core/VideoCommon/FrameDumper.cpp)
- [Official PyTorch CUDA installation commands](https://pytorch.org/get-started/previous-versions/)

## Pause-stall correction

The first long run failed at 1944 total decisions after a pause acknowledgment error. The original 45 ms stability probe could interpret a brief rendering stall as a paused emulator. The current probe samples six times at 30 ms intervals and requires **two separate counter changes** before classifying emulation as running: one pending screenshot step must not count as sustained advancement. It reconciles the measured state before toggling, holds pause input for 120 ms with a 35 ms release interval, and allows six bounded toggle attempts. Ordinary operation timeout remains 30 seconds. Boot has a separate 90-second timeout, and Galaxy 2 startup spacing is five seconds per worker index.

Failures still abort explicitly; no fabricated transition is added to a rollout. Supervision retries from complete pairs, more frequent checkpoints bound lost work, and periodic retention bounds disk growth. These measures have not eliminated every pause/boot failure. See [reproducibility](reproducibility.md) and [damage/recovery evidence](damage-and-recovery.md).
