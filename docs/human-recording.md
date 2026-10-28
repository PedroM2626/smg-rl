# Record a human demonstration

The recorder launches one private, frame-stepped Dolphin using the existing
project native build and copies of the user's assets/profile. It does not
change the user's Dolphin settings or original slots. Only the SMG1 USA Sweet
Sweet canonical start (`data/RMGE01.s01`) is used. The original near-star slot
2 is never used as a starting state by a recording session. The same recorder
also supports SMG2 Twisty Trials, with the scenario-specific initial slots 1
and 3; goal fixtures 2 and 4 are never recording starts. The separate SMG1
Center of the Universe recorder uses start slot 3 and a non-star location
reference in slot 4; see [its dedicated guide](center-of-universe.md).

1. Close other Dolphin instances. Double-click `record-sweet-sweet.bat` in the
   project root, or from PowerShell run
   `.venv\Scripts\python.exe scripts/record_human_smg1.py`.
   For SMG2 use `record-smg2-spinning.bat` or `record-smg2-double-time.bat`.
2. Wait for **SMG RL - Human recorder**, click that game window, then press
   Enter. Keep the terminal open. The game is paused whenever the recorder
   window loses focus.
3. Use **W/A/S/D** to move, **left mouse button** to jump, **middle mouse
   button** to spin and **Shift** to crouch. Button combinations are allowed.
   Use **F9** in the recorder window to stop without saving a demonstration.
4. If Mario dies, the recorder discards that attempt's video, observations and
   inputs, then reloads the original starting savestate. Release the controls
   and continue. Only a full-start star event confirmed by the selected game's RAM promotes an
   attempt to the completed folder; a near-star fixture cannot qualify.
5. After a star, the console prints `validated_recording` and the folder path.
   Share that path with the project operator. A session ends after its first
   valid star, 120 minutes or 100 attempts by default. Explicit limits can be
   changed with `--max-minutes` and `--max-attempts`; they remain finite.

Each successful attempt contains `gameplay.mp4` (silent, unmodified game
frames), `trajectory.csv` (XYZ, HP, contact and terminal telemetry), a top-down
`trajectory.png`, `inputs.jsonl`, `observations.gray8`, and `manifest.json`
with source-state and file SHA-256 hashes. Inputs are sampled synchronously at
two-emulated-frame intervals; each record stores the exact simultaneous
controls applied for those two frames. `observations.gray8` stores one initial
and one subsequent cropped 84x84 uint8 frame per input sample, in row-major
order. The exact frame duration and button-mask order are in the manifest.
Very brief button taps that occur entirely between samples may not be applied;
the recorded controls are the ones actually sent to the emulator.

This is a human demonstration, not evidence that a model learned to finish.
The recorder's simultaneous buttons and two-frame timing are **not** the
existing eight-exclusive-action, twelve-frame training signature. Do not load
the raw data directly as motor8 replay; conversion, timing compatibility and a
replay check are separate tasks. The visible game is paced to emulated 60 Hz
where possible. Video encoding uses a bounded background queue: if it fills,
the frame-stepped emulator stays paused until there is room, and the window
reports `Encoder busy - game paused`. Late frames are not replayed in a
catch-up burst. A stalled renderer or encoder can slow gameplay, and the
reported problem may have other causes; there were no per-frame wall-clock
timings in the original recording, so its specific cause cannot be established
retrospectively. There is no game audio in this isolated worker.

Failed attempts are deleted only within the recorder-owned pending directory;
unexpected software errors retain incomplete pending files in quarantine and
never mark them as completed. All recordings remain local under ignored
`runs/human-demonstrations`; ROMs, states and recordings are not committed.

## Validation and scope

On October 28, 2026, an isolated live-game check forced a death from slot 1,
confirmed the failed recording was discarded, and verified that reset restored
the canonical initial telemetry. A separate slot-2 detector fixture reached the
RAM star event but was rejected as a full-start demonstration. These are
engineering checks, not a human full-level completion. Synthetic tests cover
success gating, simultaneous inputs, output integrity and safe deletion. The 30-fps video encoder and
the interactive window were also smoke-tested separately on this machine. A
separate real-game fixture applied forward and jump simultaneously for 16
frames and observed both forward movement and increased height. A one-minute,
one-attempt interactive-window smoke subsequently registered game input and a
death; its failed attempt was discarded and the session ended at its configured
limit. See the [initial validation summary](assets/performance/human-recorder-validation-20261028.json).

Later that day, the owner recorded a full-start human completion after two
discarded deaths. Its 1,734 two-frame inputs were independently replayed from
the same original savestate in an isolated Dolphin: the native RAM star was
reached at the same sample, all 1,735 saved visual observations matched exactly,
and every sampled XYZ position matched. This establishes reproducibility of
this recording and state on this setup, not learned agent success or robustness
across independent states. The local recording contains simultaneous controls
and remains incompatible with the current motor8 replay without an explicit,
validated conversion. The owner reported intermittent display freezes; no
original timing data prove the cause. See the [replay evidence](assets/performance/human-demonstration-replay-20261028.json).

To replay a completed recording, use an isolated emulator with a finite deadline:

```powershell
.venv\Scripts\python.exe scripts/verify_human_demonstration.py <completed-attempt-folder> --max-seconds 900
```

The script verifies file and source-state hashes, records an independent result
under `runs/human-verification`, and never changes the source recording. Its
samples are verification interactions, not training steps.

On October 28, a separate live SMG2 smoke test verified both canonical start
slots, and both goal fixtures produced native star signals in their correct
scenarios. Each fixture was rejected as a full-start completion. Later, the
owner recorded both full-start SMG2 missions: Spinning after one discarded
death, and Double Time after two. Independent replays reproduced the RAM star
and every recorded image and position: 1,737 control samples for Spinning and
2,757 for Double Time. The original recordings and replays remain local.

## Using the Sweet Sweet trajectory in training

The [human route candidate](../config/routes/smg1-human-full-candidate-v1.json)
contains 47 ordered waypoints sampled from the real recorded grounded path. It
is consumed by the independent [experimental IQN profile](../config/human-route-smg1-candidate.yaml),
which keeps eight motor actions, 12-frame decisions, visual frame stacking, no
curiosity, and full-start resets. It uses a separate signed reward and replay;
old checkpoints and replays are not exact-resume compatible. The raw human
actions and frames are not relabelled or injected into the agent replay.

Schema 4 checks ordered waypoint visits against observed motion segments, even
across jumps, and releases earned credit only on a later grounded observation.
Teleport-size discontinuities invalidate eligibility for the episode. Stationary
samples and revisits do not repeatedly pay. Route credit is capped at 1 over
the full path; its per-metre early reward is therefore much smaller than with
the opening-only route. Death/timeout remain -2 and the separate RAM star is
+20. Reaching the final grounded waypoint is neither an event nor proof that
the star was collected.

On the source recording, all six offsets of 12-frame subsampling gave between
98.4% and 100% of the route, with the lowest offset ending before its final
eligible grounded observation. These are reused human data, not held-out agent
results or native mission percentages. A live idle smoke test paid no route
reward for 20 decisions, then a short motor8 test ended in death. The objective
has not yet passed an independent A/B comparison or broad reward-exploitation
audit, and no policy has been trained or evaluated with it. In particular,
airborne passage only queues visits, which may grant credit after a later
landing; this remains a possible shortcut to study. See the [adaptation evidence](assets/performance/human-route-adaptation-20261028.json).

`scripts/prepare_human_route.py` prepares a mission-specific candidate from an
exact full-start human replay, refusing to overwrite existing outputs. The
recorded SMG2 completions produced separate signed experimental profiles for
[Spinning](../config/human-route-smg2-spinning-candidate.yaml) and
[Double Time](../config/human-route-smg2-double-time-candidate.yaml), with 61
and 109 waypoints respectively. All six offsets of 12-frame subsampling
recognized 100% of each source trajectory's mapped route; this reuses the same
human demonstrations and is neither a held-out evaluation nor a native mission
percentage. Each profile is mission-specific and starts with a distinct replay;
the SMG1 route is never reused as an SMG2 route. These candidates have not been
tested for reward exploitation or evaluated in trained agents.
