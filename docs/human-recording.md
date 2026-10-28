# Record a human Sweet Sweet demonstration

The recorder launches one private, frame-stepped Dolphin using the existing
project native build and copies of the user's assets/profile. It does not
change the user's Dolphin settings or original slots. Only the SMG1 USA Sweet
Sweet canonical start (`data/RMGE01.s01`) is used. The original near-star slot
2 is never used by a recording session.

1. Close other Dolphin instances. Double-click `record-sweet-sweet.bat` in the
   project root, or from PowerShell run
   `.venv\Scripts\python.exe scripts/record_human_smg1.py`.
2. Wait for **SMG RL - Human recorder**, click that game window, then press
   Enter. Keep the terminal open. The game is paused whenever the recorder
   window loses focus.
3. Use **W/A/S/D** to move, **left mouse button** to jump, **middle mouse
   button** to spin and **Shift** to crouch. Button combinations are allowed.
   Use **F9** in the recorder window to stop without saving a demonstration.
4. If Mario dies, the recorder discards that attempt's video, observations and
   inputs, then reloads the original starting savestate. Release the controls
   and continue. Only a full-start star event confirmed by SMG1 RAM promotes an
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
where possible; if rendering or encoding cannot keep up, gameplay may run more
slowly. There is no game audio in this isolated worker.

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
success gating, simultaneous inputs, output integrity and safe deletion; a real
human full-level success remains to be recorded. The 30-fps video encoder and
the interactive window were also smoke-tested separately on this machine. A
separate real-game fixture applied forward and jump simultaneously for 16
frames and observed both forward movement and increased height. A one-minute,
one-attempt interactive-window smoke subsequently registered game input and a
death; its failed attempt was discarded and the session ended at its configured
limit. A completed human full-level recording has not yet been tested.
See the [validation summary](assets/performance/human-recorder-validation-20261028.json).
