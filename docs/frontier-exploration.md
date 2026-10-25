# Frontier savestate exploration and policy recordings

Implemented and exercised on October 25, 2026. This is a bounded,
Go-Explore-inspired return-and-explore mechanism, not a reproduction of the
[full Go-Explore algorithm](https://www.nature.com/articles/s41586-020-03157-9).
All original assets and checkpoints are retained. Default configurations remain
unchanged; `config/frontier-smg1.yaml` and `config/frontier-smg2.yaml` opt in.

## Archive and safety boundary

Each worker owns a separate archive for each scenario. A world-X/Z grid of
300-unit cells identifies already attempted states. Only a state with at least
0.5% geometric projection, six observations with at most 20 units of vertical
variation, no death/completion, unchanged health, and height at least the initial
height minus 50 can be proposed. These are conservative screening heuristics,
not a validated native grounded/contact flag.

The candidate is serialized while paused. The worker then performs five idle
decisions (60 emulated frame boundaries), rejecting a state if Mario dies,
completes, changes scenario/health, moves more than 250 units or changes height
by more than 25 units. The original snapshot is restored afterward, checking
coordinates within 0.01 units, health, scenario, scene nerve and scene-frame
counter. This preserves platform timing as part of Dolphin's full state.
Passing the short probe does not guarantee future safety on a moving platform.

Accepted states are selected with probability 0.5 at episode reset, weighted by
`1 / sqrt(1 + visits)` to revisit less explored entries. The other resets use
the original start slot. This pilot caps each worker/scenario archive at eight
accepted states and sixteen candidate attempts; it stops proposing when either
bound is reached. There is no unbounded snapshot capture or silent deletion.
Rejected snapshots are retained but quarantined from reset selection.
Use a fresh archive root or revalidate entries when changing screening criteria.
An exclusive OS writer lock rejects two owners of the same worker/scenario
archive and is released on close or process exit. Use separate roots for
concurrent policy comparisons.

The native bridge advertises optional capability bit 1 in header field 44.
Command 5 saves only slot 8; extended command 3 loads slots up to 10. Python
stages archive restores into slot 9. Both scratch slots belong to a newly
created, isolated runtime profile. Existing scratch files are retained under
unique names, while archive files are immutable and checked with SHA-256.
Slots 1-4 and the user's emulator/profile are untouched. Archive identity binds
the game, scenario and exact emulator executable digest. No RAM writes or game
function calls are introduced.

The separately deployed executable is `.local/native-dolphin-frontier/Dolphin.exe`
with SHA-256 `b9f2131c647bdcd82d3846a4a820d978316ca55b517493d51503ea9f4b016fc4`.
The old native executable is retained. The bridge source remains MIT; the
linked Dolphin executable retains its existing GPL/third-party boundaries.
Use the [native build procedure](native-backend.md) with
`scripts/build_native.ps1 -Deploy .local/native-dolphin-frontier` and the locally
configured toolchain; no binary is shipped in this repository.
The [native fixture](assets/performance/native-frontier-validation-20261025.json)
records an exact Galaxy 1 position/health/frame-counter round trip after movement.
It is detector validation, not a learned completion.

## Learning, rewards and accounting

An assisted reset starts a new replay episode, resets policy recurrence through
the existing `is_first` mask, repeats the restored observation in the frame
stack and resets health history. The global course projection high-water mark
is initialized to the restored position, so restoring a state cannot earn a
teleport reward. Policy inputs remain pixels and scenario identity; RAM/grid
coordinates are used for curriculum selection and telemetry only.

CSV and episode records include `start_kind` (`full_start` or `frontier`) and
the immutable parent state ID. Reports reject all `frontier` starts from
full-start completion/progress evidence, including status-only fallbacks and
states geometrically close to the initial spawn. Assisted successes, if any,
must be reported separately. Original near-star detector states are not seeds
for this archive.

Probe actions never enter learner replay or novelty rewards. They are real
interaction and are explicitly recorded as `real_validation_decisions`.
`physical_decisions_this_run` adds them to new learner decisions; legacy
`real_decisions` retains its learner-counter meaning. Imagined actor transitions
remain separate. The probe's unrecorded visual history is discarded on restoring
the original state, leaving the learner's observation/context unchanged.

## Bounded live pilot

Each game received exactly 1000 new learner decisions, plus 25 separately
accounted real probe decisions. Both runs completed and saved new checkpoint
manifests without stderr errors. Galaxy 1 accepted five snapshots and started
one assisted episode; Galaxy 2 accepted five snapshots across its workers and
started twelve assisted episodes. All other resets remained full starts.
This verifies saving, screening, restoration, mixed resets, learner updates and
accounting; it does not establish improved navigation or reduced time to solve.
No mission completion is claimed. The complete telemetry, archive lineage and
counts are in [the pilot record](assets/performance/frontier-pilot-20261025.json).

The maximum pilot overhead was prebounded by sixteen candidates times five
probe decisions per worker/scenario. Actual accounting was 1025 physical
decisions for each game. Longer studies require a new bounded design with
matched full-start controls and independent full-start evaluation.

## Recording actual policy behavior

```powershell
.venv\Scripts\python.exe -m smg_rl.cli --config config/staged-smg2-terminal_reward_weight.yaml record-world-policy --checkpoint runs/staged-smg2/terminal_reward_weight/world-model-20261025-010019-701413/world-final.pt --variation 0
```

Install `ffmpeg` on PATH; `libx264` encoding is required. Six videos cover
control/candidate for Spinning, Double Time and Sweet Sweet Galaxy. Each contains
all three predeclared evaluation seeds 11, 23 and 47, with at most 600 decisions
per attempt. No attempt is selected because it looks better. Recordings use
full native RGBA capture every two emulated frames (30 samples/s), while the
policy chooses an action every twelve frames. Silent H.264 MP4 playback follows
emulated game time rather than the speed of collection. Overlays show actions,
coordinates, health, death and the RAM completion flag.

The existing full-capture OpenCV grayscale/crop/resize path supplies policy
observations during filming; it may differ slightly from compact native CPU
preprocessing used in production training. All paired videos share this path.
Recording forces full-start resets, disables frontier selection/replay and
does not update any model, actor, critic or training counter. These are policy
evaluations, not additional training episodes. All six original MP4 files were
verified with ffprobe and decoded sample frames; paths, hashes and outcomes are
in [the video manifest](assets/performance/policy-videos-20261025.json).
Movies remain local under ignored `runs/` directories, alongside their CSV
trajectories and `recording.json`; they are not committed with ROM/state data.
