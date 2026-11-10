# Center of the Universe: pre-boss route recording

This is a separate, **validation-only** Super Mario Galaxy USA mission. Training
is disabled in `config/smg1-center-validation.yaml`. Slot `data/RMGE01.s03` is
the full-start reference; slot `data/RMGE01.s04` is the owner-selected position
before the Bowser fight. Slot 4 is not a star or an automatically verified
mission terminal. Do not treat loading it, reaching its approximate coordinates,
or marking a recording as an agent success.

A read-only offline inspection decoded the two Dolphin 2609a savestates and
checked the existing RMGE01 executable signature and object graph. Both have
stage `KoopaBattleVs3Galaxy`, scenario 1, HP 3, grounded Mario, no death and no
star. Their XYZ coordinates and immutable savestate hashes are in
`config/smg1-center-calibration.json`; see the
[inspection evidence](assets/performance/center-offline-snapshot-20261028.json).
This is not a live load or a demonstration of reachability. A live smoke test
remains necessary; it was not run concurrently with the existing six-worker
training job. The 300-world-unit three-dimensional marker radius is provisional
and may include a different surface or miss a valid alternative route.

## Record

1. Once other Dolphin instances have stopped, double-click
   `record-center-of-universe.bat`. The recorder intentionally refuses to run
   alongside other Dolphin processes. It opens its own isolated profile and
   starts from a copy of slot 3.
2. Focus **SMG RL - Human recorder** and press Enter. Use W/A/S/D, left mouse
   to jump, middle mouse to spin, and Shift to crouch. The game pauses if the
   recorder window loses focus or the video encoder falls behind.
3. Navigate to the location represented by slot 4. Press **F8** while Mario is
   alive, grounded and within the provisional 3D radius. F8 only records an
   owner-marked goal candidate; it does not confirm a star. Press **F9** to stop
   without saving. A death discards the attempt and restores slot 3.
4. The console prints `goal_candidate`; the local folder contains a gameplay
   video, sampled exact inputs, policy observations, XYZ/HP trajectory, top-down
   plot and hashes. The manifest records `candidate_goal`,
   `full_start_ram_star: false` and `goal_replay_verified: false`.
5. Replay the candidate with

   ```powershell
   .venv\Scripts\python.exe scripts/verify_human_demonstration.py <goal-candidate-folder> --max-seconds 1800
   ```

   The verifier checks source and goal fixture hashes, replays from slot 3,
   compares every saved observation and position, and checks the marked
   endpoint. A `goal_candidate_replayed` result demonstrates same-state
   reproduction only. Inspect the video to confirm the intended pre-boss
   location; no reference-state proximity alone confirms the mission endpoint.
   Until live smoke testing and a real human candidate are completed, the new
   recorder is implementation-tested but not live-validated.

Human recordings contain simultaneous two-frame button inputs; they are not
directly compatible with motor8 twelve-frame training replay. Converting and
validating a route and any future training require separate work. Assets and
recordings remain local and are not committed.

## Proposed training setup and gate (draft for owner review, November 10, 2026)

Nothing below is implemented as a training run. The draft config
[`config/center-bbf-draft-20261110.yaml`](../config/center-bbf-draft-20261110.yaml) has `training.enabled: false`, and
the environment still refuses to build because `config/smg1-center-calibration.json` is not marked `validated`.
That guard is deliberate and stays until the owner approves the points under "Decisions for the owner".

What exists after the November 10 work: the owner's pre-boss recording (7,103 samples, slot 3 start, F8 marker) replays
exactly (no position or observation divergence over 7,104 samples), ending 180 units from the slot-4 reference with
2 of 3 health, grounded and without a star. A projection route of 274 waypoints (`config/routes/smg1-center-projection-v1.json`,
530-unit spacing, about 149,400 units of arc) was cut from it.

**Learner and reward (same recipe as the Double Time arms, no demonstrations).** BBF-lite, `motor8-longjump-v1`, 0.1 per
newly passed route waypoint (27.3 in total, 273 segments), 50 for the gate event, 0 for death, damage and timeout,
no step cost. Episodes are capped at 36,000 frames (3,000 decisions); the human run took about 1,185 decisions.

**Proposed gate event (the "star" of this mission).** A policy attempt succeeds when, in one decision, Mario is alive,
grounded and within the calibrated 3D radius of the slot-4 reference (300 units around
(40912.6, 28495.4, 132543.5)), and the attempt was not invalidated by a teleport-size position jump. It is reported as
"pre-boss goal reached", never as a star or a Bowser victory, because the Bowser fight is outside the route.

**Proposed final check.** The frozen greedy policy from the canonical slot-3 start, with no anchors and no scripted action,
reaches the gate event in at least one run and then in repeated runs; the number of runs and the pass rule are for the
owner to set, since one run proves little (the Sweet Sweet gate was one run and is listed with that limitation).

**Not built yet, and why it matters.** (1) The environment has no terminal for "reached a marker without a RAM star";
only a RAM star ends an attempt as completed. (2) `scripts/prepare_anchor_bank.py` requires a RAM-star recording, so anchors
for this mission need a goal-candidate mode. (3) The calibration `validated` flag.

### Decisions for the owner

1. Is the end of the recording (the video) the place the agent must reach?
2. Is a 300-unit 3D radius right, given the owner's run ended 180 units away and 162 units higher? A tighter radius is
   harder; a looser one may be met by a different surface.
3. Must the policy stay grounded at the marker, and is a single frozen greedy run enough or should it be repeated?
4. May the validation flag be set after (1) to (3), and the environment get a marker-reached success event?
