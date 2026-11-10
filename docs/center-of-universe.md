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

## Training setup and gate (implemented November 10, 2026)

The owner asked for this mission to follow the Sweet Sweet setup with the end location taking the place of the star.
The owner's pre-boss recording (7,103 samples, slot 3 start, F8 marker) replays exactly (no position or observation
divergence over 7,104 samples) and ends 180 units from the slot-4 reference with 2 of 3 health, grounded and without a
star. A projection route of 274 waypoints (`config/routes/smg1-center-projection-v1.json`, 530-unit spacing, about
149,400 units of arc) was cut from it, and `config/smg1-center-calibration.json` is now marked `validated`.

**Gate event.** The environment option `goal_marker` (`enabled`, `require_grounded`) ends an attempt as a success when
Mario is alive, grounded and within the calibrated 3D radius (300 units) of the slot-4 reference
(40912.6, 28495.4, 132543.5) after a step that did not move farther than `max_step_distance` (a teleport cannot trigger
it). The completion reward is paid and the attempt is logged with `goal_reached: true`. The RAM star flag is untouched,
so `ram_star` stays false for this mission: the result is "pre-boss goal reached", never a star or a Bowser victory,
because the Bowser fight is outside the route.

**Learner and reward.** [`config/center-bbf-start-20261110.yaml`](../config/center-bbf-start-20261110.yaml): BBF-lite,
`motor8-longjump-v1`, no demonstrations, 0.1 per newly passed route waypoint (27.3 in total, 273 segments), 50 for the
gate event, 0 for death, damage and timeout, no step cost, canonical slot 3 starts, 3,000 decisions per episode (the human
run took about 1,185).

**Final check.** As for Sweet Sweet: the frozen greedy policy from the canonical slot 3 start, with no anchors and no
scripted action, reaches the gate event. One run proves little and the Sweet Sweet gate was listed with that
limitation, so the pass rule should say how many repeated runs count.

**Anchors.** `scripts/prepare_anchor_bank.py --goal-candidate` replayed the recording again and published 8 validated anchors (of 9 candidates; one failed the idle check) under `.local/center-anchors-20261110`, used by [`config/center-bbf-anchors-20261110.yaml`](../config/center-bbf-anchors-20261110.yaml) (uniform anchors plus 25 percent canonical starts). The bank is a set of spawn points, never a demonstration, and the final gate is always measured from the canonical start.

Assumptions taken from the Sweet Sweet setup and open to change by the owner: the 300-unit radius (the recording ended
180 units away and 162 units higher), the grounded requirement, and a single greedy run as the gate. The owner has
not yet confirmed the video's final frame; the video was sent on November 10.
