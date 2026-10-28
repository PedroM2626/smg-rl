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
