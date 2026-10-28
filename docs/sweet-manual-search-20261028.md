# Bounded scripted Sweet Sweet search and mapping

The owner asked whether the assistant can complete and map Sweet Sweet Galaxy.
This is a bounded attempt to construct a real full-start action sequence; it is
scripted search over game savestates and is not a reinforcement-learning policy.
No star is claimed without a live, validated RAM terminal and a separate replay
from the canonical starting slot. A geometric projection is not a native mission
percentage and does not measure a winding route's completed fraction.

The initial search begins at the deepest previously reproduced first-chocolate
candidate: the exact state and 47 full-start actions were verified in
[the preceding intervention](chocolate-frontier-search-20261028.md#search-and-physical-validation-results).
That snapshot is on a moving conveyor and **failed** the pre-existing stationary
archive gate. It is an assisted dynamic exploration fixture only; no stability
gate or curriculum bank is changed. Saved search snapshots carry parent action
lineage, coordinates, health, scene timing and SHA-256. New 3D cells are a
search heuristic, not automatically safe route waypoints.

Six isolated Dolphin workers each have at most 48 short trials, 24 decisions
per trial, 24 retained dynamic states and a shared absolute 30-minute boundary.
Actions are the original eight exclusive controls at 12 frames per decision.
Original ROMs, savestates, profiles and checkpoints are preserved. The local
ledger `.local/sweet-manual-search-20261028/state.json` records the supervisor,
process identities and finite budgets. A STOP marker requests an early stop.
All assisted stars, failures and partial data are retained. No training,
curiosity, synthetic experience or imagined transitions are involved.

Each saved state retains its exact action ancestry back to the canonical start.
If a RAM star is observed, stop the search and replay that complete lineage
from slot 1, independently recording video, actions, coordinates, health and
completion RAM. Only successful full-start replay establishes a scripted
completion; success after loading an intermediate state is separately labeled.
If no star is found, select and physically replay a promising partial path from
the full start, record the limits and make a top-down trajectory without
presenting it as a solved mission. Any successor requires review of the bounded
block; no indefinite automatic search is allocated.

## First search and reviewed continuation

The first block failed in five workers when Dolphin still held the private slot-8
file open and Windows rejected its rename with sharing violation 32. Their
partial traces and snapshots remain intact. Before the failure, the six workers
recorded 1,033 physical decisions, 67 finished trial records, 62 dynamic-state
entries (including repeated source entries), and no assisted RAM star; decisions
from interrupted trials remain included in the physical total. One worker
completed its allocated search. Source and user savestates were not replaced.

The deepest observed grounded state was `(474.60, -50.00, -2909.86)`. Its
64-action lineage was replayed from the canonical start in a separate emulator:
position, health and scene frame matched exactly, and no star was observed.
The local video and top-down plot are retained under
`runs/sweet-scripted-replay-20261028/replay-20261028-095921-292727/`. This is
a scripted partial full-start trajectory, not a trained policy or a completed
mission. The snapshot remains a nonstationary search fixture, not an accepted
stable curriculum state.

The native file handler now retries only transient Windows sharing/lock errors,
with a bounded timeout and process-liveness check; unrelated errors still
fail. Four consecutive saves and exact restorations succeeded in a separate live
Dolphin check. The next block is a **new search** rooted at the replay-verified
deeper snapshot, with different random seeds, its own profiles and the same
finite six-worker limits. It is not a blind rerun of failed attempts. Its ledger
is `.local/sweet-manual-search-stage2-20261028/state.json`; retain both blocks,
the failure logs and original source states. Stop and diagnose any recurrence.

The second block completed all 288 planned trials with 4,198 real decisions,
212 deaths, 76 bounded cuts and no assisted RAM star. The deepest retained
grounded state was `(-700.85, -50.00, -3610.62)`; its 89-action lineage was
successfully reproduced from the original full-start slot with matching scene
frame and health, without a star. It is still a scripted partial path. The
[audited counts](assets/performance/sweet-manual-stage2-20261028.json),
[top-down trajectory](assets/performance/sweet-manual-stage2-path-20261028.png)
and [location image](assets/performance/sweet-manual-stage2-location-20261028.png)
are committed; the full video and physical traces remain local.

A third finite search starts from this newly verified deeper fixture, with
independent search seeds, the same six-worker limits and separate files under
`.local/sweet-manual-search-stage3-20261028`. It is a new reviewed block, not
a retry, and must be reviewed before any further search or conclusion.

The third block completed 288 assisted trials with 3,878 real decisions,
238 deaths, 50 bounded cuts and no star. Its deepest retained grounded point,
`(-844.56, -50.00, -4425.86)`, was replayed exactly from the full start in
106 scripted decisions with health 3 and no star. The [counts](assets/performance/sweet-manual-stage3-20261028.json),
[partial path](assets/performance/sweet-manual-stage3-path-20261028.png) and
[location](assets/performance/sweet-manual-stage3-location-20261028.png) are
committed. The video and raw trajectories remain local. The screenshot shows
Mario close to a platform edge; this remains a dynamic search fixture, not a
validated stationary curriculum start.

A fourth finite search uses this verified point with new search seeds and its
own `.local/sweet-manual-search-stage4-20261028` ledger. It has the same six
workers, trial/action/state caps and 30-minute upper boundary. It must be
reviewed before any continuation; no complete mission is yet claimed.

The fourth block completed 288 trials and 4,254 real decisions: 224 deaths,
64 bounded cuts and zero assisted RAM stars. A grounded state at
`(-816.82, -50.00, -5099.48)` was replayed from the original starting slot in
114 scripted decisions with full health and matching position/scene frame;
the episode was deliberately cut at that state and did not collect a star.
The [evidence](assets/performance/sweet-manual-stage4-20261028.json),
[partial top-down map](assets/performance/sweet-manual-stage4-path-20261028.png)
and [observed location](assets/performance/sweet-manual-stage4-location-20261028.png)
are tracked. The video and raw trajectory remain local under
`runs/sweet-scripted-replay-20261028/replay-20261028-101228-172878/`.
The best alive-grounded start-to-goal X/Z projection in that replay was 6.304%;
it is neither native mission progress nor a fraction of the winding route.
The pictured location is at an obstacle and close to an edge; the state has
not passed the stable-curriculum screen. All searches are now stopped.
