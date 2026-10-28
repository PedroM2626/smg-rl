# Bounded expanding-route SMG1 trial, October 2026

This owner-authorized exploratory trial tests route discovery during IQN/IMPALA
training in Sweet Sweet Galaxy. It uses six real Dolphin workers from the
canonical starting savestate, with eight exclusive actions, 12 frames per
decision, no world model and no intrinsic novelty. The trainable visual encoder
and online policy weights come from the audited IQN checkpoint at
`runs/model-free/smg1-six-two-hour/dqn-20261027-114555-611503/checkpoint-final`.
The source is retained. Its online network is transferred to a fresh branch;
target weights are synchronized and optimizer, counters and replay are reset.
Old and new signatures must differ. Existing checkpoints and replay are not
rewritten. Changing the reward scale makes the transferred Q values initially
miscalibrated; this is an exploratory transfer, not an equivalent continuation.

The existing route is an opening, not a complete level. Regional reward is
credited only for newly reached, grounded, ordered route distance. The fixed
50,000-world-unit denominator replaces the prior opening-length denominator;
consequently old segments retain the same contribution across later versions,
and new segments can add credit without the old 100% saturation. This also
makes the initial shaping much smaller than in the original checkpoint. Death
and timeout remain -2, nonfatal HP loss -0.25, and a RAM-confirmed star +20.
The denominator and route length are not native mission percentages. No route
version is allowed to reach the denominator silently.

After each stopped, audited block, a proposal uses full-start completed traces
that pass the old endpoint and continue in continuous sampled, grounded movement
without damage, teleport-like displacement or return to existing segments. Two
separate episodes must reach nearby candidate points. Their saved trajectories
are then replayed through the actual route implementation and both must earn
credit beyond the former endpoint. The two episodes may restore exactly the same
Wii state; they are repeated observations, not independent environment tests.
Samples occur at decision boundaries, so contact between samples is not proven.
The intentionally conservative heuristic may miss valid jumps and branching or
curved passages; passing it does not prove a point leads to the star.

Versions and evidence are immutable under `.local/auto-route-smg1-20261028`.
Each accepted point appends an actually observed coordinate, keeps old points
unchanged, and generates a distinct signed profile and fresh replay/optimizer.
No policy learns across incompatible reward signatures in the same replay.
All collected traces and checkpoint artifacts are retained. Only collection
records, not a frozen-policy test, establish the best observed path in this trial.
Only real full-start terminal RAM signals establish a completed star.

The supervisor uses an initial health block of at most 3,072 real decisions or
300 seconds, followed by blocks of at most 180,000 real decisions or 30 minutes.
It audits completed blocks and saves their provenance before a successor. It
stops after two completed 30-minute main blocks with neither a repeated valid
route extension nor a RAM-confirmed full-start star, on failure, owner stop, or
the absolute deadline `2026-10-28T11:00:00+00:00` (08:00 São Paulo). The health
block is not counted as stagnation. A failed or partial block is not retried.
The 08:00 report is a separate scheduled follow-up and does not extend training.

To inspect: `.venv\Scripts\python.exe scripts/run_dynamic_route_night.py snapshot`.
The state records source and version hashes, exact commands, run/controller
identity, block accounting, decisions and proposal evidence. To request an early
stop, create `.local/dynamic-route-night-20261028.STOP`; the supervisor forwards
the stop to the current owned run. TensorBoard remains at its configured path.
