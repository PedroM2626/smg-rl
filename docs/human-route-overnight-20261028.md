# Human-route overnight continuation, October 28–29

The owner requested longer training of the current configuration and a review at
07:00 America/Sao_Paulo on October 29. This single Sweet Sweet IQN/IMPALA
continuation uses six collectors, the audited candidate checkpoint at 99,912 real
decisions and 713 completed attempts, and its own optimizer, replay, RNG, reward,
control and full-start distribution. It is exploratory: the latest paired one-hour
comparison had no clear winner and no frozen-agent RAM stars. The human route
reward has not been independently validated as a complete navigation metric.

The supervisor `scripts/run_human_route_overnight.py` uses a 300-second health
block followed by at most 1,800-second blocks, each with an explicit absolute
collection deadline of `2026-10-29T10:00:00Z`. At most 24 jobs are permitted; a
120-second reserve is kept before the deadline for closing and checkpointing. A
block is followed by exact signed checkpoint, replay, attempt, reward and CSV
accounting checks before the next block. A six-GiB available-memory dispatch
check and the original in-training memory guard remain active. Stops, partial
collections, failed audits, insufficient resources or changed configuration halt
the supervisor for review; they must never be blindly retried. Each block resets
emulator states, so this is not a bitwise uninterrupted run.

The original checkpoint and all ROMs, profiles, and savestates remain local and
are never committed. State, commands, PID/create times, source ancestry and block
evidence are persisted in `.local/human-route-overnight-20261028/state.json`.
The supervisor checks sampled trajectories against ordered human-observed XYZ
landmarks only as candidate observations; proximity is not proof of a visit or a
mission percentage. Full-start native RAM stars remain the success signal.
Collection outcomes, changing exploratory policies and frozen evaluations must
be reported separately. At 07:00 the owner-requested review decides whether
further work is warranted based on inspected real trajectories, failures and
checkpoint integrity; no automatic training extension is allocated. The
reviewer may run a bounded frozen evaluation of the last clean checkpoint if
needed before making a decision, recording its separate interaction count.
The invalid first-segment projection remains withdrawn.
