# Frozen density review and guarded four-worker continuation

On October 29 the owner requested memory diagnosis, frozen video evaluations
and further training if warranted. Both final checkpoints from the interrupted
density experiment passed agent and replay SHA-256 checks. Four frozen recording
jobs produced eight full-start episodes, each with exact policy image and
control overlays, without changing agent/optimizer/replay files.

| Policy | Mode | Episodes | Outcomes | RAM stars |
| --- | --- | ---: | --- | ---: |
| Dense candidate | Greedy | 1 | timeout after 228 decisions | 0 |
| Dense candidate | NoisyNet/random quantiles, epsilon 0.1 | 3 | 2 timeouts, 1 death | 0 |
| Control | Greedy | 1 | death after 16 decisions | 0 |
| Control | NoisyNet/random quantiles, epsilon 0.1 | 3 | 3 deaths | 0 |

Direct inspection of the greedy candidate video shows Mario on the first pink
belt at decision 50, then back at the initial platform by decision 220 before
the timeout. The greedy control falls to death. No other landmark match was
confirmed in the candidate's stochastic episodes. The control's last two
stochastic trajectories were excluded from the conservative landmark check
due to discontinuities. The single training seed and reused identical Wii
savestate limit any conclusion; these runs do not prove generalization or a
complete mission. Videos and trajectory files are local under `runs/`, and the
[audit](assets/performance/density-frozen-review-20261029.json) retains hashes,
episodes and local video paths.

## Memory diagnosis

Read-only `NtQuerySystemInformation(SystemPoolTagInformation)` snapshots showed
approximately 7.26 GiB charged to the `EtwD` paged tag and 3.69 GiB to the
`Etwr` nonpaged tag, representing most of the system's kernel pools. These tags
are associated with Windows event tracing accounting; they do not identify which
process, session or driver caused the allocations. During the one-emulator
evaluations the tags grew from 7,774,765,632 / 3,951,505,760 bytes at 12:11:03
UTC to 7,797,688,064 / 3,967,236,080 bytes at 12:13:54 UTC. This is an
observation across evaluation activity, not proof of a leak or causality.
The previous Realtek helper was approximately 47 MiB private rather than several
GiB. `logman query -ets` was run without administrative elevation and displayed
no sessions; that does not prove sessions are absent. No system ETW session,
driver, third-party service or user application was stopped or reconfigured.
Further driver/session attribution requires appropriate Windows diagnostics.

## Continuation

The owner requested continued work. To reduce memory demand, the dense candidate
is continued with four rather than six workers. The worker-layout migration
retains the existing frame-ring capacity and all replay frames, links, transition
arrays, priorities and sampling RNG. It verifies that learner weights, target,
optimizer, RNG and counters remain equal; no rewards are relabeled and no source
files are modified. The migration report and checkpoint remain local. Changing
collector count and restarting environments changes future sampling and is not
bitwise equivalent to uninterrupted six-worker training.

`scripts/run_density_four_worker_continuation.py` is the single supervisor for
up to 7,200 seconds including per-block checkpoint cleanup, with a 300-second
health block and later blocks of at most 600 seconds. Its independent wall
ceiling is three hours. A 120-second cleanup reserve, 2 GiB in-training physical
guard, 6 GiB pre-dispatch physical/commit headroom, signed profile, audited
checkpoints and no blind retry remain in force. Read-only ETW pool and physical
memory snapshots are stored between blocks. The weaker original guard is not
used for this continuation. New elapsed time, decisions and attempts are
separate from the 188,070-decision / 2,450-attempt source branch. At completion
review real trajectories, native RAM stars, deaths, hashes, resource trends
and optionally evaluate the final frozen policy with video; collection maxima
and coordinate proximity are not level-completion percentages. A resource stop
requires diagnostic review, not an automatic restart.
