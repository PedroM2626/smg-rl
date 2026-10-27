# Renderer comparison, dependency security, and bounded recovery

> **Historical context:** Experiment status, authorizations, schedules and live counters in this document refer to their stated dates. Later work may supersede them; consult [current evidence and next experiments](current-status.md) and check actual process state before acting.

This October 22, 2026 revision retains native protocol 3, 12 frame boundaries
per decision, 54 actions, four stacked 84 x 84 observations, the bounded
exploration reward, and the same PPO settings. It selects Vulkan after a
controlled renderer comparison and strengthens checkpoint/process recovery.
It does not establish a solved mission or promise a large reduction in sample
complexity. More decisions per second and fewer decisions to learn are distinct
measurements.

## Performance experiment

The test machine is an i9-14900HX with 32 logical processors, an RTX 4070 Laptop
GPU with 8 GB VRAM, Windows and NVIDIA driver 556.29. Torch 2.14.1+cu126 and
SB3/sb3-contrib 2.9.0 are the updated production dependencies. The renderer
comparison uses that updated stack. It measures environment steps and curiosity, without PPO.
Both production learners were stopped during these isolated comparisons.

Each renderer ran 128 fixed-action decisions with curiosity enabled in two
passes, reversing renderer order on the second pass. Startup and reset time
are excluded from step wall time, and the action sequence is retained in the
raw records. Rates below pool decisions over step wall time, rather than
averaging rate estimates. This is a small, single-machine engineering test;
it does not estimate confidence intervals or multi-seed learning performance.

| Renderer | Galaxy 2 decisions/s | Galaxy 1 decisions/s |
| --- | ---: | ---: |
| Direct3D 11 | 19.69 | 16.56 |
| Vulkan | 22.97 | 19.58 |
| Direct3D 12 | 18.57 | 17.25 |

Vulkan improved the pooled isolated rates by approximately 16.6% and 18.3%
over Direct3D 11, corresponding to about 14.2% and 15.5% less wall time for
the same number of measured environment decisions. These rates omit inference,
PPO, checkpointing and concurrent-emulator contention. They must not be
multiplied by worker count to predict production performance.

Raw evidence: [Galaxy 2](assets/performance/v4-graphics-smg2.json) and
[Galaxy 1](assets/performance/v4-graphics-smg1.json). Reproduce separately from
production training:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_execution.py --config config/native.yaml --graphics --steps 128 --repeats 2 --output .local/graphics-smg2.json
.\.venv\Scripts\python.exe scripts/benchmark_execution.py --config config/smg1-native.yaml --graphics --steps 128 --repeats 2 --output .local/graphics-smg1.json
```

An exploratory live affinity comparison restricted only owned Dolphin
processes to the faster CPU efficiency class, discovered with Windows CPU-set
information. It did not establish a consistent benefit, so production retains
`cpu_affinity: auto`. The optional `performance` mode respects the process's
existing allowed CPUs and avoids assuming that CPU indices always identify
performance cores. Multi-group hosts keep their existing affinity.
[Microsoft's CPU-set definition](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information)
specifies the efficiency-class interpretation. These live windows are
exploratory, with different policies/actions and some maintenance overlap;
they are not the controlled renderer comparison.

An attempted removal of the native DSU server was rejected. Although native
input masks bypass UDP updates, the existing Dolphin patch instantiates the
controller through DSU discovery. Preliminary serverless measurements are
invalid as gameplay throughput evidence and are excluded from the table.
Two emulator exits occurred during those rejected startup/reset experiments;
their underlying native cause was not established. The server was restored,
and the selected path subsequently passed movement, terminal-state calibration
and deliberate process-failure recovery. Removing DSU safely requires a native
virtual-device implementation, rather than merely disabling its server.

## Observation and telemetry validation

Each mission replayed 64 identical actions from its full-start savestate on
Direct3D 11 and Vulkan. All sampled coordinates matched exactly, as did death,
completion and health signals. Movement from reset exceeded 697 world units;
the separate movement calibration checks also exercised the controls. Renderer differences
were small but not universally pixel-identical:

| Mission | Mean absolute intensity error | Maximum intensity error |
| --- | ---: | ---: |
| Spinning | 0.0002524 | 1 |
| Double Time | 0.0002458 | 3 |
| Sweet Sweet | 0.0001617 | 6 |

Values are over 64 grayscale 84 x 84 observations per mission. They describe
these bounded trajectories, not every scene. No image reduction, action-repeat
change, or game-speed hack is responsible for the rate increase. Source and
reference images remain local. See [Galaxy 2 replay](assets/performance/v4-images-smg2.json)
and [Galaxy 1 replay](assets/performance/v4-images-smg1.json).

Both games also passed fresh movement, pause, death and near-star completion
calibration on Vulkan. Those near-star collections remain detector fixtures,
excluded from training success. A legitimate fresh native frame can be uniform
during a fade. The client now accepts that image after the protocol/layout
checks instead of treating low variance alone as an emulator failure. Invalid
layouts, stale image sequences, incorrect PIDs and inexact frame acknowledgments
remain errors.

## Reliability changes

New policy/curiosity checkpoint pairs are written to unique staged files. An
`.incomplete` marker exists throughout publication; curiosity, policy and the
SHA-256 `.pair.json` manifest are then renamed into place before removing that
marker. Recovery rejects interrupted publication, hash mismatch, corrupt ZIP
members or inconsistent timestep metadata. Existing complete local pairs without
a manifest remain resumable after archive checks. Existing checkpoints are never
overwritten by this helper. Periodic retention removes each manifest with its
policy/curiosity pair. This is a fail-closed publication protocol, not a claim of
filesystem-wide atomicity or durability against every power-loss scenario.

The measured collector bounds initialization/RPC/step receives at 120 seconds
by default and bounds normal worker shutdown at five seconds before terminating
its remaining owned process trees. Worker wrappers close their environment on
unexpected exit as well as the ordinary close command. This prevents a failed
step from leaving its emulator or memory/socket handles unclosed. Native command
timeouts remain 30 seconds, and the supervisor retains its 240-second stale
learner watchdog. Eight consecutive attempts without a new checkpoint now produce
a visible failed state instead of an infinite retry loop. Progress resets that
consecutive-failure counter.

Profile preparation failures now enter constructor cleanup, emulator shutdown
has a terminate/kill fallback, and native bridge close is idempotent. The legacy
DSU endpoint checks version, declared length and CRC, limits peers to eight,
wraps its 32-bit packet counter and tolerates disconnected clients. Bad packets
cannot bypass its scheduled state-send path. Its bind remains loopback-only.
Owned worker profiles explicitly disable analytics. See the repository
[security policy](../SECURITY.md) for the remaining trust boundaries.

## Dependency advisory audit

An isolated pip-audit environment inspected the installed training environment.
The initial path audit flagged pip 24.0, but skipped CUDA Torch because its local
version was not found on PyPI. A separate normalized `torch==2.6.0` query exposed
the additional Torch advisories. Skipped dependencies are therefore explicitly
accounted for, rather than reported as clean.

The training environment now uses pip 26.2.2, Torch 2.14.1+cu126,
Stable-Baselines3 2.9.0 and sb3-contrib 2.9.0. CUDA matrix multiplication,
recurrent forward/backward, existing-checkpoint continuation and the collector
tests were verified. SB3 2.9 uses restricted tensor loading; its metadata still
uses cloudpickle and must come from a trusted source. Curiosity loads continue
to specify `weights_only=True`. The CUDA dependency lock and project minimum
versions were updated together; no system driver update was performed.

The retained [audit summary](assets/performance/v4-security-audit.json) records
advisory identifiers, fixed runtime versions, skipped items and coverage limits.
The post-update installed-package audit and normalized Torch query reported no
known advisories at the time of inspection. This does not prove the absence of
unknown vulnerabilities, audit gaps, malicious model metadata or upstream native
issues. `scripts/audit_dependencies.py` makes the normalized installed-version
query reproducible; instructions are in `SECURITY.md`.

## Validation and production continuation

The final portable suite passed **98 tests**, Ruff and `pip check`. New cases
exercise interrupted checkpoint writes, hash corruption, malformed DSU packets,
variable-sized CPU topology records, bounded hung-worker teardown and legitimate
uniform transition images. The existing collector tests compare reset, terminal
observations, observation ownership and worker ordering against SB3 itself.

A test-owned Galaxy 1 emulator was deliberately terminated after a complete
64-step pair. The supervisor resumed in attempt two and completed the original
256-step budget using the new dependency stack and Vulkan. The
[recovery record](assets/performance/v4-recovery.json) remains outside production
success accounting. Protocol and calibration records are summarized in
[validation evidence](assets/performance/v4-validation.json).

Production stopped gracefully at 803876 Galaxy 2 decisions and 214719 Galaxy 1
decisions. The continuation requests only 197264 and 785281 additional decisions,
respectively, retaining the original targets 1001140 and 1000000. Four Galaxy 2
workers and one Galaxy 1 worker resume from those final legacy pairs, which
remain local. The Galaxy 2 count is shared across both variations. The next
periodic checkpoints use the new pair manifests. Resets/recurrent state and
partial rollout limitations remain as documented in the preceding revisions.

The [dated production snapshot](assets/performance/v4-runtime-snapshot.json)
records actual simultaneous throughput, attempts, checkpoint integrity and the
emulator inventory after continuation. That early snapshot observed 807012
Galaxy 2 and 215679 Galaxy 1 decisions, both on supervisor attempt one, with
five owned emulators and newly verified manifest-backed checkpoint pairs.
Its one-minute windows measured 16.39 aggregate Galaxy 2 and 5.09 Galaxy 1
decisions/s; cumulative early-run rates were 15.39 and 4.52. GPU utilization was
91%, with other user applications also using the GPU. These loaded evening
observations are slower than the earlier daytime production observations and
are not evidence of an end-to-end speedup. They cannot isolate renderer,
dependency, thermal, policy or competing-application effects. The controlled
single-worker comparison is the basis for the renderer choice; a large total
training-time reduction has not been demonstrated.

It is a historical observation, not a live dashboard. Repository code/docs remain MIT; the unchanged patched Dolphin
binary retains its separate GPL licensing. ROMs, savestates, full models and
compiled binaries remain ignored and local.
