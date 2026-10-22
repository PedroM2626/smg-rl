# Documentation index

Documentation describes the checked implementation, measured engineering
validation, and an experimental protocol. A method being specified here does
not mean its ablations or full-course learning results have been obtained.
Use run artifacts for live status; historical observations below are dated
evidence rather than continuously updated dashboards.

## Start and operate

| Need | Document |
| --- | --- |
| Installation, supported tasks, rewards, and recommended quick start | [Main README](../README.md) |
| Exact command arguments, backend, and side effects | [CLI reference](cli.md) |
| Local ROM and savestate filenames and roles | [Asset guide](../data/README.md) |
| Native build, shared-memory capture, exact frame stepping, profiling, and benchmarks | [Native backend](native-backend.md) |
| Protocol v2, curiosity batches, bounded exploration, and the stagnation audit | [Optimization revision](optimization-v2.md) |
| Current protocol v3, event wakeups, native preprocessing, and worker barrier profiling | [Native pipeline and collection](optimization-v3.md) |
| Vulkan comparison, dependency audit, transactional checkpoint pairs and bounded recovery | [Renderer and security revision](optimization-v4.md), [security policy](../SECURITY.md) |
| Hidden rendering, isolated inputs, CUDA, timing, and throughput | [Background training](background-training.md) |
| Two learners, resource measurements, status, and cross-game reports | [Concurrent training](concurrent-training.md) |
| Health penalties, checkpoint recovery, and measured damage evidence | [Damage and recovery](damage-and-recovery.md) |
| Practical reward tuning, evaluation, and troubleshooting | [Training guide](training.md) |
| Visual replay, recurrent world model, prediction gates and imagined actor/critic experiment | [World-model implementation](world-model.md) |
| Authorized online world-model experiment, initial counters and recovery procedure | [Live world-model experiment](world-model-live.md) |

## Understand and reproduce

| Need | Document |
| --- | --- |
| POMDP, action/observation spaces, reward equations, novelty, PPO, and limitations | [Methodology](methodology.md) |
| Seeds, baselines, ablations, metrics, uncertainty, and evidence register | [Experimental protocol](experiments.md) |
| Module architecture, environment, configuration, checkpoint semantics, and outputs | [Reproducibility](reproducibility.md) |
| Galaxy 2 pointer chains, executable checks, terminal states, and derivation | [Galaxy 2 memory profile](memory-profile.md) |
| Galaxy 1 pointer chains, health, detector checks, and integration | [Galaxy 1 report](smg1.md) |
| Historical portable tests, live detector tests, smoke learning, and failures | [Validation register](validation.md) |
| Primary research and upstream provenance | [BibTeX bibliography](references.bib) |
| Original project license and material outside its scope | [MIT License](../LICENSE), [third-party notices](../THIRD_PARTY_NOTICES.md) |

## Evidence boundaries

- Portable tests verify code behavior without booting Dolphin or a ROM.
- Near-star positive checks and deliberate falls verify live detectors.
- The measured Galaxy 1 3-to-2 HP trace verifies health reading; reward replay
  verifies the corresponding penalty, rather than learned damage avoidance.
- Smoke training verifies collection, updates, and checkpoint persistence.
- Recovery observations verify continuation after an infrastructure failure.
- Full-course success requires a full-start trajectory and the game-specific
  completion event. Geometric progress and novelty returns alone do not prove it.
- The multi-seed comparative study and ablations are a protocol, not reported
  completed experiments.

When changing behavior, update the relevant specification, command reference,
configuration example, and validation record together. Keep live counters in
reports rather than hardcoding them into the README. Preserve negative evidence
and timing revisions when reporting performance.
