# Return to IQN after the genetic plateau

**Superseded active allocation:** the owner subsequently selected
[full-visual genetic evolution](genetic-visual.md). This IQN return is
intentionally paused and retained; do not resume it automatically.

On October27 the owner requested resuming IQN and asked whether frozen vision
caused the genetic experiment's plateau. The GA is completed and retained;
only Sweet Sweet IQN resumes, using six collectors and one shared policy.

## Checkpoint and initial allocation

Resume the exact latest audited IQN pause at **523004 real decisions,130239
optimizer updates and8000 completed attempts**, preserving online/target
weights, optimizer, replay, mutation-independent learner RNG and exploration
schedule. No evolved genetic head is imported into IQN. The older481532-real
feature checkpoint used by GA remains immutable, as do all genetic archives.

The initial engineering allocation is a finite **two-hour block**, not an
indefinite run: health at most3072 new decisions/300 seconds, then reviewed
30-minute,60-minute and remainder segments,7200 accounted seconds including
health,720000-new-real and8-dispatch safety ceilings. Every completed segment
passes the existing hash/replay/CSV/reward/attempt audit before its successor.
Failures, owner stops or memory pressure retain partial evidence without blind
retry. No new final-policy evaluation or extra allocation is implicit.

The launcher reuses `scripts/run_smg1_six.py` with explicit independent state,
stop and log paths, and re-audits the retained owner-stop checkpoint before
dispatch. Original ledgers/stop files are preserved. The active state is
`.local/iqn-return-20261027-state.json`; create
`.local/iqn-return-20261027.STOP` to stop it. Inspection is read-only:

```powershell
.venv\Scripts\python.exe scripts/run_smg1_six.py snapshot --state .local/iqn-return-20261027-state.json
```

TensorBoard continues distinct learner runs and cumulative reward, sampled
distance and attempt series; the exporter explicitly watches this new ledger.
Its source attempt indices were reconciled (4081 assisted,3919 canonical) to
avoid gaps or counting historical attempts as new collection. A one-time
completion review is scheduled for09:40 BRT, without another automatic block.

## What the frozen-vision result does and does not establish

The genetic pilot froze the entire loaded IQN feature network and evolved only
the final4104 action-head parameters. It could change how existing visual
features map to actions, but could not adapt the representation to distinguish
new obstacles. This is a plausible restriction, especially away from the
states encountered during its source training. It is not an isolated causal
finding: no matched genetic run with a trainable/evolved visual encoder was
performed, and the fixed mutation/selection behavior, partial route coverage,
fitness geometry and control also remain possible limitations.

IQN's online IMPALA convolutions and projection layers are already trainable.
`Adam(self.online.parameters())` includes the encoder; only the separate target
network is intentionally gradient-free. The old trained checkpoint contains
state dictionaries, not the temporary GA evaluator's `requires_grad_(False)`
flags. Resuming through `DistributionalAgent.load` constructs a fresh trainable
online network before loading the retained weights and optimizer.

Runtime status now records `visual_encoder_trainable` and a single first-update
probe on the first convolution kernel: observed gradient L2, whether actual
weights changed and source/observed optimizer update counts. The probe clones
only one small kernel once per segment and adds no recurring per-update
measurement after that probe. Portable loop tests verify nonzero visual
gradients and actual weight changes for IQN/IMPALA and Rainbow/Nature fixtures;
these fixtures establish engineering behavior, not trained mission success.

The original IQN reward still credits only the verified opening, with native
star/death/HP terms. It was not silently replaced with GA's goal projection.
Consequently, returning to gradient learning does not fix the lack of dense
guidance beyond the opening.100% opening and geometric projection are not
native mission percentages. Canonical and assisted RAM stars remain separate.

## Actual initial verification

The resumed online run confirmed that the first convolution kernel changed on
optimizer update130240 (source130239), with gradient L2 **0.0432028212** and
`visual_encoder_trainable=true`. This is measured from real replay optimization,
not inferred only from configuration. The actual six-emulator collectors were
healthy and counters advanced.234 portable tests and Ruff passed, including
the visual weight-update assertions. See [the dispatch/update evidence](assets/performance/iqn-return-authorization-20261027.json).
