# Genetic evolution with unlocked visual parameters

On October27 the owner explicitly requested continuing the GA with the visual
network unlocked. This replaces the intervening IQN return; the live IQN is
stopped and audited before the six emulator instances are reassigned. The first
allocation is a finite two-hour trial, with no automatic IQN resumption or
additional block. All earlier checkpoints, genetic runs and IQN replay remain
immutable and separate.

## What is unlocked

The new `full_visual` genotype contains **1869736 deterministic action-path
parameters**, including **1569184 visual parameters** in every IMPALA
convolution/residual block and visual projection. Quantile embedding and both
advantage layers also evolve. The former head-only experiment evolved4104
parameters. The common value branch cancels from action argmax and is omitted;
NoisyNet sigma parameters are also omitted because deterministic evaluation
does not use them. These are inactive parameters, not frozen visual features.

This remains a pure GA: there is no backpropagation, Adam update, replay fitting,
curiosity or world model. Torch autograd is deliberately disabled. Each
individual supplies its own CNN and decision-path weights through
`torch.func.functional_call` and `vmap`, so inference actually uses its mutated
visual representation. A `requires_grad=false` flag therefore does not mean
that these evolutionary genes are fixed. Portable tests change only CNN genes,
verify a change in logits, and check that another individual's output and the
original source tensors remain unaffected.

One parent batches up to six different policy parameter sets on the GPU; six
Torch-free Dolphin workers evaluate episodes in parallel. GPU genotype caches
are bounded to active candidates and hold strong array references to prevent
recycled identity collisions. Collector candidates remain immutable during an
evaluation: reproduction creates new arrays and never mutates an evaluated
candidate in place. New candidates are checked for finite values and layout
compatibility.

## Initialization and fitness

The source is the completed overnight head-only GA's pending population at
generation319, using its exact original481532-real IQN feature checkpoint.
The latest IQN return checkpoint is retained separately and is not substituted
as a new representation, avoiding that additional initialization change.
The ancestral head genomes are lifted into full action-path genotypes.
Individual0 is exactly the prior champion, with its original CNN. Individuals
1–23 retain their ancestral head variation and receive initial variation in
the other active layers. The old RNG state is continued into this new branch.

No old fitness is reused: changing the genotype layout/visual weights starts a
new generation0 and every individual is evaluated again. Prior attempts,
901806 real emulator decisions and8.0665% geometric champion are ancestry,
not new learning. Gene-layout fingerprints and explicit `policy_mode` tags
prevent treating a full-visual checkpoint as a4104-gene head checkpoint.

The population remains24, with4 unchanged elites, size-3 tournaments, uniform
crossover and Gaussian mutation. To avoid applying the same absolute0.02 noise
to millions of differently scaled weights, active non-output layers use
sigma_l=0.02*max(RMS(theta_l),1/sqrt(fan_in_l)); vector biases use their length
as the fallback fan-in. Output-head parameters retain the previous absolute
sigma0.02. Parameters remain clipped to[-5,5]. This necessary mutation-scale
change is explicit: this is an exploratory intervention, not a perfectly
isolated causal ablation of encoder freezing.

Fitness is unchanged: maximum alive-grounded X/Z projection toward the
calibrated near-star goal, without extra HP/death/time/novelty terms. Input is
the same cropped84x84 grayscale four-frame stack plus scenario identity; eight
exclusive actions and frame skip12 remain unchanged. All fitness episodes use
canonical savestate1, seed11 and at most1200 decisions, without assisted starts.
Geometric projection and100% mapped opening are not native mission percentage.
Completion still requires the native RAM star signal from a full start.

## Budgets, storage and recovery

`scripts/run_visual_genetic_trial.py supervise` persists its pending commands,
stops the actual current IQN supervisor, audits its clean owner-stop checkpoint
and verifies that no Dolphin remains before dispatch. The state is
`.local/genetic-visual-20261027-state.json`. The trainer has7200 seconds, an
absolute dispatch-time deadline, a1000-generation safety ceiling, no early
confirmation reserve and no extra final-evaluation budget. Each completed
generation passes finite-fitness/identity checks before reproduction, without
scheduled emulator restarts. Failure, pressure or owner stop retains partial
data and forbids blind retries. Create `.local/genetic-visual-20261027.STOP`
to stop this trial. Read-only inspection:

```powershell
.venv\Scripts\python.exe scripts/run_visual_genetic_trial.py snapshot
```

The full24-individual population is about171.18 MiB uncompressed. Initial and
final complete population snapshots are written transactionally, with SHA256,
mutation RNG state, generation and layout metadata; original snapshots are not
deleted. Per-generation archives retain fitness, genotype hashes, inherited
trial references and exact reproduction RNG state instead of duplicating the
entire population every generation. Every intermediate population can be
reconstructed from the initial snapshot and these deterministic transitions.
This avoids producing many gigabytes of redundant snapshots during the trial.
An8-GiB disk-headroom guard and existing RAM/commit guards remain active.

Full-visual continuation uses its own final-population loader with hash/layout/
source verification. It preserves its baseline genome and only reuses complete
pending-generation trials whose genome identity matches. Budget-cut trials
remain partial and must be reevaluated under another explicit allocation.
Source/champion files include the full genotype and layout; a conventional IQN
or head-only loader must not silently interpret them as a compatible checkpoint.

## Evidence and limits

The production GPU preflight verified all six candidate outputs were finite and
that lifting the ancestral champion preserved its logits/action argmax within
float32 tolerance. The initialized mutant visual delta L2 was0.7296743484.
This proves that visual genes are mutable and used in inference; it does not
prove that the new search will learn useful visual representations or complete
the level. The larger genotype makes the search harder, and fitness geometry,
control and observation coverage remain alternative explanations for a plateau.

TensorBoard uses a separate run with fitness, projection, opening, return,
sampled path distance and attempts. Trial ledgers include actual visual-genome
delta L2; a champion with zero visual delta may still be the unchanged baseline
and must not be described as learned visual improvement. Morning GA results and
current IQN results remain archived. The completion review must distinguish
baseline reproduction, mutated-visual learning, collection maxima, native
stars, completed/partial attempts and inherited data; no new frozen evaluation
is implicitly allocated. No improvement or zero stars warrants diagnosis, not
an automatic extension.

## Initial live verification

238 portable tests and Ruff passed. The first complete physical generation
passed review; at the dispatch audit46 real trials had completed,44 of them
with nonzero visual-genome delta. The unchanged baseline reproduced the prior
8.0664975654% grounded goal projection. This is baseline reproduction, not a
new record or proof of visual improvement. Six owned emulators and fresh status
were verified, with no errors. Collection's absolute cutoff is October27 at
09:56:22 BRT, and final review is scheduled for10:02 BRT.
See [the dispatch/provenance evidence](assets/performance/genetic-visual-authorization-20261027.json).
