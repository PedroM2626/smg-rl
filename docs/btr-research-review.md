# BTR appendix and public-code review

> **Historical context:** Experiment status, authorizations, schedules and live counters in this document refer to their stated dates. Later work may supersede them; consult [current evidence and next experiments](current-status.md) and check actual process state before acting.

Date: October 25, 2026. This analysis accompanies the
[SMG1 prediction-block repair](smg1-boundary-repair.md). It does not silently
replace the active RSSM learner with Rainbow/BTR or change the owner's eight
action labels. External code was read, not installed, executed or copied.

## Sources and what is actually published

The paper is Clark, Towers, Evers and Hare, *Beyond The Rainbow: High Performance
Deep Reinforcement Learning on a Desktop PC*, ICML 2025, PMLR 267. The
[publication record](https://proceedings.mlr.press/v267/clark25a.html) and
[arXiv v2, Appendix J.1, page 27](https://arxiv.org/pdf/2411.03820v2#page=27)
were checked. The repository's description says 2024, reflecting its earlier
history rather than the final conference year.

Appendix J concerns Dolphin interfaces, RAM reward/termination, grayscale
140x114 observations and frameskip 4. J.1 studies reaching the final fight in
The Center of the Universe, not Sweet Sweet Galaxy or either SMG2 mission.
It describes six action labels, continuing movement during a following jump,
memory-derived progress adjusted by region, +/-1 for its three-unit life signal,
and starts at several points. It supplies no SMG RAM address, exact variable
definition, region map or portable native mission-percentage specification.

Public code inspected at these pinned main-branch revisions:

| Repository | Commit | Relevant finding |
| --- | --- | --- |
| [VIPTankz/BTR](https://github.com/VIPTankz/BTR/tree/093ca77f08540d646f12fd942ee5fda763e0fd8a) | `093ca77f08540d646f12fd942ee5fda763e0fd8a` | Public training entry point constructs ALE Atari environments. The inspected code does not provide the SMG environment/addresses. |
| [VIPTankz/Wii-RL](https://github.com/VIPTankz/Wii-RL/tree/d8358cbc5feef41161522e51b60fba100506d489) | `d8358cbc5feef41161522e51b60fba100506d489` | Separate Wii interface, with a Mario Kart example; its published Dolphin script contains Kart telemetry. No SMG address was found in the inspected main-branch environment. |

In particular, [BTR main.py](https://github.com/VIPTankz/BTR/blob/093ca77f08540d646f12fd942ee5fda763e0fd8a/main.py)
constructs `ALE/<game>-v5` and defaults to 64 environments. Those are not 64
Dolphin instances. The [Wii DolphinScript.py](https://github.com/VIPTankz/Wii-RL/blob/d8358cbc5feef41161522e51b60fba100506d489/DolphinScript.py)
uses MarioKart save-state paths and speed/lap/race-completion fields. Its Kart
addresses cannot be copied to Galaxy. This review covers the current inspected
code, not a proof that no historical branch or unpublished implementation exists.

## Corrections to the proposed interpretation

**RAM does not imply native percentage.** The appendix's region-dependent
description is compatible with choosing local coordinates or other fields and
stitching them into a task-specific scalar. It does not establish one universal
0-100 mission variable. That stitching interpretation is an inference; the
missing environment code prevents determining its exact implementation. There
is also no evidence here that the authors wrote a progress variable into RAM.
Our implementation should continue reading game state rather than patching it.

**Three units do not identify the extra-life counter.** Nintendo's
[official instruction booklet, printed page7](https://m1.nintendo.net/docvc/RVL/USA/RMGE/RMGE_E.pdf#page=5)
distinguishes the life meter, depleted by damage, from remaining lives, lost
when Mario dies. The appendix's three-unit wording appears consistent with HP,
but the precise mapping remains uncertain without its environment code. Do not
equate it automatically with 1-Up lives or duplicate HP and death penalties.
This project already validates HP 3 to 2 and charges nonfatal damage separately.

**Six labels do not establish six mutually exclusive physical inputs.** The
paper says movement continues on a following jump, without specifying whether
the implementation holds direction or relies on physics. Our exclusive motor8
fixture already verifies jump inertia after walking. An explicit direction-carry
jump could be tested as a separately signed motor profile with the same eight
logical labels. It must not silently change existing replay semantics.

**Starts and environment definitions matter.** The relevant lesson is structured
reward and coverage of later regions, not evidence that another learner will
solve our different missions merely by running longer. Assisted-start success
and full-start star collection must remain separate. The paper does not provide
a controlled ablation isolating the effect of each Wii environment choice.

## Bounded local RAM screening

To move beyond speculation, `scripts/probe_progress_ram.py` performed a 33-decision
scripted SMG1 initial-prefix probe. It read aligned float32 and uint32 values
across 24 MiB of MEM1 in ten paused snapshots: initial, stationary, reset,
right/forward milestones, backward movement and final reset. It made no memory
writes, used no near-star state and added no training replay.

Filtering rejected nonfinite/unbounded values, idle-changing timers, reset
drift, constants, nonmonotone movement traces and exact XYZ aliases. Nevertheless,
3724 float interpretations and 328 integer interpretations survived. These are
ambiguous correlations, not validated progress addresses; several encodings may
refer to related state. **Zero native progress addresses were accepted or enabled.**
The [screening evidence](assets/performance/smg1-progress-ram-probe-20261025.json)
retains samples and a bounded candidate list. Raw snapshots remain local only.
The helper safely rejects nonfinite bit patterns without arithmetic warnings.

This limited test cannot certify anything about MEM2, unaligned/other encodings,
later segments, pointer stability across boots, SMG2, or a game's lack of a
progress field. Further acceptance requires negative controls for stationary
animation/jump/spin/crouch, reverse/revisit loops, falls, HP/death transitions,
multiple region boundaries, pointer resolution and the actual end state. Mere
correlation along one short route is insufficient. A field from the paper's
final galaxy would also need independent validation for our selected stages.

## Recommended experimental order

1. **Repair reward topology and scale.** Validate an ordered regional route over
   more than the current short prefix. Use regions to select local progress,
   with continuous transitions and no stationary waypoint credit. Audit rewards
   for falling, revisiting and damage. Test discount-consistent potential shaping
   with explicit timeout semantics; RAM-confirmed stars remain the task goal.
2. **Add safe intermediate starts.** Reuse the existing frontier infrastructure
   only after validating new intermediate anchors. The owner's original near-star
   states remain detector fixtures. Record assisted starts and their learning
   contribution separately; final evaluation always includes canonical starts.
   Reset shaping baselines at each chosen start to avoid free reward on loading.
3. **Test temporal control as its own ablation.** Frameskip4 gives three times as
   many decisions per emulated second as our12-frame profile. It may improve
   platform timing even if decisions/second decrease. Compare game frames,
   wall time and full-start progress, not step labels alone. The current
   jump/spin8-on/4-off execution is explicitly12 frames; a four-frame variant
   needs new pulse semantics and a new action signature, not a YAML-only change.
4. **Test visual resolution separately.** 140x114 has about 2.26 times as many
   pixels as 84x84. It might preserve small character/platform cues, but increases
   model/data cost and requires non-square observation/decoder support plus
   compatible replay. Combining this with three times the decision frequency
   substantially raises observation throughput. It is not automatically faster.
5. **Then compare learners and parallelism.** BTR is a legitimate separate
   model-free baseline after matching reward, starts, controls and frame budgets.
   Its published Atari wall times are not Dolphin/Mario wall-time predictions.
   Do not copy Atari reward clipping or old checkpoints into this task unchanged.

These are proposed separate comparisons, not simultaneously enabled changes.
The most direct local need is validated progression plus later-state coverage;
prediction-gate repair alone does not address the known positive partial-death
returns or the sparse experience beyond the opening platform.

## Six instances versus six independent agents

The measured local hardware is an i9-14900HX with 24 physical / 32 logical cores,
about 31.71 GiB RAM and an RTX 4070 Laptop with 8 GiB VRAM. Six Dolphin collectors
are plausible, but sustained throughput/responsiveness has not been benchmarked
for the repaired learner. More environments may saturate CPU/memory or increase
learner lag rather than improve wall time.

Two collectors per mission can share fewer policies and aggregate experience;
that is different from six independent agents with separate seeds, replay,
models and optimizers. Independent replicas help measure variability, but must
not be mistaken for one agent learning six times faster. Benchmark 2/4/6 collectors
under finite budgets after reward validation, then select concurrency by measured
real frames/second, update-to-data ratio and navigation gains. No six-instance
or indefinite run was launched for this research review.
