# Frozen-vision genetic search with goal-distance fitness

On October 27, 2026, the owner selected a route-free goal-proximity objective and
a genetic algorithm with frozen visual features. This is a separate experiment;
the earlier genetic populations, source checkpoints and fitness definitions are
preserved. Earlier trials do not establish that genetic search generally
outperforms IQN or that frozen vision is always better.

## Objective and boundaries

The new `goal-distance-xyz-v1` fitness is the largest reduction, in game world
units, of **three-dimensional Euclidean distance** from the canonical reset
position to the calibrated near-star goal. Only finite, continuous, alive and
grounded samples are eligible. The high-water tracker credits a new closest
point once; an airborne, dead or completion-animation sample does not add
fitness. Native RAM completion is recorded independently. There is no time,
damage, death, novelty or other fitness term. The environment can still record
health and terminal outcomes, but selection ignores environment return.

There is no mapped route or route-based idle termination in the distance run.
A finite 1,200-decision episode limit remains. Prior X/Z projection is logged
separately as a diagnostic and is not the distance fitness. Coordinate distance
can reward an unreachable shortcut or fail to reward a necessary detour, and
its percentage is not a native completion percentage. A moving platform or a
faulty grounded signal can contaminate a sample. A confirmed full-start RAM
star is the success criterion.

## Policy transfer and isolation

The frozen IMPALA/IQN feature source is the original checkpoint used by the
earlier head-only genetic experiment. Its 4,104-gene champion is imported only
after checking the earlier run's SHA-256 and exact source network tensors.
The CNN and remaining feature/quantile layers are frozen; evolution changes
only the action head. The imported genome becomes individual 0 and the rest of
the 24-member initial population are mutations. **No old fitness or trials are
reused**: every candidate is scored again under the new objective. Historical
`--resume-run` populations are rejected for the new mode to prevent mixing
incompatible scores.

`config/genetic-goal-distance-smg1.yaml` retains the source policy signature
for loading. Only after validating that signature does the genetic runtime turn
off curriculum and the regional route, use isolated output/profile paths and
apply the episode budget. This does not modify the source checkpoint or train
IQN. Six Dolphin collectors share one parent policy; no gradient updates,
curiosity, world model or imagined transitions are used. Disk and memory
headroom are checked; a fault preserves partial evidence and does not cause a
blind retry.

## Initial bounded pilot

The first pilot has a 300-second collection deadline including initialization;
shutdown and checkpoint writing can finish afterward. At most 45 seconds are
reserved for frozen confirmation attempts.
There is no automatic successor. Seed labels can repeat the same restored Wii
state; they are not independent game environments. Partial episodes cannot
establish fitness or a star. This is a health and exploratory test, not a
demonstrated success or an algorithm ranking.

```powershell
.venv\Scripts\python.exe scripts/train_genetic.py --config config/genetic-goal-distance-smg1.yaml --source runs/model-free/smg1-six-two-hour/dqn-20261026-204307-660910/checkpoint-final --initial-head runs/model-free/genetic-smg1/genetic-20261026-231504-090947/champion.pt --fitness-mode goal-distance-xyz-v1 --policy-mode head --seconds 300 --confirmation-seconds 45
```

The run directory is printed at dispatch, with its controller PID/creation
time, `status.json`, `trials.jsonl`, generation archives, champion and final
summary. Inspect actual process identity, source/champion hashes, completed and
partial trials, distance and projection as separate values, deaths/HP, and RAM
stars. Do not launch duplicate jobs or extend the budget without review.

## Completed pilot and audit

The pilot ended cleanly after 294.45 seconds, with no automatic successor and no
remaining Dolphin instances. It recorded 6,939 real decisions: 5,863 during
evolution and 1,076 in frozen confirmation. The ledger contains 35 completed
evolution attempts, four completed confirmation attempts, and six partial
evolution attempts; one full generation completed. The partial second
generation was not used to reproduce. There were 36 deaths, six points of
nonfatal HP loss and **zero RAM-confirmed full-start stars**.

The best observed candidate was individual 0 in generation 0: the unchanged
imported champion. Its new XYZ distance reduction was 2,075.671 world units,
or 7.0869% of the initial XYZ distance; it died after 269 decisions. Its X/Z
projection was 8.0665%, a different geometric quantity. Neither is a native
mission percentage. No newly mutated candidate exceeded this champion within
the completed trial. Four frozen confirmations (the unchanged initial champion
and the selected champion, each with labels 23 and 47) reproduced the same
distance and death. Those labels restore the same game state and are not
independent generalization tests.

The [result audit](assets/performance/genetic-goal-distance-pilot-20261027.json)
recomputed all 45 recorded trajectories, matched real-decision and
complete/partial counts, checked full-start terminal evidence and source and
champion hashes, and verified every frozen source-network tensor. No previous
fitness was reused and no gradients were applied. The [champion trajectory](assets/performance/genetic-goal-distance-pilot-20261027.png)
shows measured X/Z movement and the XYZ distance high-water curve.

This short pilot validates the implementation in actual gameplay; it does not
show improved navigation from the new objective. The pilot's retained full
generation and candidate traces are useful for a reviewed longer comparison,
but the implementation deliberately disallows resuming a population under the
new metric until a metric-specific resume and artifact audit are implemented.
Starting a new population from a verified head is different from exact population
and RNG continuation, and must not be reported as a resume.

## Owner-requested three-hour run

The owner subsequently requested six workers for three hours. A new population
was initialized from the verified pilot champion with population seed 143;
old fitness and partial-generation records were not reused. The run used the
same `goal-distance-xyz-v1` score, frozen features, 24-member population and
canonical game start. The absolute deadline was October 27, 2026, 22:26:16
America/Sao_Paulo. No frozen-evaluation reserve or automatic successor was
allocated. The retained local dispatch identity is
`.local/genetic-goal-three-hours-20261027.json`.

The run stopped at its deadline, with no reported error or remaining Dolphin
instances. It lasted 10,801.3 seconds including shutdown and produced 291,513
real decisions, 134 completed generations, 3,239 completed attempts and one
partial attempt. There were 3,158 deaths, 81 timeouts, 219 nonfatal HP lost and
**zero RAM-confirmed full-start stars**. No gradients, imagined transitions or
additional confirmation evaluations were used.

The initial champion was physically reevaluated in generation 0: its best XYZ
distance reduction was 2,075.671 world units, or 7.0869% of the initial XYZ
distance, followed by death. It remained the best candidate. The strongest
different genome reached 2,015.774 units, or 6.8824%, and also died. Thus the
new population did not improve over its initial champion in this run. These
values are geometric distance reduction, not native mission percentages, and
are collection results rather than a new frozen-policy evaluation. The fixed
Wii start is not an independent environment for each seed label.

The [three-hour audit](assets/performance/genetic-goal-distance-three-hours-20261027.json)
recomputed all 3,240 trajectories and checked source/champion digests, exact
frozen source-network tensors, decision/attempt/terminal counts and native
completion evidence. The retained [top-down path](assets/performance/genetic-goal-distance-three-hours-20261027.png)
and XYZ distance curve represent the unchanged initial champion. Actions were
41 none, 25,350 right, 368 left, 85,853 forward, 28 backward, 145,694 jump,
4,995 spin and 29,184 crouch. The original checkpoints, local trajectories,
partial attempt and run logs remain retained.
