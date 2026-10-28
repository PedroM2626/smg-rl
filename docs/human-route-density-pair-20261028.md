# Matched human-route reward-density comparison

On October 28 the owner stopped the previous overnight continuation to test
whether preserving the old opening reward density over the full human route
helps. The stopped checkpoint passed owner-stop audit: 172,158 cumulative real
decisions, 1,258 completed attempts, no CSV gap, and zero native full-start
stars in its accumulated lineage. This includes the previous source; the
stopped continuation added 72,246 real decisions and 545 attempts. Its
exploratory collection is not a frozen-policy result.

Both comparison arms start from exactly that checkpoint's online weights and
seed 142, with separate fresh replay, optimizer, counters, RNG and synchronized
target networks. The same human route, full-start state, 12-frame controls,
IQN/IMPALA settings, +20 RAM star, -2 death/timeout and -0.25 HP loss apply.
Only reward density and its explicit signature differ: the control pays at most
+1 along 37,872.533 route units; the candidate pays up to +17.32391036, i.e.
about +1 per 2,186.1423 route units. Since credit follows ordered waypoint
confirmation it is sparse and not guaranteed at every unit. The increased
reward can result in a positive return for an episode that dies after partial
progress: this is a deliberate experimental objective, not the prior
negative-partial-failure quality guarantee. Inspect this incentive and
landmark/video evidence, not reward alone. Old profiles retain their original
validation gate.

The earlier assertion that `v_max=25` clips the IQN target was incorrect: this
implementation uses fixed support and categorical projection only for Rainbow
C51; IQN predicts unconstrained quantiles. Its unchanged `v_max` setting does
not truncate IQN predictions or targets. The candidate uses an explicitly
separate distance-advance reward mode; older reward signatures are unchanged.

`scripts/run_human_route_density_pair.py` is the sole supervisor. It alternates
at-most-one-hour arm batches made of bounded, audited segments, a 300-second
health segment per arm first, at most six emulators, 17,100 accounted seconds per
arm and an absolute collection deadline at 07:00 America/Sao_Paulo on October
29. Segmented resets are not bitwise continuous training. RAM and disk guards,
source and config hashes, replay/attempt reconciliation and native-star checks
are required before successors. Failed/partial work is retained and never
retried blindly. State and per-job commands are under
`.local/human-route-density-pair-20261028/`.

At 07:00 audit elapsed times, real decisions, attempts, reward integrity and
star flags. A single greedy and three stochastic full-start frozen evaluation
episodes per arm, with separate counts and video, are authorized after clean
completion. Landmark proximity is a candidate for visual inspection, not a
verified visit or mission percentage. Action seeds restore the same Wii state;
one training seed cannot establish statistical generalization. One arm may
receive less time if the absolute deadline or a failure interrupts the study.
Do not compare maxima from collection as frozen results and do not report the
withdrawn first-segment projection as progress.
