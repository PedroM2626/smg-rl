# Interpreting the final Sweet Sweet evaluation metrics

The October 28 reward-hour result reported a 52.15% stochastic mean for the
human-route candidate and 49.93% for the opening control. These are not
fractions of the mission, the human route or the star objective. They are
the means of three maximum projections onto **only the first segment** of the
human waypoint polyline. The first segment runs from approximately
`(-50, 45, -50)` to `(-19.23, -50, -1022.84)` and spans roughly 973 horizontal
world units. A sample can qualify when within 750 horizontal units of the
segment and 500 vertical units of the interpolated segment. Those tolerances
are broad relative to the segment and the projected maximum can be achieved
on a nearby platform. The score is a high-water mark, not the final position;
it can remain high after Mario returns or dies.

Direct video/telemetry checks illustrate the problem:

| Final frozen evaluation | Episode and decision | Sample position (X, Y, Z) | First-segment score | Video inspection |
| --- | --- | --- | ---: | --- |
| Opening control, noisy action seed 23 | Episode 2, decision 15 | `(453.12, 45.00, -532.96)` | 51.23% | Mario is at the starting platform edge; the sample is approximately 488 horizontal units sideways from the projected segment. |
| Human-route candidate, noisy action seed 11 | Episode 1, decision 75 | `(20.03, 56.68, -640.34)` | 60.85% | Mario is close to the start, on the first blue belt, and subsequently dies. |

The candidate's previously reported 52.15% is the mean of 60.85%, 39.25% and
56.35% across action seeds 11, 23 and 47; its 60.85% is one of those episodes,
not a full-course percentage. For the control the mean is 49.93%, across
46.04%, 51.23% and 52.53%. Both videos are concatenations of three separate
full-start episodes. A single frame is only evidence of that episode and time,
but these examples corroborate that the reported numbers can look large even
when the agent has barely left its starting area. The candidate's strict
ordered full-route score is zero in each frozen final evaluation, and no
agent star was observed.

The first-segment score is **not the training reward**: it is an offline
diagnostic for comparing videos. The active training reward reads a different,
separately signed route object and completion from RAM. See
[the bounded comparison](reward-hour-pair-20261028.md) and its
[episode evidence](assets/performance/reward-hour-pair-final-20261028.json).
The naming and reporting of this partial metric can mislead readers; until a
verified full-route progress measurement is available, report it only with its
precise scope, corridor and limitations, next to native RAM-confirmed stars.
