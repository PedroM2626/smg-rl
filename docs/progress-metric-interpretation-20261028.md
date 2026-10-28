# Withdrawn Sweet Sweet progress metric

The reported 52.15% and 49.93% first-section figures are **withdrawn as
invalid progress measurements**. They did not establish that Mario reached the
corresponding locations. The old calculation projected a grounded position onto
a line between two human-route points, with a 750-unit lateral tolerance and a
500-unit height tolerance. A position far beside the route could receive a large
score. The flaw is not explained merely by saving a maximum over the episode.

A direct inspection of the control evaluation with stochastic action seed 23
found Mario at the starting platform edge at `(453.12, 45.00, -532.96)` while
the calculation reported 51.23% of the first segment. In the candidate
evaluation with action seed 11, a sample near the start on the first blue belt
was assigned 60.85%. Neither number demonstrates a visit to the corresponding
point on the human route. The original reported aggregate scores and other
first-segment results must not be used for ranking rewards or describing
visited course progress. The original evidence remains unchanged for audit.

The retired calculation is disabled by default. Explicit opt-in exists only to
reproduce historical artifacts; it does not make the score valid. A new
conservative audit checks grounded, living XYZ samples against eight ordered
landmarks selected from actual frames of the owner’s completed, replay-verified
Sweet Sweet demonstration. It never interpolates an agent visit, enforces the
canonical start and rejects nonfinite or discontinuous positions. A match is a
**candidate for visual review**, not proof of the exact surface reached; the
tolerances may miss other valid routes. Landmark counts are not a mission
percentage. This instrument is for evaluation and does not change training
rewards, signed profiles, existing checkpoints or historical replay.

The human source trajectory matched all eight landmarks on six sampling phases
from the same recording; this is a consistency check, not independent
generalization. Reanalysis of all 16 frozen evaluations in the one-hour
comparison found zero candidate landmark matches and zero full-start native RAM
stars. This is not proof of zero movement, zero mission percentage, or failure
of every possible alternate path. Completion requires the native RAM star.

See [landmark audit evidence](assets/performance/observed-landmark-audit-v2-20261028.json),
[withdrawal manifest](assets/performance/withdrawn-progress-metric-20261028.json),
and [the comparison](reward-hour-pair-20261028.md). The old first-segment
projection was an offline diagnostic, not the training reward. The latter uses
different signed route objects and still requires its own exploit and
generalization validation. No training was started as part of this correction.
