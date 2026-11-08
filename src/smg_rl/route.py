"""Ordered, contact-gated route-prefix evaluation, separate from mission progress."""

import numpy as np


class RouteProgress:
    def __init__(self, specification):
        self.specification = specification
        self.schema = specification.get("schema", 1)
        if self.schema not in (1, 2, 3, 4, 5, 6):
            raise ValueError("Unsupported route schema.")
        self.points = np.asarray(specification["waypoints"], dtype=float)
        if (self.points.ndim != 2 or self.points.shape[1] != 3 or len(self.points) < 2
                or not np.isfinite(self.points).all()):
            raise ValueError("A route needs finite XYZ waypoints.")
        self.lengths = np.linalg.norm(np.diff(self.points[:, [0, 2]], axis=0), axis=1)
        if np.any(self.lengths < 1):
            raise ValueError("Route segments must have distinct horizontal endpoints.")
        self.total_length = self.lengths.sum()
        if self.schema == 6:
            # Schema 6 pays by projecting the position onto the 3D polyline inside a corridor (airborne
            # progress counts); its waypoints only mark where credit units fall along the arc length.
            self.seg3 = np.diff(self.points, axis=0)
            self.len3 = np.linalg.norm(self.seg3, axis=1)
            self.cum3 = np.concatenate([[0.0], np.cumsum(self.len3)])
            self.total3 = float(self.cum3[-1])
            self.lookahead = float(specification.get('lookahead', 600.0))
            self.lookbehind = float(specification.get('lookbehind', 300.0))
            if (not np.isfinite([self.lookahead, self.lookbehind]).all() or self.lookahead <= 0
                    or self.lookbehind < 0 or np.any(self.len3 <= 0)):
                raise ValueError('Projection routes need positive lookahead and distinct 3D waypoints.')
        if (min(specification[k] for k in ("waypoint_radius", "corridor_radius", "height_tolerance")) <= 0
                or not 1 <= specification["exit_confirmations"] <= 30
                or specification.get("requires_grounded") is not True):
            raise ValueError("Routes require positive tolerances and grounded confirmation.")
        if self.schema in (4, 5, 6) and (not np.isfinite(specification.get('max_sample_distance', 1000.0))
                                 or specification.get('max_sample_distance', 1000.0) <= 0):
            raise ValueError('Human-route sample distance must be finite and positive.')
        if self.schema == 5 and (not np.isfinite(specification.get('landing_radius', 0))
                                 or specification.get('landing_radius', 0) <= 0):
            raise ValueError('Landing radius must be positive and finite.')
        lookahead = specification.get('waypoint_lookahead', 0)
        if type(lookahead) is not int or not 0 <= lookahead <= 6 or (lookahead and self.schema != 5):
            raise ValueError('Waypoint lookahead is a small integer for schema-5 routes only.')
        self.reset(None)

    def reset(self, telemetry, segment=None):
        if self.schema == 6:
            return self._reset_projection(telemetry)
        self.segment, self.best, self.confirmations = 0, 0.0, 0
        self.exited = False
        self.initial_health = telemetry.health if telemetry else None
        self.previous_position = np.asarray(telemetry.position, dtype=float) if telemetry else None
        self.pending_segment = 0
        self.committed_segment = 0
        self.eligible = telemetry is not None and np.linalg.norm(
            np.asarray(telemetry.position) - self.points[0]) <= self.specification["waypoint_radius"]
        if self.schema == 3 and segment is not None:
            if not isinstance(segment, int) or not 0 <= segment < len(self.lengths):
                raise ValueError('Invalid curriculum route region.')
            self.segment = segment
            point = np.asarray(telemetry.position, dtype=float)
            start, end = self.points[segment:segment + 2]
            axis = (end - start)[[0, 2]]
            t = float(np.clip((point - start)[[0, 2]] @ axis / (axis @ axis), 0, 1))
            nearest = start + t * (end - start)
            self.eligible = bool(telemetry.grounded is True and np.isfinite(point).all()
                and np.linalg.norm((point - nearest)[[0, 2]]) <= self.specification['corridor_radius']
                and abs(point[1] - nearest[1]) <= self.specification['height_tolerance'])
            if not self.eligible:
                raise ValueError('Curriculum position is outside its verified route region.')
            self.best = float((self.lengths[:segment].sum() + t * self.lengths[segment]) / self.total_length)
        elif self.schema in (4, 5) and segment is not None:
            raise ValueError('Human-derived routes do not support assisted starts.')

    def update(self, telemetry):
        if self.schema == 6:
            return self._update_projection(telemetry)
        if self.schema in (4, 5):
            return self._update_human_route(telemetry)
        if not self.eligible or telemetry.dead or telemetry.complete:
            self.confirmations = 0
            return 0.0
        point = np.asarray(telemetry.position, dtype=float)
        if (not np.isfinite(point).all() or telemetry.grounded is not True
                or (self.specification.get('requires_full_health', True) and self.initial_health is not None
                    and (telemetry.health is None or telemetry.health < self.initial_health))):
            self.confirmations = 0
            return 0.0
        if self.schema in (2, 3):
            return self._update_endpoint_crossing(point)
        start, end = self.points[self.segment:self.segment + 2]
        axis = (end - start)[[0, 2]]
        t = float(np.clip((point - start)[[0, 2]] @ axis / (axis @ axis), 0, 1))
        nearest = start + t * (end - start)
        if (np.linalg.norm((point - nearest)[[0, 2]]) > self.specification["corridor_radius"]
                or abs(point[1] - nearest[1]) > self.specification["height_tolerance"]):
            self.confirmations = 0
            return 0.0
        value = (self.lengths[:self.segment].sum() + t * self.lengths[self.segment]) / self.total_length
        delta = max(0.0, value - self.best)
        self.best = max(self.best, value)
        reached = np.linalg.norm(point - end) <= self.specification["waypoint_radius"]
        if reached and self.segment < len(self.lengths) - 1:
            self.segment += 1
        elif reached:
            self.confirmations += 1
            self.exited |= self.confirmations >= self.specification["exit_confirmations"]
        else:
            self.confirmations = 0
        return delta

    def _project(self, point, lo, hi):
        """(arc length, 3D distance) of the closest point on segments lo..hi-1."""
        a, d, n = self.points[lo:hi], self.seg3[lo:hi], self.len3[lo:hi]
        t = np.clip(np.einsum('ij,ij->i', point - a, d) / n ** 2, 0.0, 1.0)
        distance = np.linalg.norm(a + t[:, None] * d - point, axis=1)
        i = int(np.argmin(distance))
        return float(self.cum3[lo + i] + t[i] * n[i]), float(distance[i])

    def _reset_projection(self, telemetry):
        self.segment = self.pending_segment = self.committed_segment = 0
        self.best, self.best_s, self.confirmations, self.exited = 0.0, 0.0, 0, False
        self.initial_health = telemetry.health if telemetry else None
        self.previous_position = np.asarray(telemetry.position, dtype=float) if telemetry else None
        self.eligible = False
        if telemetry is not None and np.isfinite(telemetry.position).all():
            arc, distance = self._project(np.asarray(telemetry.position, dtype=float), 0, len(self.seg3))
            if distance <= self.specification['corridor_radius']:
                self.eligible, self.best_s = True, arc
                self.best = arc / self.total3
                self.committed_segment = self.pending_segment = int(np.searchsorted(self.cum3, arc, side='right')) - 1

    def _update_projection(self, telemetry):
        point = np.asarray(telemetry.position, dtype=float)
        if not self.eligible or telemetry.dead or telemetry.complete or not np.isfinite(point).all():
            return 0.0
        previous, self.previous_position = self.previous_position, point.copy()
        if previous is not None and np.linalg.norm(point - previous) > self.specification.get('max_sample_distance', 1000.0):
            self.eligible = False  # A teleport or respawn must never create route credit.
            return 0.0
        lo = max(int(np.searchsorted(self.cum3, self.best_s - self.lookbehind, side='right')) - 1, 0)
        hi = min(max(int(np.searchsorted(self.cum3, self.best_s + self.lookahead, side='left')), lo + 1), len(self.seg3))
        arc, distance = self._project(point, lo, hi)
        if distance > self.specification['corridor_radius'] or arc > self.best_s + self.lookahead or arc <= self.best_s:
            return 0.0
        credit = (arc - self.best_s) / self.total3
        self.best_s, self.best = arc, arc / self.total3
        self.committed_segment = self.pending_segment = int(np.searchsorted(self.cum3, arc, side='right')) - 1
        self.segment = min(self.committed_segment, len(self.seg3) - 1)
        self.exited = self.committed_segment >= len(self.seg3)
        return credit

    def reset_human_anchor(self, telemetry, waypoint):
        """Initialize a verified demonstration anchor without paying its prefix."""
        if self.schema != 5 or type(waypoint) is not int or not 0 <= waypoint < len(self.lengths):
            raise ValueError('A human curriculum anchor needs a valid schema5 waypoint.')
        point = np.asarray(telemetry.position, dtype=float)
        if (not np.isfinite(point).all() or telemetry.dead or telemetry.complete
                or telemetry.grounded is not True
                or np.linalg.norm(point - self.points[waypoint]) > .01):
            raise ValueError('Human anchor must exactly restore its living grounded waypoint.')
        self.reset(telemetry)
        self.eligible = True
        self.segment = self.pending_segment = self.committed_segment = waypoint
        self.best = float(self.lengths[:waypoint].sum() / self.total_length)

    def _update_human_route(self, telemetry):
        """Track ordered swept waypoint visits and pay only after live contact."""
        point = np.asarray(telemetry.position, dtype=float)
        if not self.eligible or not np.isfinite(point).all() or telemetry.dead or telemetry.complete:
            return 0.0
        previous = self.previous_position
        self.previous_position = point.copy()
        if previous is None:
            return 0.0
        delta = point - previous
        horizontal = delta[[0, 2]]
        length2 = float(horizontal @ horizontal)
        max_step = self.specification.get('max_sample_distance', 1000.0)
        if np.linalg.norm(delta) > max_step:
            # Discontinuous observations must never manufacture route credit.
            self.eligible = False
            return 0.0
        if length2 > 1e-6:
            # A point reached in the air can be credited only after a later landing.
            # Every waypoint must intersect the actually observed motion segment.
            # An optional lookahead lets a swept later waypoint skip up to that many missed ones.
            last_t, lookahead = 0.0, self.specification.get('waypoint_lookahead', 0)
            while self.pending_segment < len(self.lengths):
                for target in range(self.pending_segment + 1,
                                    min(self.pending_segment + 2 + lookahead, len(self.points))):
                    t = self._sweep(previous, delta, horizontal, length2, self.points[target])
                    if t is not None and t >= last_t:
                        break
                else:
                    break
                self.pending_segment = target
                last_t = t
        if telemetry.grounded is not True:
            return 0.0
        if self.schema == 5 and self.pending_segment > self.committed_segment:
            end = self.points[self.pending_segment]
            radius = min(self.specification['landing_radius'], .45 * self.lengths[self.pending_segment - 1])
            if (np.linalg.norm((point - end)[[0, 2]]) > radius
                    or abs(point[1] - end[1]) > self.specification['height_tolerance']):
                self.pending_segment = self.committed_segment
            else:
                self.committed_segment = self.pending_segment
        self.segment = min(self.pending_segment, len(self.lengths) - 1)
        value = float(self.lengths[:self.pending_segment].sum() / self.total_length)
        credit = max(0.0, value - self.best)
        self.best = max(self.best, value)
        self.exited |= self.pending_segment == len(self.lengths)
        return credit

    def _sweep(self, previous, delta, horizontal, length2, end):
        """Return where the observed motion segment passes a waypoint, or None if it misses."""
        approach = float((end - previous)[[0, 2]] @ horizontal)
        if approach <= 0:
            return None
        t = float(np.clip(approach / length2, 0, 1))
        nearest = previous + t * delta
        if (np.linalg.norm((end - nearest)[[0, 2]]) > self.specification['waypoint_radius']
                or abs(end[1] - nearest[1]) > self.specification['height_tolerance']):
            return None
        return t

    def _update_endpoint_crossing(self, point):
        """Resolve crossings now; proximity must not manufacture later progress.

        Schema 1 is retained for historical reward/checkpoint reproducibility.
        Schema 2 remains an isolated candidate. Schema 3 uses these crossings
        with explicitly anchored curriculum resets and separate potential shaping.
        """
        segment = self.segment
        while True:
            start, end = self.points[segment:segment + 2]
            axis = (end - start)[[0, 2]]
            raw = float((point - start)[[0, 2]] @ axis / (axis @ axis))
            t = float(np.clip(raw, 0, 1))
            nearest = start + t * (end - start)
            reached = raw >= 1 and np.linalg.norm(point - end) <= self.specification["waypoint_radius"]
            if (np.linalg.norm((point - nearest)[[0, 2]]) > self.specification["corridor_radius"]
                    or abs(point[1] - nearest[1]) > self.specification["height_tolerance"]
                    or (raw >= 1 and not reached)):
                self.confirmations = 0
                return 0.0
            if reached and segment < len(self.lengths) - 1:
                segment += 1
                continue
            break
        self.segment = segment
        value = float((self.lengths[:segment].sum() + t * self.lengths[segment]) / self.total_length)
        delta = max(0.0, value - self.best)
        self.best = max(self.best, value)
        self.confirmations = self.confirmations + 1 if reached else 0
        self.exited |= self.confirmations >= self.specification["exit_confirmations"]
        return delta
