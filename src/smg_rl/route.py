"""Ordered, contact-gated route-prefix evaluation, separate from mission progress."""

import numpy as np


class RouteProgress:
    def __init__(self, specification):
        self.specification = specification
        self.points = np.asarray(specification["waypoints"], dtype=float)
        if (self.points.ndim != 2 or self.points.shape[1] != 3 or len(self.points) < 2
                or not np.isfinite(self.points).all()):
            raise ValueError("A route needs finite XYZ waypoints.")
        self.lengths = np.linalg.norm(np.diff(self.points[:, [0, 2]], axis=0), axis=1)
        if np.any(self.lengths < 1):
            raise ValueError("Route segments must have distinct horizontal endpoints.")
        self.total_length = self.lengths.sum()
        if (min(specification[k] for k in ("waypoint_radius", "corridor_radius", "height_tolerance")) <= 0
                or not 1 <= specification["exit_confirmations"] <= 30
                or specification.get("requires_grounded") is not True):
            raise ValueError("Routes require positive tolerances and grounded confirmation.")
        self.reset(None)

    def reset(self, telemetry):
        self.segment, self.best, self.confirmations = 0, 0.0, 0
        self.exited = False
        self.initial_health = telemetry.health if telemetry else None
        self.eligible = telemetry is not None and np.linalg.norm(
            np.asarray(telemetry.position) - self.points[0]) <= self.specification["waypoint_radius"]

    def update(self, telemetry):
        if not self.eligible or telemetry.dead or telemetry.complete:
            self.confirmations = 0
            return 0.0
        point = np.asarray(telemetry.position, dtype=float)
        if (not np.isfinite(point).all() or telemetry.grounded is not True
                or (self.initial_health is not None and (telemetry.health is None or telemetry.health < self.initial_health))):
            self.confirmations = 0
            return 0.0
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
