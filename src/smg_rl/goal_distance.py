"""Grounded geometric goal-distance tracking for explicit experiments."""

import math

import numpy as np


class GroundedGoalDistance:
    """Track closest valid grounded X/Z or XYZ distance from an episode reset.

    The update is intentionally independent of a recorded route. It is an
    experimental geometric proxy, not native mission progress. The genetic
    distance experiment selects XYZ explicitly; older offline audits use X/Z.
    """

    def __init__(self, start, goal, initial_position, max_step, dimensions=(0, 2)):
        start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.previous = np.asarray(initial_position, dtype=float)
        if (start.shape != (3,) or self.goal.shape != (3,) or self.previous.shape != (3,)
                or not np.isfinite(np.concatenate((start, self.goal, self.previous))).all()
                or not math.isfinite(max_step) or max_step <= 0):
            raise ValueError('Goal-distance tracking requires finite XYZ and a positive step bound.')
        if dimensions not in ((0, 2), (0, 1, 2)):
            raise ValueError('Choose X/Z or full XYZ distance explicitly.')
        self.dimensions = list(dimensions)
        self.normalization = float(np.linalg.norm((start - self.goal)[self.dimensions]))
        if self.normalization < 1:
            raise ValueError('Goal-distance normalization requires distinct selected coordinates.')
        self.max_step = float(max_step)
        self.best_distance = float(np.linalg.norm((self.previous - self.goal)[self.dimensions]))
        self.invalid = False

    def update(self, position, grounded, dead=False, complete=False):
        point = np.asarray(position, dtype=float)
        if (self.invalid or point.shape != (3,) or not np.isfinite(point).all()
                or np.linalg.norm(point - self.previous) > self.max_step):
            self.invalid = True
            return 0.0
        self.previous = point
        if grounded is not True or dead or complete:
            return 0.0
        distance = float(np.linalg.norm((point - self.goal)[self.dimensions]))
        improvement = max(0.0, self.best_distance - distance)
        self.best_distance = min(self.best_distance, distance)
        return improvement / self.normalization
