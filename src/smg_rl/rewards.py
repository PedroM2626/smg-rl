"""Progress is a bounded world-space high-water mark, never camera motion."""

import numpy as np


class ProgressReward:
    def __init__(self, start, goal, min_y=-1800.0, max_step_distance=2500.0):
        self.start = np.asarray(start, dtype=np.float64)[[0, 2]]
        self.goal = np.asarray(goal, dtype=np.float64)[[0, 2]]
        self.axis = self.goal - self.start
        self.length_squared = float(self.axis @ self.axis)
        if self.length_squared < 1:
            raise ValueError("Start and goal must be distinct world positions.")
        self.min_y, self.max_step_distance = min_y, max_step_distance
        self.best = 0.0
        self.previous = None

    def reset(self, position):
        self.previous = np.asarray(position, dtype=np.float64)
        self.best = self.fraction(position)

    def fraction(self, position):
        point = np.asarray(position)[[0, 2]]
        return float(np.clip((point - self.start) @ self.axis / self.length_squared, 0, 1))

    def update(self, position, dead=False, complete=False):
        position = np.asarray(position, dtype=np.float64)
        fraction = self.fraction(position)
        valid = (
            np.isfinite(position).all()
            and position[1] >= self.min_y
            and not dead
            and self.previous is not None
            and np.linalg.norm(position - self.previous) <= self.max_step_distance
        )
        self.previous = position
        delta = max(0.0, fraction - self.best) if valid else 0.0
        if valid:
            self.best = max(self.best, fraction)
        return delta, fraction
