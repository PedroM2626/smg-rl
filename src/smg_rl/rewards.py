"""Progress is a bounded world-space high-water mark, never camera motion."""

import numpy as np


class RegionalPotential:
    """Frame-discounted shaping with an absorbing zero terminal potential."""

    def __init__(self, scale=1.0, gamma_per_frame=0.997 ** (1 / 12)):
        if not np.isfinite([scale, gamma_per_frame]).all() or not 0 < scale <= 5 or not 0 < gamma_per_frame <= 1:
            raise ValueError('Invalid regional potential scale/discount.')
        self.scale, self.gamma_per_frame, self.value = scale, gamma_per_frame, 0.0

    def reset(self, progress):
        if not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError('Regional progress must lie in[0,1].')
        self.value = self.scale * progress

    def update(self, progress, frames, terminal=False):
        if not isinstance(frames, int) or frames < 1 or not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError('Shaping needs finite progress and a positive frame count.')
        following = 0.0 if terminal else self.scale * progress
        reward = self.gamma_per_frame ** frames * following - self.value
        self.value = following
        return float(reward)


class RegionalAdvance:
    """Pay each new verified route high-water mark once, without terminal clawback.

    This is a dense auxiliary objective, not policy-invariant potential shaping.
    Duration neutrality requires the validated undiscounted, zero-step-cost profile.
    """

    def __init__(self, scale=1.0):
        if not np.isfinite(scale) or not 0 < scale <= 5:
            raise ValueError('Invalid regional advance scale.')
        self.scale, self.value = scale, 0.0

    def reset(self, progress):
        if not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError('Regional progress must lie in[0,1].')
        self.value = self.scale * progress

    def update(self, progress, frames, terminal=False):
        if not isinstance(frames, int) or frames < 1 or not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError('Advance needs finite progress and a positive frame count.')
        following = max(self.value, self.scale * progress)
        reward, self.value = following - self.value, following
        return float(reward)


class RegionalDistanceAdvance(RegionalAdvance):
    """Explicit experimental high-water reward with a larger total budget."""

    def __init__(self, scale):
        if not np.isfinite(scale) or not 0 < scale <= 20:
            raise ValueError('Invalid experimental distance advance scale.')
        self.scale, self.value = scale, 0.0


class RegionalWaypointBonus:
    """A fixed bonus for each newly committed route waypoint, never taken back.

    `update` receives the number of waypoints the route has committed so far (after a grounded
    landing). Its total over a route is `bonus * segments`, which the objective validation keeps
    below the completion reward."""

    def __init__(self, bonus, segments):
        if (not np.isfinite(bonus) or not 0 < bonus <= 5 or type(segments) is not int
                or not 1 <= segments <= 2000):
            raise ValueError('Invalid waypoint bonus or segment count.')
        self.bonus, self.segments, self.committed = float(bonus), segments, 0
        self.scale = self.bonus * segments

    @property
    def value(self):
        """Total bonus earned so far, in the same units as the other shaping classes' potential."""
        return self.bonus * self.committed

    def reset(self, committed):
        if type(committed) is not int or not 0 <= committed <= self.segments:
            raise ValueError('Committed waypoints must lie in [0, segments].')
        self.committed = committed

    def update(self, committed, frames, terminal=False):
        if (not isinstance(frames, int) or frames < 1 or type(committed) is not int
                or not 0 <= committed <= self.segments):
            raise ValueError('A waypoint bonus needs an integer committed count and a positive frame count.')
        gained = max(0, committed - self.committed)
        self.committed = max(self.committed, committed)
        return self.bonus * gained


def validate_distance_objective(settings, weights, gamma):
    """Allow positive partial failure returns only for an explicit experiment."""
    values = [weights[k] for k in ('step', 'death', 'timeout', 'completion')] + [weights.get('damage', 0.)]
    if (not np.isfinite(values).all() or gamma != 1.0 or settings['gamma_per_frame'] != 1.0
            or weights['step'] != 0 or weights['death'] != weights['timeout']
            or weights['death'] >= 0 or weights['completion'] <= settings['scale']
            or weights.get('damage', 0.) > 0):
        raise ValueError('Invalid experimental distance advance objective.')


def validate_discounted_objective(settings, weights, gamma):
    """A separately signed continuing task; artificial cuts retain bootstrap."""
    values = [weights[k] for k in ('step', 'death', 'timeout', 'completion', 'damage')]
    mode = settings.get('timeout_mode')
    # idle_terminal_penalty: a no-progress cut is a real failure (absorbing, penalized no worse than
    # death); only the global episode limit stays an unpenalized bootstrap truncation.
    timeout_valid = (weights['timeout'] == 0 if mode == 'truncation_bootstrap'
                     else weights['death'] <= weights['timeout'] < 0 if mode == 'idle_terminal_penalty'
                     else False)
    # An explicit opt-in allows an experiment with no failure penalty: dying then costs only the
    # discounted future reward it forfeits, which the owner wants tested against a passive policy.
    unpenalized = settings.get('allow_unpenalized_failure') is True
    death_valid = weights['death'] <= 0 if unpenalized else weights['death'] < 0
    if (not np.isfinite(values).all() or not 0 < gamma < 1 or not timeout_valid
            or weights['step'] != 0 or not death_valid
            or weights['damage'] > 0 or weights['completion'] <= settings['scale']):
        raise ValueError('Discounted distance objective requires unpenalized bootstrap cuts.')


def validate_simple_objective(settings, weights, gamma):
    """Reject profiles that silently reintroduce delayed-failure preference."""
    values = [weights[k] for k in ('step', 'death', 'timeout', 'completion')] + [weights.get('damage', 0.)]
    if (not np.isfinite(values).all() or gamma != 1.0 or settings['gamma_per_frame'] != 1.0
            or weights['step'] != 0 or weights['death'] != weights['timeout']
            or not weights['death'] < -settings['scale']
            or weights['completion'] <= settings['scale'] or weights.get('damage', 0.) > 0):
        raise ValueError('Simple advance requires gamma1, zero time cost and equal bounded failure penalties.')


class ExplorationBudget:
    """Bound per-episode novelty and fade it during a measured progress stall."""

    def __init__(self, limit=None, decay_start=None, decay_end=None):
        if limit is not None and limit < 0:
            raise ValueError("The intrinsic episode budget must be nonnegative.")
        if decay_end is not None and (decay_start is None or not 0 <= decay_start < decay_end):
            raise ValueError("Intrinsic decay requires 0 <= start < end.")
        self.limit, self.decay_start, self.decay_end = limit, decay_start, decay_end
        self.paid = 0.0

    @property
    def remaining(self):
        return None if self.limit is None else max(0.0, self.limit - self.paid)

    def pay(self, raw_reward, idle_steps):
        if not np.isfinite(raw_reward) or raw_reward < 0:
            raise ValueError("Intrinsic reward must be finite and nonnegative.")
        gate = 1.0
        if self.decay_end is not None:
            gate = float(np.clip((self.decay_end - idle_steps) /
                                 (self.decay_end - self.decay_start), 0, 1))
        amount = raw_reward * gate
        if self.remaining is not None:
            amount = min(amount, self.remaining)
        self.paid += amount
        return amount, gate


class DamageTracker:
    """Count net HP loss once, with a fresh baseline after every savestate reset."""

    def reset(self, health):
        self.previous = health

    def update(self, health):
        lost = (
            max(0, self.previous - health)
            if self.previous is not None and health is not None
            else 0
        )
        self.previous = health
        return lost


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
