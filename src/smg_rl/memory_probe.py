"""Conservative scalar screening; correlation does not validate native progress."""

import numpy as np


def candidates(values, positions):
    """Reject idle timers, reset drift and exact coordinate aliases."""
    if values.ndim != 2 or len(values) != 10 or np.asarray(positions).shape != (10, 3):
        raise ValueError('The prefix probe requires ten aligned scalar/XYZ snapshots.')
    valid = np.isfinite(values).all(0) & (np.abs(values) < 1e6).all(0)
    values = np.where(valid[None], values, 0.0)
    valid &= np.abs(values[0] - values[1]) < 1e-3
    valid &= np.abs(values[0] - values[2]) < 1e-3
    valid &= np.abs(values[0] - values[-1]) < 1e-3
    moving = values[3:8]
    delta = np.diff(moving, axis=0)
    valid &= ((delta >= -1e-3).all(0) | (delta <= 1e-3).all(0))
    valid &= np.ptp(moving, axis=0) > .5
    for axis in range(3):
        valid &= ~np.isclose(values, np.asarray(positions)[:, axis, None], atol=1e-2).all(0)
    return np.flatnonzero(valid)
