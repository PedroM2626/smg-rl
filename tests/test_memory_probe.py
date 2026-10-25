import numpy as np

from smg_rl.memory_probe import candidates


def test_ram_probe_rejects_timers_coordinates_nonfinite_and_reset_drift():
    progress = np.array([0, 0, 0, 1, 2, 3, 4, 5, 4, 0.], dtype=float)
    timer = np.array([0, 10, 0, 36, 84, 132, 180, 228, 276, 0.])
    positions = np.column_stack([progress * 20 - 50, np.ones(10) * 45, -progress * 100 - 50])
    reset_drift = progress.copy()
    reset_drift[-1] = 1
    invalid = progress.copy()
    invalid[5] = np.nan
    values = np.column_stack([progress, timer, positions[:, 0], reset_drift, invalid, np.zeros(10)])
    with np.errstate(invalid='raise'):
        assert candidates(values, positions).tolist() == [0]
