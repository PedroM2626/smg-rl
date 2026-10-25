"""Prediction-only fragment construction, separate from trainable replay."""

import numpy as np


def validation_batches(traces, settings):
    if len(traces) != 4:
        raise ValueError('This bounded audit requires four complete trajectories.')
    rng = np.random.default_rng(202610251)
    length, burn = settings['sequence_length'], settings['burn_in']
    width = length + burn
    batches = []
    for batch_index in range(8):
        rows = []
        for index in range(settings['batch_size']):
            trace = traces[index % len(traces)]
            n = len(trace['actions'])
            if n < 1 or len(trace['images']) != n + 1:
                raise ValueError('Prediction audit frames/transitions are misaligned.')
            learn = max(0, n - length) if batch_index >= 4 else int(rng.integers(n))
            start = max(0, learn - burn)
            available = min(n - start, width)
            frames = trace['images'][start:start + available + 1]
            row = {'images': np.concatenate([frames, np.repeat(frames[-1:], width - available, axis=0)]),
                   'scenario': trace['scenario'],
                   'is_first': np.pad(trace['is_first'][start:start + available + 1], (0, width - available)),
                   'mask': np.zeros(width, np.float32)}
            for name in ('actions', 'rewards', 'terminated', 'truncated'):
                row[name] = np.pad(trace[name][start:start + available], (0, width - available))
            row['mask'][learn - start:available] = 1
            rows.append(row)
        batches.append({key: np.stack([row[key] for row in rows]) for key in rows[0]})
    return batches
