"""Bounded wall-clock stage measurements and resilient diagnostic publication."""

import json
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np


def atomic_json(path, data):
    """A Windows viewer lock must not abort a rollout; retain the previous snapshot."""
    path = Path(path)
    pending = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        pending.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        for delay in (0, 0.005, 0.01, 0.02):
            if delay:
                time.sleep(delay)
            try:
                pending.replace(path)
                return True
            except PermissionError:
                continue
        return False
    finally:
        try:
            pending.unlink(missing_ok=True)
        except PermissionError:
            pass


class StageProfiler:
    """Totals cover all calls; percentiles cover the most recent 512 per stage.

    Nested stages overlap. They must not be summed as exclusive percentages.
    Measurements use perf_counter and do not synchronize CUDA kernels.
    """

    def __init__(self):
        self.values = defaultdict(lambda: deque(maxlen=512))
        self.counts = defaultdict(int)
        self.totals = defaultdict(float)
        self.last = {}

    @contextmanager
    def measure(self, stage):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            self.values[stage].append(elapsed)
            self.counts[stage] += 1
            self.totals[stage] += elapsed
            self.last[stage] = elapsed

    def snapshot(self):
        return {
            stage: {
                "count": self.counts[stage],
                "total_ms": self.totals[stage],
                "mean_ms": self.totals[stage] / self.counts[stage],
                "recent_p50_ms": float(np.percentile(values, 50)),
                "recent_p95_ms": float(np.percentile(values, 95)),
            }
            for stage, values in self.values.items()
        }


def measured(stage):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            if not hasattr(self, "profiler"):
                self.profiler = StageProfiler()
            with self.profiler.measure(stage):
                return function(self, *args, **kwargs)

        return wrapped

    return decorate
