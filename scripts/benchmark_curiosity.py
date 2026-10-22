"""Seeded synthetic curiosity cost; this is not an emulator or learning benchmark."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from smg_rl.curiosity import IntrinsicReward


def benchmark(samples, warmup, output):
    images = np.random.default_rng(52).integers(0, 256, (samples + 1, 84, 84), dtype=np.uint8)
    results = []
    conditions = [(1, "cpu", 1), (1, "cpu", 8), (1, "cpu", 16), (2, "cpu", 8)]
    if torch.cuda.is_available():
        conditions += [(1, "cuda", 8), (1, "cuda", 16)]
    for threads, device, batch in conditions:
        torch.set_num_threads(threads)
        torch.manual_seed(52)
        module = IntrinsicReward(54, dict(embedding_size=64, neighbors=10, capacity=4000,
                                         rnd_clip=5, update_every=batch, device=device))
        module.reset(images[0])
        durations = []
        for index in range(samples):
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            module.step(images[index], images[index + 1], index % 54)
            if device == "cuda":
                torch.cuda.synchronize()
            durations.append((time.perf_counter() - started) * 1000)
        results.append(dict(threads=threads, device=device, update_every=batch, steps=samples,
                            mean_ms_after_warmup=float(np.mean(durations[warmup:])),
                            p95_ms_after_warmup=float(np.percentile(durations[warmup:], 95)),
                            optimizer_updates=module.updates))
    Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.samples:
        parser.error("Require 0 <= warmup < samples.")
    benchmark(args.samples, args.warmup, args.output)
