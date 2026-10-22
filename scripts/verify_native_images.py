"""Replay identical native actions to check preprocessing error and RAM invariance."""

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from smg_rl.config import load_config
from smg_rl.native import background_backend


def verify(config_path, steps=32, output=None):
    config = deepcopy(load_config(config_path))
    folder = Path(output or f".local/image-check-{time.time_ns()}").resolve()
    folder.mkdir(parents=True)
    config["paths"]["user"] = str(folder / "template")
    backend = background_backend(config)
    results = []
    try:
        for variation in config["variations"]:
            reference_images, reference_positions = None, None
            for mode in (0, 1, 2):
                backend.bridge.set(43, mode)
                backend.load_slot(variation["start_slot"])
                images, positions, diagnostics = [], [], []
                for index in range(steps):
                    backend.advance([["w"], ["w", "jump"], ["spin"], [], ["a"], ["s"]][index % 6])
                    sample = backend.memory.sample()
                    images.append(backend.image().copy())
                    positions.append(sample.position)
                    diagnostics.append(backend.bridge.diagnostics())
                    if sample.dead or sample.complete:
                        backend.load_slot(variation["start_slot"])
                images, positions = np.asarray(images), np.asarray(positions)
                if mode == 0:
                    reference_images, reference_positions = images, positions
                difference = np.abs(images.astype(np.int16) - reference_images.astype(np.int16))
                displacement = float(np.max(np.linalg.norm(positions - reference_positions, axis=1)))
                result = {"variation": variation["name"], "capture_mode": mode, "steps": steps,
                          "pixel_mean_absolute_error": float(difference.mean()),
                          "pixel_max_absolute_error": int(difference.max()),
                          "coordinate_max_error": displacement,
                          "mean_readback_us": float(np.mean([d["readback_us"] for d in diagnostics])),
                          "mean_preprocess_us": float(np.mean([d["preprocess_us"] for d in diagnostics])),
                          "image_bytes": diagnostics[-1]["image_bytes"],
                          "source_width": diagnostics[-1]["source_width"],
                          "source_height": diagnostics[-1]["source_height"]}
                results.append(result)
                assert displacement < 0.001, result
                if mode == 1:
                    assert difference.max() <= 1, result
        (folder / "images.json").write_text(json.dumps({"passed": True, "results": results,
            "executable_sha256": hashlib.sha256(Path(config["paths"]["dolphin"]).read_bytes()).hexdigest(),
            "note": "Full RGBA/OpenCV reference versus native area grayscale and experimental GPU-linear reduction. CPU compact must differ by at most one intensity level. GPU pixels are measured, not declared equivalent."}, indent=2), encoding="utf-8")
        print(folder, flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output")
    args = parser.parse_args()
    verify(args.config, args.steps, args.output)
