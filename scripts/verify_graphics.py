"""Check fixed-action trajectory and image differences across graphics backends."""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from smg_rl.config import load_config
from smg_rl.native import background_backend


def verify(config_path, output, steps=64):
    config = load_config(config_path)
    results, reference = [], {}
    folder = Path(f".local/graphics-check-{time.time_ns()}").resolve()
    for graphics in ("D3D", "Vulkan"):
        current = deepcopy(config)
        current["parallel"]["graphics_backend"] = graphics
        current["paths"]["user"] = str(folder / graphics / "template")
        backend = background_backend(current)
        try:
            for variation in config["variations"]:
                backend.load_slot(variation["start_slot"])
                initial = backend.memory.sample().position
                images, positions, signals = [], [], []
                for index in range(steps):
                    backend.advance([["w"], ["w", "jump"], ["spin"], [], ["a"], ["s"]][index % 6])
                    sample = backend.memory.sample()
                    images.append(backend.image())
                    positions.append(sample.position)
                    signals.append([sample.dead, sample.complete, sample.health])
                    if sample.dead or sample.complete:
                        backend.load_slot(variation["start_slot"])
                images, positions = np.asarray(images), np.asarray(positions)
                if graphics == "D3D":
                    reference[variation["name"]] = (images, positions, signals)
                original, coordinates, original_signals = reference[variation["name"]]
                errors = np.abs(images.astype(np.int16) - original.astype(np.int16))
                result = dict(variation=variation["name"], graphics_backend=graphics, steps=steps,
                              image_mean_error=float(errors.mean()), image_max_error=int(errors.max()),
                              position_max_error=float(np.linalg.norm(positions - coordinates, axis=1).max()),
                              displacement_from_reset=float(np.linalg.norm(positions - initial, axis=1).max()),
                              signals_equal=signals == original_signals)
                results.append(result)
                Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")
                assert result["position_max_error"] < 0.001 and result["signals_equal"], result
                assert result["displacement_from_reset"] > 10, result
                # Different renderers can round a few edge pixels differently.
                assert result["image_mean_error"] <= 0.5, result
                print(result, flush=True)
        finally:
            backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    verify(args.config, args.output, args.steps)
