"""Exercise native frame completion, reset barriers, and paused-state stability."""

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

from smg_rl.config import load_config
from smg_rl.native import background_backend


def verify(config_path, steps=256, output=None):
    config = deepcopy(load_config(config_path))
    folder = Path(output or f".local/protocol-check-{time.time_ns()}").resolve()
    folder.mkdir(parents=True)
    config["paths"]["user"] = str(folder / "template")
    backend = background_backend(config)
    checks, resets, pause_checks, digest = 0, 0, 0, hashlib.sha256()
    started = time.perf_counter()
    try:
        for variation in config["variations"]:
            backend.load_slot(variation["start_slot"])
            for index in range(steps):
                frames = [1, 2, 4, 12, 24][index % 5]
                config["parallel"]["frame_skip"] = frames
                backend.advance([[], ["w"], ["jump"], ["spin"], ["a"], ["s"]][index % 6])
                assert backend.last_frame_count == frames
                diagnostics = backend.bridge.diagnostics()
                assert diagnostics["step_complete_sequence"] == diagnostics["response"]
                assert diagnostics["field_count"] == diagnostics["acknowledged_field"]
                digest.update(backend.image().tobytes())
                checks += 1
                if index % 16 == 0:
                    before = backend.memory.sample()
                    field = backend.bridge.get(17)
                    time.sleep(0.025)
                    after = backend.memory.sample()
                    assert before == after and backend.bridge.get(17) == field
                    pause_checks += 1
                current = backend.memory.sample()
                if current.dead or current.complete or index % 37 == 0:
                    backend.load_slot(variation["start_slot"])
                    current = backend.memory.sample()
                    assert not current.dead and not current.complete
                    assert current.scenario == variation["scenario"]
                    resets += 1
        result = {"passed": True, "commands": checks, "resets": resets,
                  "executable_sha256": hashlib.sha256(Path(config["paths"]["dolphin"]).read_bytes()).hexdigest(),
                  "paused_wait_checks": pause_checks, "observation_digest": digest.hexdigest(),
                  "elapsed_seconds": time.perf_counter() - started,
                  "native_diagnostics": backend.bridge.diagnostics()}
        (folder / "protocol.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(folder, flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/native.yaml")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--output")
    args = parser.parse_args()
    verify(args.config, args.steps, args.output)
