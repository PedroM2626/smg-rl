"""Live calibration and reproducible positive/negative reward checks."""

import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from smg_rl.memory import GameMemory, MemoryFault
from smg_rl.plotting import plot_trajectory
from smg_rl.profiles import game_id


def write_trace(path, samples, variation, outcome, start, goal):
    from smg_rl.rewards import ProgressReward

    progress = ProgressReward(start, goal)
    fields = [
        "step",
        "variation",
        "x",
        "y",
        "z",
        "progress",
        "outcome",
        "actor_nerve",
        "scene_nerve",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, sample in enumerate(samples):
            writer.writerow(
                dict(
                    step=i,
                    variation=variation,
                    x=sample.position[0],
                    y=sample.position[1],
                    z=sample.position[2],
                    progress=progress.fraction(sample.position),
                    outcome=outcome if i == len(samples) - 1 else "running",
                    actor_nerve=f"0x{sample.actor_nerve:08X}",
                    scene_nerve=f"0x{sample.scene_nerve:08X}",
                )
            )
    return plot_trajectory(path)


def calibrate(config, backend, output_dir):
    memory = getattr(backend, "memory", None) or GameMemory(game_id=game_id(config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "game_id": game_id(config),
        "backend": config.get("parallel", {}).get("backend", "legacy")
        if "parallel" in config
        else "desktop",
        "emulator_sha256": hashlib.sha256(
            Path(config["paths"]["dolphin"]).read_bytes()
        ).hexdigest(),
        "validated": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variations": {},
    }
    for variation in config["variations"]:
        name = variation["name"]
        baseline = {}
        for label, slot in (("start", variation["start_slot"]), ("goal", variation["goal_slot"])):
            backend.load_slot(slot)
            first = memory.sample()
            time.sleep(0.15)
            second = memory.sample()
            if first.scene_frame != second.scene_frame:
                raise RuntimeError("Pause did not stop emulation. Check the Dolphin F10 binding.")
            if first.scenario != variation["scenario"] or first.complete or first.dead:
                raise MemoryFault(f"The {label} state is not a live baseline for {name}.")
            baseline[label] = first.position
            baseline[f"{label}_telemetry"] = first.to_dict()
        print(f"{name}: start={baseline['start']}, goal={baseline['goal']}", flush=True)
        # Both near-star states must begin incomplete, then complete after jumping.
        samples = [memory.sample()]
        backend.advance(["jump"])
        for _ in range(60):
            sample = memory.sample()
            samples.append(sample)
            if sample.complete or sample.dead:
                break
            backend.advance([])
        write_trace(
            output_dir / f"{name}-star.csv",
            samples,
            name,
            "success" if samples[-1].complete else "failed",
            baseline["start"],
            baseline["goal"],
        )
        if not samples[-1].complete or samples[-1].dead:
            raise RuntimeError(f"Star collection could not be validated for {name}.")
        baseline["completion_telemetry"] = samples[-1].to_dict()
        # Walk backward from the start into the void; no RAM is modified.
        backend.load_slot(variation["start_slot"])
        samples = [memory.sample()]
        for _ in range(100):
            backend.advance(config.get("validation", {}).get("death_action", ["s"]))
            sample = memory.sample()
            samples.append(sample)
            if sample.dead or sample.complete:
                break
        write_trace(
            output_dir / f"{name}-death.csv",
            samples,
            name,
            "death" if samples[-1].dead else "failed",
            baseline["start"],
            baseline["goal"],
        )
        if not samples[-1].dead or samples[-1].complete:
            raise RuntimeError(f"Death could not be validated for {name}.")
        baseline["death_telemetry"] = samples[-1].to_dict()
        # Prove coordinates respond to a movement input without death/completion.
        backend.load_slot(variation["start_slot"])
        before = memory.sample()
        # Short frame-based decisions may finish before the spawn animation
        # releases player control; require measured movement within a few steps.
        for _ in range(5):
            backend.advance(["w"])
            after = memory.sample()
            if (
                after.dead
                or after.complete
                or np.linalg.norm(np.array(after.position) - before.position) >= 1
            ):
                break
        if (
            after.dead
            or after.complete
            or np.linalg.norm(np.array(after.position) - before.position) < 1
        ):
            raise RuntimeError(f"Movement/coordinate validation failed for {name}.")
        baseline["movement_telemetry"] = after.to_dict()
        result["variations"][name] = baseline
        print(f"{name}: completion, death, movement, and pause checks passed", flush=True)
    result["validated"] = True
    destination = Path(config["paths"]["calibration"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    backend.load_slot(1)
    return result
