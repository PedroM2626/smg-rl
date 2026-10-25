"""Paired horizon audit and guarded, bounded offline actor-only adaptation."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json
from smg_rl.profiles import require_training_enabled
from smg_rl.replay import SequenceReplay, replay_signature
from smg_rl.train import new_run
from smg_rl.world_ablation import evaluate, freeze_validation
from smg_rl.world_model import ImaginedAgent


def horizon_trial(config, checkpoint, updates=64, seeds=(11, 23, 47), horizons=(8, 16)):
    require_training_enabled(config)
    if not 1 <= updates <= 1024 or not seeds or any(h not in (8, 16, 32) for h in horizons):
        raise ValueError("Use a bounded actor budget and supported horizons.")
    checkpoint = Path(checkpoint)
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    settings, signature = metadata["settings"], replay_signature(config)
    replay = SequenceReplay(config["paths"]["replay"], signature)
    summary = replay.refresh()
    if summary["rejected"]:
        raise ValueError("Rejected replay prevents a horizon trial.")
    torch.set_num_threads(1)
    run = new_run(config, "horizon-trial")
    result = {"source": str(checkpoint), "source_sha256": digest(checkpoint),
              "scope": "Fixed world model; paired offline actor updates only after every horizon audit passes. No new real decisions.",
              "seeds": list(seeds), "updates_per_arm": updates, "rows": []}
    fixed, _ = freeze_validation(replay, settings, 8, True)
    device = config["training"]["device"]
    for horizon in horizons:
        variant = {**settings, "imagination_horizon": horizon}
        agent = ImaginedAgent.load(checkpoint, signature, settings, device)
        audits = [{"seed": seed, **evaluate(agent, fixed, variant, seed, summary)} for seed in seeds]
        row = {"horizon": horizon, "audits": audits, "gate_passes": sum(a["gate"]["ready"] for a in audits),
               "fits": []}
        result["rows"].append(row)
        del agent
    # Predeclare a joint comparison: do not fit either arm if one horizon fails.
    eligible = all(row["gate_passes"] == len(seeds) for row in result["rows"])
    result["joint_comparison_eligible"] = eligible
    atomic_json(run / "results.json", result)
    if eligible:
        for row in result["rows"]:
            variant = {**settings, "imagination_horizon": row["horizon"]}
            for seed in seeds:
                agent = ImaginedAgent.load(checkpoint, signature, settings, device)
                counts = (agent.real_decisions, agent.model_updates)
                starting_imagined = agent.imagined_transitions
                world = {k: v.clone() for k, v in agent.world.state_dict().items()}
                agent.settings = agent.world.settings = variant
                replay.rng = np.random.default_rng(seed)
                started = time.monotonic()
                for step in range(updates):
                    if (run / "STOP").exists():
                        raise InterruptedError("Horizon trial stopped by marker.")
                    torch.manual_seed(seed * 100000 + step)
                    batch = replay.sample(variant["batch_size"], variant["sequence_length"], variant["burn_in"])
                    with torch.no_grad():
                        _, _, posterior, scenarios = agent.world.loss(agent.tensors(batch), auxiliary=False)
                    agent.update_imagination(posterior, scenarios)
                assert counts == (agent.real_decisions, agent.model_updates)
                assert all(torch.equal(value, world[key]) for key, value in agent.world.state_dict().items())
                destination = run / f"horizon-{row['horizon']}-seed-{seed}.pt"
                agent.save(destination)
                row["fits"].append({"seed": seed, "checkpoint": str(destination),
                                    "seconds": time.monotonic() - started,
                                    "new_actor_updates": updates,
                                    "new_imagined_transitions": agent.imagined_transitions - starting_imagined,
                                    "actor_updates": agent.actor_updates, "imagined_transitions": agent.imagined_transitions})
                atomic_json(run / "results.json", result)
                del agent
    result["state"] = "completed"
    atomic_json(run / "results.json", result)
    return run
