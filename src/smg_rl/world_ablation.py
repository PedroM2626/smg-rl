"""Bounded, explicitly authorized offline SMG2 objective ablations; no emulator."""

import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json
from smg_rl.replay import SequenceReplay, replay_signature
from smg_rl.world_model import ImaginedAgent
from smg_rl.world_training import quality_gate, validate_agent

EXPERIMENT = "smg2-world-model-ablation"
FULL = {"reconstruction_weight": 3528.0, "terminal_reward_weight": 5.0,
        "training_terminal_fraction": 0.5, "prior_prediction_weight": 0.5,
        "open_loop_training_horizon": 8, "open_loop_training_weight": 0.5}
ARMS = {
    "continued_baseline": {},
    "terminal_sampling": {"training_terminal_fraction": 0.5},
    "terminal_reward_weight": {"terminal_reward_weight": 5.0},
    "pixel_likelihood_scale": {"reconstruction_weight": 3528.0},
    "one_step_prior": {"prior_prediction_weight": 0.5},
    "eight_step_rollout": {"open_loop_training_horizon": 8, "open_loop_training_weight": 0.5},
    "full_smg1": FULL,
    "full_without_visual_scale": {key: value for key, value in FULL.items() if key != "reconstruction_weight"},
    "moderate_visual_scale": {"reconstruction_weight": 100.0},
    "four_step_rollout": {"open_loop_training_horizon": 4, "open_loop_training_weight": 0.5},
}
METRICS = ("posterior_image_mse", "open_loop_image_mse", "reward_mae",
           "terminal_reward_mae", "continuation_brier", "terminal_brier")


def authorize(config, checkpoint, updates, seeds):
    """A narrow grant does not release the global online/offline training hold."""
    path = Path(config["root"]) / ".local" / "world-ablation-authorization.json"
    grant = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if (not config.get("training", {}).get("enabled", True)
            or replay_signature(config)["game_id"] != "SB4E01"
            or grant.get("experiment") != EXPERIMENT or not grant.get("active")
            or grant.get("checkpoint_sha256") != digest(checkpoint)
            or not 1 <= updates <= grant.get("maximum_updates_per_arm", 0)
            or not seeds or len(set(seeds)) != len(seeds)
            or not set(seeds).issubset(set(grant.get("seeds", [])))):
        raise ValueError("This bounded offline ablation requires a matching explicit authorization grant.")
    return path, grant


def audit_partition(episode_id):
    """Split only existing held-out episodes; never split an episode's chunks."""
    return int(hashlib.sha256(("ablation-audit:" + episode_id).encode()).hexdigest()[:8], 16) % 2 == 0


class FixedValidation:
    def __init__(self, signature, batches):
        self.signature, self.batches, self.position = signature, batches, 0

    def sample(self, *args, **kwargs):
        if not kwargs.get("validation"):
            raise ValueError("Fixed audit data cannot be used as training data.")
        result = self.batches[self.position % len(self.batches)]
        self.position += 1
        return result


def freeze_validation(replay, settings, count, final):
    subset = deepcopy(replay)
    subset.index = [item for item in subset.index
                    if item["validation"] and audit_partition(item["episode_id"]) == final]
    names = replay.signature["scenario_names"]
    if {item["scenario"] for item in subset.index} != set(range(len(names))):
        raise ValueError("Every audit partition must cover every scenario.")
    subset.rng = np.random.default_rng(20261024 + int(final))
    batches = [subset.sample(settings["batch_size"], settings["sequence_length"],
                             settings["burn_in"], validation=True,
                             terminal_fraction=0.0 if i < count // 2 else 0.75)
               for i in range(count)]
    return FixedValidation(replay.signature, batches), sorted({item["episode_id"] for item in subset.index})


def evaluate(agent, fixed, settings, seed, summary):
    fixed.position = 0
    torch.manual_seed(seed)
    values = validate_agent(agent, fixed, {**settings, "validation_batches": len(fixed.batches)})
    if not all(np.isfinite(list(row.values())).all() for row in values.values()):
        raise FloatingPointError("Nonfinite ablation audit metrics.")
    return {"metrics": values,
            "gate": quality_gate(summary, values, settings, fixed.signature["scenario_names"])}


def summarize(rows, scenarios):
    """Report signed paired changes against equal-update control; lower is better."""
    output = {}
    controls = {row["seed"]: row for row in rows if row["arm"] == "continued_baseline"}
    for arm in dict.fromkeys(row["arm"] for row in rows):
        group = [row for row in rows if row["arm"] == arm]
        result = {"seeds": len(group), "scenarios": {},
                  "mean_fitting_seconds": float(np.mean([row["fitting_seconds"] for row in group])),
                  "gate_passes": sum(row["final_audit"]["gate"]["ready"] for row in group)}
        for scenario in scenarios:
            result["scenarios"][scenario] = {}
            for metric in METRICS:
                values = [row["final_audit"]["metrics"][scenario][metric] for row in group]
                changes = [100 * (value / max(controls[row["seed"]]["final_audit"]["metrics"][scenario][metric], 1e-12) - 1)
                           for row, value in zip(group, values)]
                result["scenarios"][scenario][metric] = {
                    "mean": float(np.mean(values)), "std_across_seeds": float(np.std(values)),
                    "paired_percent_change_mean": float(np.mean(changes)),
                    "paired_percent_change_by_seed": changes}
        output[arm] = result
    return output


@torch.no_grad()
def policy_trace(agent, batch):
    """Mean-posterior frozen-actor diagnostic; omit terminal/padded states."""
    data = agent.tensors(batch)
    images, scenario = data["images"], data["scenario"]
    count, frames = images.shape[:2]
    world = agent.world
    embedding = world.encode(images.reshape(-1, 1, world.size, world.size)).reshape(count, frames, -1)
    state, _, _ = world.observe(embedding[:, 0], torch.zeros(count, dtype=torch.long, device=agent.device),
                                world.initial(count), scenario, data["is_first"][:, 0], sample=False)
    values = []
    for step in range(frames - 1):
        state, _, _ = world.observe(embedding[:, step + 1], data["actions"][:, step], state,
                                    scenario, data["is_first"][:, step + 1], sample=False)
        logits = agent.actor(world.feature(state, scenario))
        valid = data["mask"][:, step].bool() & ~data["terminated"][:, step]
        values.append(logits.log_softmax(-1)[valid].cpu())
    trace = torch.cat(values)
    if not len(trace) or not torch.isfinite(trace).all():
        raise ValueError("Policy drift needs finite nonterminal real posterior states.")
    return trace


def policy_drift(reference, candidate):
    """Drift is an action-distribution change, not an improvement/degradation score."""
    if reference.shape != candidate.shape:
        raise ValueError("Policy traces must use identical real observations and masks.")
    probability = reference.exp()
    return {"nonterminal_states": len(reference),
            "source_to_candidate_kl_nats": float((probability * (reference - candidate)).sum(-1).mean()),
            "greedy_action_agreement": float((reference.argmax(-1) == candidate.argmax(-1)).float().mean()),
            "candidate_entropy_nats": float(-(candidate.exp() * candidate).sum(-1).mean())}


def run_ablation(config, checkpoint, updates=128, seeds=(11, 23, 47), validation_batches=8, arms=None):
    names = list(ARMS) if arms is None else list(arms)
    if (not names or len(set(names)) != len(names)
            or "continued_baseline" not in names or any(name not in ARMS for name in names)):
        raise ValueError("Select unique known arms including continued_baseline.")
    selected = {name: ARMS[name] for name in names}
    checkpoint = Path(checkpoint).resolve()
    grant_path, grant = authorize(config, checkpoint, updates, seeds)
    if validation_batches < 4 or validation_batches % 2:
        raise ValueError("Use an even audit batch count of at least four.")
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    settings, signature = metadata["settings"], replay_signature(config)
    if settings != config["world_model"]:
        raise ValueError("The control configuration must match the source checkpoint settings.")
    torch.set_num_threads(config["training"]["torch_threads"])
    replay = SequenceReplay(config["paths"]["replay"], signature)
    summary = replay.refresh()
    if summary["rejected"] or summary["transitions"] < settings.get("minimum_replay", 4096):
        raise ValueError("The experiment needs sufficient valid frozen replay.")
    development, dev_episodes = freeze_validation(replay, settings, validation_batches, False)
    final, final_episodes = freeze_validation(replay, settings, validation_batches, True)
    assert not set(dev_episodes).intersection(final_episodes)
    run = Path(config["paths"]["runs"]) / ("ablation-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))
    run.mkdir(parents=True, exist_ok=False)
    manifest = [{**{key: value for key, value in item.items() if key != "path"},
                 "path": str(item["path"]), "size": item["path"].stat().st_size,
                 "mtime_ns": item["path"].stat().st_mtime_ns} for item in replay.index]
    atomic_json(run / "replay-manifest.json", manifest)
    result = {"schema": 1, "experiment": EXPERIMENT, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_checkpoint": str(checkpoint), "source_sha256": metadata["sha256"],
              "updates_per_arm": updates, "seeds": list(seeds), "arms": selected,
              "validation_batches_per_partition": validation_batches, "replay": summary,
              "development_episodes": dev_episodes, "final_audit_episodes": final_episodes,
              "new_real_decisions": 0, "new_actor_updates": 0, "new_imagined_transitions": 0,
              "scope": "Warm-start offline prediction screening; no real control or time-to-solution claims.",
              "rows": []}
    device = config["training"]["device"]
    distribution_validation = torch.distributions.Distribution._validate_args
    try:
        # These constructors otherwise synchronize CUDA for every Normal check
        # at every recurrent step. Replay, finite losses/gradients and final
        # audit metrics are explicitly checked; no model equation is changed.
        torch.distributions.Distribution.set_default_validate_args(False)
        for seed in seeds:
            for arm, changes in selected.items():
                if (run / "STOP").exists():
                    raise InterruptedError("Offline ablation stopped by marker.")
                agent = ImaginedAgent.load(checkpoint, signature, settings, device)
                original_counts = (agent.real_decisions, agent.actor_updates, agent.imagined_transitions)
                actor = {key: value.clone() for key, value in agent.actor.state_dict().items()}
                variant = {**settings, **changes}
                agent.settings = agent.world.settings = variant
                replay.rng = np.random.default_rng(seed)
                before = time.perf_counter()
                if agent.device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                for step in range(updates):
                    if (run / "STOP").exists():
                        raise InterruptedError("Offline ablation stopped by marker.")
                    torch.manual_seed(seed * 100000 + step)
                    batch = replay.sample(variant["batch_size"], variant["sequence_length"], variant["burn_in"],
                                          terminal_fraction=variant.get("training_terminal_fraction", 0))
                    agent.update_model(batch)
                    if (step + 1) % 32 == 0 or step + 1 == updates:
                        atomic_json(run / "progress.json", {
                            "state": "fitting", "updated_utc": datetime.now(timezone.utc).isoformat(),
                            "seed": seed, "arm": arm, "updates_completed": step + 1,
                            "updates_per_arm": updates, "completed_fits": len(result["rows"]),
                            "total_fits": len(seeds) * len(selected),
                            "elapsed_fit_seconds": time.perf_counter() - before})
                if agent.device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                assert original_counts == (agent.real_decisions, agent.actor_updates, agent.imagined_transitions)
                assert all(torch.equal(value, actor[key]) for key, value in agent.actor.state_dict().items())
                row = {"seed": seed, "arm": arm, "fitting_seconds": elapsed,
                       "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if agent.device.type == "cuda" else 0,
                       "development": evaluate(agent, development, settings, seed, summary)}
                destination = run / f"{arm}-{seed}.pt"
                agent.replay_rng_state = replay.rng.bit_generator.state
                agent.save(destination)
                row["checkpoint"] = str(destination)
                result["rows"].append(row)
                atomic_json(run / "results.json", result)
                print(json.dumps({"stage": "fit", "seed": seed, "arm": arm, "seconds": elapsed}), flush=True)
                del agent
        # Inspect the final partition only after every predeclared fit is finished.
        for row in result["rows"]:
            variant = {**settings, **ARMS[row["arm"]]}
            agent = ImaginedAgent.load(row["checkpoint"], signature, variant, device)
            row["final_audit"] = evaluate(agent, final, settings, row["seed"], summary)
            del agent
        for item in manifest:
            stat = Path(item["path"]).stat()
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise RuntimeError("Replay changed during the frozen experiment.")
        result["summary"] = summarize(result["rows"], signature["scenario_names"])
        result["state"] = "completed"
        atomic_json(run / "results.json", result)
        atomic_json(run / "progress.json", {"state": "completed", "completed_fits": len(result["rows"])})
    except BaseException as exc:
        atomic_json(run / "error.json", {"error": repr(exc)})
        atomic_json(run / "progress.json", {"state": "failed", "error": repr(exc)})
        raise
    finally:
        torch.distributions.Distribution.set_default_validate_args(distribution_validation)
        grant.update(active=False, consumed_by=str(run))
        atomic_json(grant_path, grant)
    return run
