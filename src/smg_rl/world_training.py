"""Guarded online/offline world-model workflows; no training starts on import."""

import os
import time
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import psutil
import torch

from smg_rl.diagnostics import atomic_json
from smg_rl.parallel import WorkerFactory, runtime_directory
from smg_rl.profiles import require_training_enabled
from smg_rl.replay import SequenceReplay, replay_signature
from smg_rl.train import new_run
from smg_rl.vector import MeasuredSubprocVecEnv
from smg_rl.world_model import ImaginedAgent


def quality_gate(summary, validation, settings, names):
    reasons = []
    if summary["transitions"] < settings.get("minimum_replay", 4096):
        reasons.append("Insufficient real visual transitions")
    if summary["train_episodes"] < settings.get("minimum_train_episodes", 10):
        reasons.append("Insufficient training episodes")
    if summary["validation_episodes"] < settings.get("minimum_validation_episodes", 4):
        reasons.append("Insufficient held-out episodes")
    for name in names:
        if summary.get("train_scenarios", {}).get(name, 0) < settings.get("minimum_scenario_train", 256):
            reasons.append(f"{name}: insufficient real training coverage")
        if summary.get("validation_scenarios", {}).get(name, 0) < settings.get("minimum_scenario_validation", 64):
            reasons.append(f"{name}: insufficient held-out coverage")
        values = validation.get(name)
        if not values:
            reasons.append(f"{name}: no held-out prediction metrics")
            continue
        checks = {"posterior_image_mse": settings.get("maximum_reconstruction_mse", 0.02),
                  "open_loop_image_mse": settings.get("maximum_open_loop_mse", 0.04),
                  "reward_mae": settings.get("maximum_reward_mae", 0.5),
                  "terminal_reward_mae": settings.get("maximum_terminal_reward_mae", 0.5),
                  "continuation_brier": settings.get("maximum_continuation_brier", 0.1),
                  "terminal_brier": settings.get("maximum_terminal_brier", 0.2)}
        for metric, limit in checks.items():
            value = values.get(metric, float("inf"))
            if not np.isfinite(value) or value > limit:
                reasons.append(f"{name}: {metric} exceeds {limit}")
        if values.get("terminal_examples", 0) < settings.get("minimum_validation_terminals", 2):
            reasons.append(f"{name}: insufficient held-out terminal examples")
        baseline = values.get("persistence_image_mse", 0)
        threshold = max(1e-6, baseline * settings.get("persistence_ratio", 0.98))
        if values.get("open_loop_image_mse", float("inf")) > threshold:
            reasons.append(f"{name}: open-loop prediction does not beat image persistence")
    return {"ready": not reasons, "reasons": reasons,
            "scope": "Held-out visited states only; no guarantee for unseen geometry or star collection."}


@torch.no_grad()
def validate_agent(agent, replay, settings):
    metrics = {name: [] for name in replay.signature["scenario_names"]}
    for _ in range(settings.get("validation_batches", 4)):
        batch = replay.sample(settings.get("batch_size", 8), settings.get("sequence_length", 32),
                              settings.get("burn_in", 8), validation=True, terminal_fraction=0.75)
        for index, name in enumerate(replay.signature["scenario_names"]):
            selection = batch["scenario"].argmax(1) == index
            if selection.any():
                values = agent.world.validation(agent.tensors({key: value[selection] for key, value in batch.items()}),
                                                 settings.get("imagination_horizon", 8))
                metrics[name].append(values)
    result = {}
    for name, rows in metrics.items():
        if not rows:
            continue
        result[name] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        terminal_count = sum(row["terminal_examples"] for row in rows)
        result[name]["terminal_examples"] = terminal_count
        result[name]["terminal_brier"] = sum(row["terminal_brier"] * row["terminal_examples"] for row in rows) / max(1, terminal_count)
        result[name]["terminal_reward_mae"] = sum(row["terminal_reward_mae"] * row["terminal_examples"] for row in rows) / max(1, terminal_count)
        result[name]["validation_transitions"] = sum(row["validation_transitions"] for row in rows)
    return result


def prepare(config, resume=None):
    require_training_enabled(config)
    settings = config["world_model"]
    integers = ("batch_size", "sequence_length", "updates_per_cycle", "collect_vector_steps", "imagination_horizon", "imagination_batch", "checkpoint_steps")
    if any(int(settings.get(key, 0)) < 1 for key in integers) or settings.get("burn_in", -1) < 0:
        raise ValueError("World-model batch, sequence, update, imagination and checkpoint settings must be positive.")
    if settings["imagination_horizon"] > 32:
        raise ValueError("This baseline bounds imagination at 32 steps to limit model-error compounding.")
    if settings["batch_size"] < len(config["variations"]):
        raise ValueError("The replay batch must cover every configured scenario.")
    if (not 0 < settings.get("gamma", 0.997) <= 1 or not 0 <= settings.get("lambda", 0.95) <= 1
            or not 0 <= settings.get("epsilon", 0.05) <= 1
            or not np.isfinite([settings.get("entropy", 0.003), settings.get("disagreement_scale", 0.01)]).all()
            or min(settings.get("entropy", 0.003), settings.get("disagreement_scale", 0.01)) < 0):
        raise ValueError("Invalid world-model discount, exploration or entropy settings.")
    if not config.get("replay", {}).get("enabled"):
        raise ValueError("World-model training requires visual replay recording.")
    torch.set_num_threads(config["training"]["torch_threads"])
    torch.manual_seed(config["training"]["seed"])
    signature = replay_signature(config)
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    agent = ImaginedAgent.load(resume, signature, settings, device) if resume else ImaginedAgent(signature, settings, device)
    replay = SequenceReplay(config["paths"]["replay"], signature, config["training"]["seed"])
    if agent.replay_rng_state is not None:
        replay.rng.bit_generator.state = agent.replay_rng_state
    replay.refresh()
    return agent, replay


def update_cycle(agent, replay, settings):
    summary = replay.refresh()
    validation, metrics, seeds, scenarios = {}, {}, None, None
    if (summary["transitions"] >= settings.get("minimum_replay", 4096)
            and summary["train_episodes"] and summary["validation_episodes"]):
        for _ in range(settings["updates_per_cycle"]):
            batch = replay.sample(settings["batch_size"], settings["sequence_length"], settings["burn_in"])
            metrics, seeds, scenarios = agent.update_model(batch)
        validation = validate_agent(agent, replay, settings)
    gate = quality_gate(summary, validation, settings, replay.signature["scenario_names"])
    if gate["ready"] and (seeds is None or not len(scenarios)):
        gate["ready"] = False
        gate["reasons"].append("No nonterminal real posterior seeds for imagination")
    if gate["ready"] and seeds is not None and len(scenarios):
        metrics.update(agent.update_imagination(seeds, scenarios))
    return {"replay": summary, "validation": validation, "imagination_gate": gate, "losses": metrics}


def publish(run, agent, started, state, diagnostics=None, infos=(), start_real=0):
    elapsed = time.monotonic() - started
    data = {"state": state, "algorithm": "gaussian-rssm-imagination-v1",
            "timesteps": agent.real_decisions, "real_decisions": agent.real_decisions,
            "model_updates": agent.model_updates, "actor_updates": agent.actor_updates,
            "imagined_transitions": agent.imagined_transitions,
            "elapsed_seconds": elapsed, "decisions_per_second": (agent.real_decisions - start_real) / max(elapsed, 1e-6),
            "updated_at": datetime.now(timezone.utc).isoformat(), "progress": [info.get("best_progress", 0) for info in infos],
            "diagnostics": diagnostics or {},
            "note": "Only real RAM-verified episodes can establish mission completion. Imagined transitions are not real timesteps."}
    atomic_json(run / "status.json", data)


def train_world_model(config, steps=None, resume=None):
    # The user hold is checked before any emulator is created.
    require_training_enabled(config)
    config = deepcopy(config)
    agent, replay = prepare(config, resume)
    settings = config["world_model"]
    config["world_model_run"] = "online"
    run = new_run(config, "world-model")
    workers = config["parallel"]["workers"]
    budget = steps if steps is not None else config["training"]["steps"]
    if budget < 1:
        raise ValueError("Real-decision budget must be positive.")
    config["paths"]["user"] = str(runtime_directory(config, run) / "template")
    controller = {"pid": os.getpid(), "created_time": psutil.Process().create_time(),
                  "workers": workers, "algorithm": "gaussian-rssm-imagination-v1", "resume": str(resume) if resume else None}
    atomic_json(run / "controller.json", controller)
    environments = None
    started, start_real = time.monotonic(), agent.real_decisions
    target, checkpoint_at = agent.real_decisions + budget, agent.real_decisions
    diagnostics, infos, state = {}, [], "running"
    try:
        shape = (config["environment"]["frame_stack"], config["environment"]["image_size"], config["environment"]["image_size"])
        environments = MeasuredSubprocVecEnv([WorkerFactory(config, i, str(run)) for i in range(workers)],
                                             shape, shared=False, receive_order="ordered",
                                             timeout=config["parallel"].get("worker_timeout", 120))
        environments.seed(config["training"]["seed"])
        observation = environments.reset()
        latent, previous, first, vector_steps = None, None, np.ones(workers, dtype=bool), 0
        while agent.real_decisions < target and not (run / "STOP").exists():
            actions, latent, previous = agent.act(observation, latent, previous, first,
                                                  epsilon=1.0 if agent.actor_updates == 0 else settings.get("epsilon", 0.05))
            observation, _, first, infos = environments.step(actions)
            agent.real_decisions += workers
            vector_steps += 1
            if vector_steps % settings["collect_vector_steps"] == 0:
                diagnostics = update_cycle(agent, replay, settings)
            if agent.real_decisions - checkpoint_at >= settings["checkpoint_steps"]:
                agent.replay_rng_state = replay.rng.bit_generator.state
                agent.save(run / f"world-{agent.real_decisions}.pt")
                checkpoint_at = agent.real_decisions
                # Retain three periodic checkpoints plus final; replay has its own budget.
                saved = sorted(run.glob("world-[0-9]*.pt"), key=lambda p: int(p.stem[6:]))
                for old in saved[:-3]:
                    old.unlink()
                    old.with_suffix(".json").unlink(missing_ok=True)
            if vector_steps % 16 == 0:
                publish(run, agent, started, state, diagnostics, infos, start_real)
        state = "stopped" if (run / "STOP").exists() else "completed"
    except KeyboardInterrupt:
        state = "stopped"
    except BaseException as exc:
        state = "failed"
        (run / "error.txt").write_text(repr(exc), encoding="utf-8")
        raise
    finally:
        try:
            agent.replay_rng_state = replay.rng.bit_generator.state
            agent.save(run / "world-final.pt")
            publish(run, agent, started, state, diagnostics, infos, start_real)
        finally:
            if environments is not None:
                environments.close()
    return run


def fit_world_model(config, cycles=100, resume=None):
    require_training_enabled(config)
    if cycles < 1:
        raise ValueError("Offline cycles must be positive.")
    agent, replay = prepare(config, resume)
    summary = replay.summary()
    if summary["transitions"] < config["world_model"].get("minimum_replay", 4096):
        raise ValueError("Insufficient visual replay. Historical coordinate CSVs cannot replace pixels.")
    if not summary["train_episodes"] or not summary["validation_episodes"]:
        raise ValueError("Offline fitting requires separate training and validation episodes.")
    config = deepcopy(config)
    config["world_model_run"] = "offline"
    run = new_run(config, "world-model-offline")
    atomic_json(run / "controller.json", {"pid": os.getpid(), "created_time": psutil.Process().create_time(),
                                           "workers": 0, "algorithm": "gaussian-rssm-offline-v1"})
    started, start_real, diagnostics = time.monotonic(), agent.real_decisions, {}
    state = "completed"
    try:
        for _ in range(cycles):
            if (run / "STOP").exists():
                state = "stopped"
                break
            diagnostics = update_cycle(agent, replay, config["world_model"])
            publish(run, agent, started, "running", diagnostics, start_real=start_real)
    except BaseException:
        state = "failed"
        raise
    finally:
        agent.replay_rng_state = replay.rng.bit_generator.state
        agent.save(run / "world-final.pt")
        publish(run, agent, started, state, diagnostics, start_real=start_real)
    return run
