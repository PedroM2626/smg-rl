"""Read-only actor-objective attribution; never fits or starts an emulator."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from smg_rl.actions import ACTIONS, MODIFIERS
from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.replay import SequenceReplay, replay_signature
from smg_rl.world_model import ImaginedAgent, lambda_returns, weighted_mean


def actor_objective(agent, features, actions, extrinsic, bonus, discounts, values,
                    bootstrap, curiosity_scale, entropy_scale):
    """Reproduce the current actor loss on a fixed rollout, including clipping."""
    returns = lambda_returns(extrinsic + curiosity_scale * bonus, discounts,
                             torch.cat([values[1:], bootstrap[None]]), bootstrap,
                             agent.settings.get("lambda", 0.95)).clamp(-100, 100)
    advantage = returns - values
    advantage = (advantage / advantage.std(unbiased=False).clamp_min(1)).clamp(-10, 10)
    weights = torch.cat([torch.ones_like(discounts[:1]), discounts[:-1]]).cumprod(0)
    distribution = Categorical(logits=agent.actor(features.detach()))
    loss = -weighted_mean(distribution.log_prob(actions) * advantage.detach()
                          + entropy_scale * distribution.entropy(), weights)
    return loss, returns


def _gradient(agent, *args):
    loss, returns = actor_objective(agent, *args)
    gradient = torch.cat([g.flatten() for g in torch.autograd.grad(loss, agent.actor.parameters())])
    return gradient.detach(), returns.detach()


def _cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a[None], b[None]))


def audit_batch(agent, batch, seed):
    """Counterfactual objectives share states, actions, latent noise and critic."""
    torch.manual_seed(seed)
    settings = agent.settings
    with torch.no_grad():
        _, _, state, scenario = agent.world.loss(agent.tensors(batch), sample=False, auxiliary=False)
        selection = torch.randperm(len(scenario), device=agent.device)[:settings.get("imagination_batch", 128)]
        state, scenario = {k: v[selection] for k, v in state.items()}, scenario[selection]
        if not len(scenario):
            raise ValueError("The objective audit needs nonterminal real posterior states.")
        features, actions, extrinsic, bonuses, discounts = [], [], [], [], []
        initial = agent.world.feature(state, scenario)
        probabilities = Categorical(logits=agent.actor(initial)).probs
        all_bonus = torch.stack([agent.world.disagreement(
            initial, torch.full((len(initial),), i, device=agent.device, dtype=torch.long)
        ).clamp(0, 1) for i in range(agent.world.actions)], -1)
        for _ in range(settings.get("imagination_horizon", 8)):
            feature = agent.world.feature(state, scenario)
            action = Categorical(logits=agent.actor(feature)).sample()
            features.append(feature)
            actions.append(action)
            bonuses.append(agent.world.disagreement(feature, action).clamp(0, 1))
            state = agent.world.imagine_step(state, action, scenario)
            following = agent.world.feature(state, scenario)
            extrinsic.append(agent.world.predict_reward(following))
            discounts.append(settings.get("gamma", 0.997) * agent.world.continuation(following).squeeze(-1).sigmoid())
        features, actions, extrinsic, bonuses, discounts = map(
            torch.stack, (features, actions, extrinsic, bonuses, discounts))
        values = agent.target_critic(features).squeeze(-1)
        bootstrap = agent.target_critic(agent.world.feature(state, scenario)).squeeze(-1)
    args = (features, actions, extrinsic, bonuses, discounts, values, bootstrap)
    scale, entropy = settings.get("disagreement_scale", 0.01), settings.get("entropy", 0.003)
    full, returns = _gradient(agent, *args, scale, entropy)
    zero, zero_returns = _gradient(agent, *args, 0.0, entropy)
    external, _ = _gradient(agent, *args, 0.0, 0.0)
    lower_entropy, _ = _gradient(agent, *args, scale, entropy / 10)
    special = torch.tensor([bool(set(a) & {"jump", "spin", "crouch"}) for a in ACTIONS], device=agent.device)
    distribution = Categorical(logits=agent.actor(features.detach()))
    result = {
        "transitions": actions.numel(), "posterior_seeds": len(scenario),
        "scenario_seed_counts": scenario.sum(0).cpu().int().tolist(),
        "entropy_nats": float(distribution.entropy().mean().detach()),
        "special_action_probability": float(probabilities[:, special].sum(-1).mean()),
        "predicted_extrinsic_mean": float(extrinsic.mean()),
        "weighted_bonus_mean": float((scale * bonuses).mean()),
        "action_bonus_range_mean": float((all_bonus.max(-1).values - all_bonus.min(-1).values).mean()),
        "full_gradient_norm": float(full.norm()),
        "extrinsic_gradient_norm": float(external.norm()),
        "remove_intrinsic_gradient_change_fraction": float((full - zero).norm() / full.norm().clamp_min(1e-12)),
        "remove_intrinsic_gradient_cosine": _cosine(full, zero),
        "entropy_gradient_over_extrinsic": float((zero - external).norm() / external.norm().clamp_min(1e-12)),
        "lower_entropy_gradient_change_fraction": float((full - lower_entropy).norm() / full.norm().clamp_min(1e-12)),
        "intrinsic_return_increment_mean": float((returns - zero_returns).mean()),
        "bonus_by_modifier": {"+".join(m) or "none": float(all_bonus[:, i * 9:(i + 1) * 9].mean())
                              for i, m in enumerate(MODIFIERS)},
    }
    if not all(np.isfinite(v) for v in result.values() if isinstance(v, (float, int))):
        raise FloatingPointError("Nonfinite objective attribution.")
    return result


def recording_audit(path):
    """Summarize actual recorded actions, not inferred successful traversal."""
    path = Path(path)
    recording = json.loads((path / "recording.json").read_text(encoding="utf-8"))
    rows, episodes = [], []
    for file in sorted(path.glob("worker-*/*/episode-*.csv")):
        with file.open(newline="", encoding="utf-8") as stream:
            trace = list(csv.DictReader(stream))
        decisions = [r for r in trace if r["action"] != "reset"]
        rows.extend(decisions)
        direction = [tuple(k for k in r["action"].split("+") if k in "wasd" and len(k) == 1)
                     for r in decisions]
        longest, streak = 0, 0
        for i, item in enumerate(direction):
            streak = streak + 1 if item and i and item == direction[i - 1] else int(bool(item))
            longest = max(longest, streak)
        episodes.append({"file": str(file), "decisions": len(decisions),
                         "longest_same_direction_decisions": longest,
                         "progress_reward_total": sum(float(r["progress_reward"]) for r in decisions),
                         "reward_total": sum(float(r["reward"]) for r in decisions)})
    if not rows:
        raise ValueError("Recording has no decision traces.")
    return {"run": str(path), "checkpoint_sha256": recording["checkpoint_sha256"],
            "decisions": len(rows), "episodes": episodes,
            "modifier_fractions": {m: sum(m in r["action"].split("+") for r in rows) / len(rows)
                                   for m in ("jump", "spin", "crouch")},
            "special_action_fraction": sum(any(m in r["action"].split("+") for m in ("jump", "spin", "crouch"))
                                           for r in rows) / len(rows)}


def run(config, checkpoint, output, batches=4, seeds=(11, 23, 47), recordings=()):
    if not 1 <= batches <= 16 or not 1 <= len(seeds) <= 10:
        raise ValueError("Use a bounded diagnostic sample budget.")
    torch.set_num_threads(1)
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError("Retain previous diagnostic results.")
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    settings, signature = metadata["settings"], replay_signature(config)
    replay = SequenceReplay(config["paths"]["replay"], signature)
    summary = replay.refresh()
    if summary["rejected"]:
        raise ValueError("Rejected replay prevents this audit.")
    agent = ImaginedAgent.load(checkpoint, signature, settings, config["training"]["device"])
    counters = (agent.real_decisions, agent.model_updates, agent.actor_updates, agent.imagined_transitions)
    before = {group: {k: v.clone() for k, v in getattr(agent, group).state_dict().items()}
              for group in ("world", "actor", "critic", "target_critic")}
    rows = []
    for seed in seeds:
        replay.rng = np.random.default_rng(seed)
        for index in range(batches):
            batch = replay.sample(settings["batch_size"], settings["sequence_length"], settings["burn_in"],
                                  validation=True, terminal_fraction=0.0)
            rows.append({"seed": seed, "batch": index,
                         **audit_batch(agent, batch, seed * 1000 + index)})
    assert all(torch.equal(v, before[group][k]) for group in before
               for k, v in getattr(agent, group).state_dict().items())
    assert digest(checkpoint) == metadata["sha256"]
    assert counters == (agent.real_decisions, agent.model_updates, agent.actor_updates, agent.imagined_transitions)
    result = {"scope": "Read-only attribution on reused held-out data; no optimizer, emulator or counter updates. Fixed-critic gradient changes are not trained-policy ablations.",
              "checkpoint": str(checkpoint), "sha256": metadata["sha256"], "settings": settings,
              "preserved_counters": dict(zip(("real_decisions", "model_updates", "actor_updates", "imagined_transitions"), counters)),
              "seeds": list(seeds), "batches_per_seed": batches, "rows": rows,
              "summary": {k: {"mean": float(np.mean([r[k] for r in rows])),
                               "min": float(min(r[k] for r in rows)), "max": float(max(r[k] for r in rows))}
                          for k in rows[0] if isinstance(rows[0][k], float)},
              "recordings": [recording_audit(p) for p in recordings]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args()
    result = run(load_config(args.config), args.checkpoint, args.output, args.batches,
                 recordings=args.recording)
    print(json.dumps(result["summary"], indent=2))
