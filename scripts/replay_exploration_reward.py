"""Counterfactual incentive audit on recorded trajectories, not a learning evaluation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from smg_rl.config import load_config
from smg_rl.rewards import ExplorationBudget


def replay(config_path, run, output, last=50):
    config = load_config(config_path)
    weights, environment = config["rewards"], config["environment"]
    results = []
    for worker in sorted(Path(run).glob("worker-*")):
        summaries = [json.loads(line) for line in
                     (worker / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        for episode in summaries[-last:]:
            if episode["outcome"] not in ("timeout", "death", "success"):
                continue
            source = worker / episode["variation"] / f"episode-{episode['episode']:06d}.csv"
            with source.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))[1:]
            budget = ExplorationBudget(weights["intrinsic_episode_budget"],
                                       weights["intrinsic_decay_start"], weights["intrinsic_decay_end"])
            idle, best, total, steps = 0, 0.0, 0.0, 0
            for row in rows:
                steps += 1
                high = float(row["best_progress"])
                delta = max(0.0, high - best)
                best = max(best, high)
                idle = 0 if delta > 1e-5 else idle + 1
                intrinsic, _ = budget.pay(float(row["intrinsic_reward"]) / 0.02 * weights["intrinsic"], idle)
                reward = weights["step"] + weights["progress"] * delta + intrinsic
                complete, dead = row["complete"] == "True", row["dead"] == "True"
                reward += float(row.get("damage_penalty") or 0)
                timeout = not (complete or dead) and (idle >= environment["idle_limit"] or
                                                      steps >= environment["max_steps"])
                reward += weights["completion"] if complete else weights["death"] if dead else weights["timeout"] if timeout else 0
                total += reward
                if complete or dead or timeout:
                    break
            results.append({"source": str(source), "recorded_return": episode["reward"],
                            "counterfactual_return": total, "counterfactual_steps": steps,
                            "intrinsic_paid": budget.paid, "recorded_outcome": episode["outcome"]})
    result = {"config": str(config_path), "source_run": str(run), "episodes": len(results),
              "mean_recorded_return": float(np.mean([r["recorded_return"] for r in results])),
              "mean_counterfactual_return": float(np.mean([r["counterfactual_return"] for r in results])),
              "note": "Reuses legacy intrinsic bonuses divided by their 0.02 coefficient. Stops recorded prefixes at the new idle limit. Does not simulate changed policy actions, batched learning, or future outcomes.",
              "traces": results}
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print({k: v for k, v in result.items() if k != "traces"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--last", type=int, default=50)
    args = parser.parse_args()
    replay(args.config, args.run, args.output, args.last)
