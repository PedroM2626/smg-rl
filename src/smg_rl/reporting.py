"""Audit training-only success and valid progress across games and live episodes."""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import psutil

from smg_rl.plotting import plot_trajectory
from smg_rl.profiles import PROFILES, game_id


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def controller_alive(controller):
    try:
        process = psutil.Process(controller["pid"])
        return abs(process.create_time() - controller["created_time"]) < 0.01
    except (KeyError, psutil.Error):
        return False


def training_report(root, output=None):
    """Exclude validation/evaluation roots and require a verified start reset."""
    root = Path(root)
    tasks, runs, warnings, evidence = {}, [], [], {}
    for config_path in sorted(root.rglob("config.json")):
        run = config_path.parent
        if not run.name.startswith(("train-", "parallel-", "world-model-", 'dqn-')):
            continue
        config = read_json(config_path, {})
        if run.name.startswith("world-model-") and config.get("world_model_run") != "online":
            continue
        if run.name.startswith('dqn-') and config.get('model_free_run') != 'online':
            continue
        gid = game_id(config)
        if gid not in PROFILES or not config.get("variations"):
            continue
        calibration = read_json(config["paths"]["calibration"], {})
        if not calibration.get("validated") or calibration.get("game_id", "SB4E01") != gid:
            warnings.append(f"Unverified calibration; skipped {run}")
            continue
        status = read_json(run / "status.json", {})
        controller = read_json(run / "controller.json", {})
        alive = controller_alive(controller) if controller else False
        runs.append(
            {
                "path": str(run),
                "algorithm": config.get('dqn', {}).get('algorithm', 'rssm' if config.get('world_model_run') else 'ppo'),
                "game_id": gid,
                "controller_running": alive,
                "state": status.get(
                    "state", "failed" if (run / "error.txt").exists() else "untracked"
                ),
                "timesteps": status.get("timesteps"),
                "decisions_per_second": status.get("decisions_per_second"),
                "error": (run / "error.txt").read_text(encoding="utf-8")
                if (run / "error.txt").exists()
                else None,
            }
        )
        for variation in config["variations"]:
            name = variation["name"]
            key = f"{gid}/{name}"
            task = tasks.setdefault(
                key,
                {
                    "game_id": gid,
                    "variation": name,
                    "title": variation["title"],
                    "successful_training_episodes": 0,
                    "recorded_training_episodes": 0,
                    "maximum_valid_progress_percent": None,
                    "best_evidence": None,
                    "success_evidence": [],
                    "trajectory_plot": None,
                },
            )
            baseline = calibration.get("variations", {}).get(name, {}).get("start")
            if baseline is None:
                warnings.append(f"Missing start calibration for {key} in {run}")
                continue
            paths = sorted(run.rglob(f"{name}/episode-*.csv"))
            paths.extend(run.rglob("live.csv"))
            for path in paths:
                try:
                    with path.open(encoding="utf-8", newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    if not rows or rows[0].get("variation") != name:
                        continue
                    if rows[0].get("action") != "reset":
                        continue
                    if rows[0].get("start_kind", "full_start") != "full_start":
                        continue  # Assisted starts never establish full-start success.
                    start = [float(rows[0][axis]) for axis in ("x", "y", "z")]
                    if not all(math.isfinite(x) for x in start) or math.dist(start, baseline) > 500:
                        warnings.append(f"Rejected non-start trajectory: {path}")
                        continue
                    if rows[0].get("complete") == "True" or rows[0].get("dead") == "True":
                        continue
                    partial = path.name == "live.csv"
                    if partial and rows[-1].get("outcome") != "running":
                        continue  # Completed CSV is authoritative; do not double-count.
                    progress = max(float(row["best_progress"]) for row in rows)
                    if not math.isfinite(progress) or not 0 <= progress <= 1:
                        raise ValueError("Invalid best_progress")
                    if not partial:
                        task["recorded_training_episodes"] += 1
                        last = rows[-1]
                        success = (
                            last.get("outcome") == "success"
                            and last.get("complete") == "True"
                            and last.get("dead") != "True"
                            and int(last["scene_nerve"], 16) in PROFILES[gid].complete
                        )
                        if success:
                            task["successful_training_episodes"] += 1
                            task["success_evidence"].append(str(path))
                    if (
                        task["best_evidence"] is None
                        or progress * 100 > task["maximum_valid_progress_percent"]
                    ):
                        task["maximum_valid_progress_percent"] = progress * 100
                        task["best_evidence"] = str(path)
                        task["best_is_partial_episode"] = partial
                        task.pop("trajectory_limitation", None)
                        evidence[key] = rows
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    warnings.append(f"Could not audit {path}: {exc}")
            # Older workers have no live.csv. Their cached RAM-valid episode high
            # water is useful, but cannot supply an unrecorded trajectory.
            for index, progress in enumerate(status.get("progress", [])):
                starts = status.get("start_kinds", [])
                if index < len(starts) and starts[index] != "full_start":
                    continue
                if index % len(config["variations"]) != config["variations"].index(variation):
                    continue
                if (
                    isinstance(progress, (float, int))
                    and 0 <= progress <= 1
                    and (
                        task["best_evidence"] is None
                        or progress * 100 > task["maximum_valid_progress_percent"]
                    )
                ):
                    task["maximum_valid_progress_percent"] = progress * 100
                    task["best_evidence"] = str(run / "status.json")
                    task["best_is_partial_episode"] = True
                    task["trajectory_limitation"] = (
                        "Status-only record from an older worker; no matching live trajectory."
                    )
                    evidence.pop(key, None)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "progress_definition": "Maximum RAM-coordinate-derived valid high-water X/Z projection; not native game completion percentage.",
        "tasks": list(tasks.values()),
        "runs": runs,
        "warnings": warnings,
        "supervisors": [
            {
                "path": str(path.parent),
                "controller_running": controller_alive(
                    read_json(path.parent / "controller.json", {})
                ),
                **read_json(path, {}),
            }
            for path in sorted(root.rglob("supervised-*/status.json"))
        ],
    }
    if output:
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        trajectories = output.parent / (output.stem + "-trajectories")
        trajectories.mkdir(exist_ok=True)
        for key, rows in evidence.items():
            path = trajectories / (key.replace("/", "-") + ".csv")
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            tasks[key]["trajectory_plot"] = str(plot_trajectory(path))
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
