import csv
import json
from pathlib import Path

import pytest

from smg_rl.reporting import training_report


def make_run(root, name="parallel-example"):
    root.mkdir(exist_ok=True)
    calibration = root / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "validated": True,
                "game_id": "SB4E01",
                "variations": {"spinning": {"start": [0, 0, 0], "goal": [0, 0, -1000]}},
            }
        )
    )
    run = root / name
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "game": {"id": "SB4E01"},
                "paths": {"calibration": str(calibration)},
                "variations": [
                    {"name": "spinning", "title": "Course", "start_slot": 1, "goal_slot": 2}
                ],
            }
        )
    )
    return run


def trajectory(path, progress=0.4, success=False, start=(0, 0, 0), nerve=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(
            action="reset",
            variation="spinning",
            x=start[0],
            y=start[1],
            z=start[2],
            complete=False,
            dead=False,
            best_progress=0.0,
            progress=0.0,
            scene_nerve="0x807D5310",
            outcome="running",
        )
    ]
    rows.append(
        {
            **rows[0],
            "action": "w",
            "z": -1000 * progress,
            "progress": progress,
            "best_progress": progress,
            "complete": success,
            "outcome": "success" if success else "running",
            "scene_nerve": nerve or ("0x807D5318" if success else "0x807D5310"),
        }
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    "validation_name",
    ["background-validation-example", "damage-validation-example", "eval-example"],
)
def test_report_excludes_calibration_and_near_star_resets(tmp_path, validation_name):
    run = make_run(tmp_path)
    validation = make_run(tmp_path, validation_name)
    trajectory(run / "worker-00/spinning/episode-000001.csv")
    trajectory(run / "worker-00/spinning/episode-000002.csv", success=True, start=(0, 0, -1000))
    trajectory(validation / "spinning/episode-000001.csv", success=True)
    report = training_report(tmp_path)
    task = report["tasks"][0]
    assert len(report["runs"]) == 1
    assert task["successful_training_episodes"] == 0
    assert task["recorded_training_episodes"] == 1
    assert task["maximum_valid_progress_percent"] == 40
    assert report["warnings"]


def test_report_separates_world_model_real_episodes_from_offline_imagination(tmp_path):
    online = make_run(tmp_path, "world-model-online")
    offline = make_run(tmp_path, "world-model-offline")
    for run, kind in [(online, "online"), (offline, "offline")]:
        path = run / "config.json"
        config = json.loads(path.read_text())
        config["world_model_run"] = kind
        path.write_text(json.dumps(config))
        trajectory(run / "worker-00/spinning/episode-000001.csv", progress=0.1, success=True)
    report = training_report(tmp_path)
    assert len(report["runs"]) == 1
    assert report["tasks"][0]["successful_training_episodes"] == 1


def test_success_requires_ram_completion_and_live_terminal_is_not_duplicated(tmp_path):
    run = make_run(tmp_path)
    trajectory(run / "worker-00/spinning/episode-000001.csv", success=True, progress=1)
    trajectory(
        run / "worker-00/spinning/episode-000002.csv", success=True, progress=1, nerve="0x806A1BD8"
    )
    trajectory(run / "worker-00/live.csv", success=True, progress=1)
    task = training_report(tmp_path)["tasks"][0]
    assert task["recorded_training_episodes"] == 2
    assert task["successful_training_episodes"] == 1
    assert len(task["success_evidence"]) == 1


def test_report_includes_live_episode_and_saves_matching_measured_plot(tmp_path):
    run = make_run(tmp_path)
    trajectory(run / "worker-00/spinning/episode-000001.csv", progress=0.2)
    trajectory(run / "worker-00/live.csv", progress=0.6)
    report = training_report(tmp_path, tmp_path / "report.json")
    task = report["tasks"][0]
    assert task["maximum_valid_progress_percent"] == 60
    assert task["best_is_partial_episode"]
    assert Path(task["trajectory_plot"]).exists()
    assert json.loads((tmp_path / "report.json").read_text())["tasks"][0]["trajectory_plot"]


def test_missing_evidence_is_unknown_and_legacy_status_is_labeled(tmp_path):
    run = make_run(tmp_path)
    assert training_report(tmp_path)["tasks"][0]["maximum_valid_progress_percent"] is None
    (run / "status.json").write_text(json.dumps({"progress": [0.35]}))
    task = training_report(tmp_path)["tasks"][0]
    assert task["maximum_valid_progress_percent"] == 35
    assert task["trajectory_plot"] is None and task["trajectory_limitation"]
