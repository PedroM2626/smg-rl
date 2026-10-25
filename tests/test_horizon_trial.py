import json

import numpy as np
import torch

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.horizon_trial import horizon_trial
from smg_rl.replay import replay_signature
from smg_rl.world_model import ImaginedAgent


def test_horizon_trial_freezes_world_counts_and_requires_both_gates(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = load_config("config/world-model.yaml")
    config["root"] = str(tmp_path)
    config["paths"]["runs"] = str(tmp_path / "runs")
    config["training"]["device"] = "cpu"
    config["environment"]["image_size"] = 16
    settings = {"deterministic_size": 8, "stochastic_size": 2, "embedding_size": 4,
                "hidden_size": 8, "channels": 2, "ensemble_size": 2, "batch_size": 2,
                "sequence_length": 2, "burn_in": 1, "imagination_batch": 2,
                "imagination_horizon": 8}
    config["world_model"] = settings
    source = ImaginedAgent(replay_signature(config), settings)
    source.real_decisions = 123
    checkpoint = tmp_path / "source.pt"
    source.save(checkpoint)
    original = digest(checkpoint)

    class Replay:
        def __init__(self, *args):
            pass

        def refresh(self):
            return {"rejected": []}

        def sample(self, *args):
            return {"images": np.zeros((2, 4, 16, 16), np.uint8),
                    "actions": np.zeros((2, 3), np.int64),
                    "rewards": np.zeros((2, 3), np.float32),
                    "terminated": np.zeros((2, 3), bool),
                    "truncated": np.zeros((2, 3), bool),
                    "mask": np.array([[0, 1, 1]] * 2, np.float32),
                    "is_first": np.array([[True, False, False, False]] * 2),
                    "scenario": np.eye(2, dtype=np.float32)}

    monkeypatch.setattr("smg_rl.horizon_trial.SequenceReplay", Replay)
    monkeypatch.setattr("smg_rl.horizon_trial.freeze_validation", lambda *args: (None, []))
    monkeypatch.setattr("smg_rl.horizon_trial.evaluate", lambda *args: {"gate": {"ready": True}})
    run = horizon_trial(config, checkpoint, updates=2, seeds=(11,))
    result = json.loads((run / "results.json").read_text())
    assert result["joint_comparison_eligible"] and len(result["rows"]) == 2
    for row in result["rows"]:
        variant = {**settings, "imagination_horizon": row["horizon"]}
        candidate = ImaginedAgent.load(row["fits"][0]["checkpoint"], replay_signature(config), variant)
        assert candidate.real_decisions == 123 and candidate.model_updates == 0
        assert candidate.actor_updates == 2
        for key, value in source.world.state_dict().items():
            assert torch.equal(value, candidate.world.state_dict()[key])
    assert digest(checkpoint) == original
    monkeypatch.setattr("smg_rl.horizon_trial.evaluate",
                        lambda agent, fixed, settings, *args: {"gate": {"ready": settings["imagination_horizon"] == 8}})
    blocked = horizon_trial(config, checkpoint, updates=2, seeds=(11,))
    result = json.loads((blocked / "results.json").read_text())
    assert not result["joint_comparison_eligible"]
    assert all(not row["fits"] for row in result["rows"])
