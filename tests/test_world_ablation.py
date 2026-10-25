import hashlib
import json

import numpy as np
import pytest
import torch

from smg_rl.config import load_config
from smg_rl.profiles import require_training_enabled
from smg_rl.replay import ReplayRecorder, replay_signature
from smg_rl.world_ablation import (
    ARMS,
    EXPERIMENT,
    audit_partition,
    authorize,
    policy_drift,
    policy_trace,
    run_ablation,
)
from smg_rl.world_model import ImaginedAgent


def test_ablation_grant_does_not_release_general_training_hold(tmp_path):
    config = load_config("config/world-model.yaml")
    config["root"] = str(tmp_path)
    (tmp_path / ".local").mkdir()
    (tmp_path / ".local/training-hold.json").write_text("{}")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="authorization"):
        authorize(config, checkpoint, 1, [11])
    grant = {"experiment": EXPERIMENT, "active": True,
             "checkpoint_sha256": hashlib.sha256(b"fixture").hexdigest(),
             "maximum_updates_per_arm": 2, "seeds": [11]}
    (tmp_path / ".local/world-ablation-authorization.json").write_text(json.dumps(grant))
    authorize(config, checkpoint, 1, [11])
    with pytest.raises(ValueError, match="on hold"):
        require_training_enabled(config)
    with pytest.raises(ValueError, match="authorization"):
        authorize(config, checkpoint, 3, [11])
    with pytest.raises(ValueError, match="authorization"):
        authorize(config, checkpoint, 1, [11, 11])


def test_offline_paired_experiment_keeps_actors_and_real_counts(tmp_path, monkeypatch):
    config = load_config("config/world-model.yaml")
    config["root"] = str(tmp_path)
    config["environment"]["image_size"] = 16
    config["paths"]["replay"] = str(tmp_path / "replay")
    config["paths"]["runs"] = str(tmp_path / "runs")
    config["training"]["device"] = "cpu"
    config["world_model"].update(deterministic_size=8, stochastic_size=2,
                                 embedding_size=4, hidden_size=8, channels=2,
                                 ensemble_size=2, batch_size=2, sequence_length=2,
                                 burn_in=1, minimum_replay=1)
    signature = replay_signature(config)
    recorder = ReplayRecorder(tmp_path / "replay", signature)
    counter = 0
    for scenario in range(2):
        for partition in ("train", "development", "final"):
            while True:
                episode = f"episode-{counter}"
                counter += 1
                held = int(hashlib.sha256(episode.encode()).hexdigest()[:8], 16) / 2**32 < 0.2
                if ((partition == "train" and not held)
                        or (held and audit_partition(episode) == (partition == "final") and partition != "train")):
                    break
            image = np.full((16, 16), 64 + scenario, np.uint8)
            recorder.begin(image, scenario)
            recorder.episode_id = episode
            for i in range(4):
                recorder.append(0, image, -2 if i == 3 else 0, 0, i == 3, False)
    agent = ImaginedAgent(signature, config["world_model"])
    agent.real_decisions = 123
    checkpoint = tmp_path / "source.pt"
    agent.save(checkpoint)
    folder = tmp_path / ".local"
    folder.mkdir()
    (folder / "training-hold.json").write_text("{}")
    grant = {"experiment": EXPERIMENT, "active": True,
             "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             "maximum_updates_per_arm": 1, "seeds": [11]}
    (folder / "world-ablation-authorization.json").write_text(json.dumps(grant))
    monkeypatch.setattr("smg_rl.world_ablation.ARMS", {key: ARMS[key] for key in ("continued_baseline", "terminal_sampling")})
    original_validation = torch.distributions.Distribution._validate_args
    with pytest.raises(ValueError, match="unique known arms"):
        run_ablation(config, checkpoint, 1, [11], 4, ["terminal_sampling"])
    with pytest.raises(ValueError, match="unique known arms"):
        run_ablation(config, checkpoint, 1, [11], 4, ["continued_baseline", "unknown"])
    run = run_ablation(config, checkpoint, 1, [11], 4, ["continued_baseline"])
    assert torch.distributions.Distribution._validate_args == original_validation
    result = json.loads((run / "results.json").read_text())
    assert result["state"] == "completed"
    assert result["new_real_decisions"] == result["new_actor_updates"] == 0
    assert not set(result["development_episodes"]) & set(result["final_audit_episodes"])
    assert len(result["rows"]) == 1
    assert list(result["arms"]) == ["continued_baseline"]
    assert json.loads((run / "progress.json").read_text())["state"] == "completed"
    assert result["summary"]["continued_baseline"]["scenarios"]["spinning"]["reward_mae"]["paired_percent_change_mean"] == 0
    assert not json.loads((folder / "world-ablation-authorization.json").read_text())["active"]
    assert (folder / "training-hold.json").exists()


def test_policy_drift_excludes_burn_in_and_terminal_states():
    config = load_config("config/world-model.yaml")
    config["environment"]["image_size"] = 16
    settings = dict(deterministic_size=8, stochastic_size=2, embedding_size=4,
                    hidden_size=8, channels=2, ensemble_size=2)
    agent = ImaginedAgent(replay_signature(config), settings)
    batch = {"images": np.zeros((1, 4, 16, 16), np.uint8),
             "scenario": np.array([[1, 0]], np.float32), "actions": np.zeros((1, 3), np.int64),
             "is_first": np.array([[True, False, False, False]]),
             "mask": np.array([[0, 1, 1]], np.float32),
             "terminated": np.array([[False, False, True]])}
    original = policy_trace(agent, batch)
    assert original.shape[0] == 1
    equal = policy_drift(original, policy_trace(agent, batch))
    assert equal["source_to_candidate_kl_nats"] == 0
    assert equal["greedy_action_agreement"] == 1
    with torch.no_grad():
        agent.actor[-1].bias[0] += 10
    assert policy_drift(original, policy_trace(agent, batch))["source_to_candidate_kl_nats"] > 0
