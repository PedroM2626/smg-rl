import json

import numpy as np
import pytest
import torch

from smg_rl.actions import ACTIONS
from smg_rl.config import load_config
from smg_rl.profiles import require_training_enabled
from smg_rl.replay import ReplayRecorder, SequenceReplay, load_chunk, replay_signature
from smg_rl.world_model import ImaginedAgent, lambda_returns, symexp, symlog
from smg_rl.world_training import quality_gate


def configuration(tmp_path):
    config = load_config("config/world-model.yaml")
    config["root"] = str(tmp_path)
    config["environment"]["image_size"] = 16
    config["paths"]["replay"] = str(tmp_path / "replay")
    config["paths"]["runs"] = str(tmp_path / "runs")
    config["paths"]["user"] = str(tmp_path / "user")
    return config


def small_agent(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    config = configuration(tmp_path)
    settings = dict(deterministic_size=16, stochastic_size=4, embedding_size=8,
                    hidden_size=16, channels=4, ensemble_size=3,
                    imagination_horizon=3, imagination_batch=4)
    return ImaginedAgent(replay_signature(config), settings), config


def synthetic_batch():
    rng = np.random.default_rng(13)
    return {"images": rng.integers(0, 256, (2, 5, 16, 16), np.uint8),
            "actions": np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int64),
            "rewards": np.array([[0.01, 0.1, 0, -2], [0, 0.1, 0, 0]], np.float32),
            "terminated": np.array([[False, False, False, True], [False] * 4]),
            "truncated": np.array([[False] * 4, [False, False, False, True]]),
            "is_first": np.array([[True, False, False, False, False]] * 2),
            "mask": np.array([[0, 1, 1, 1], [0, 1, 1, 1]], np.float32),
            "scenario": np.eye(2, dtype=np.float32)}


def test_model_gradients_imagination_and_safe_roundtrip(tmp_path):
    agent, config = small_agent(tmp_path)
    metrics, seeds, scenarios = agent.update_model(synthetic_batch())
    assert np.isfinite(list(metrics.values())).all()
    assert len(scenarios) == 5  # Real terminal posterior is never an imagination seed.
    before = {key: value.clone() for key, value in agent.world.state_dict().items()}
    imagined = agent.update_imagination(seeds, scenarios)
    assert imagined["imagined_transitions"] == 12
    assert agent.real_decisions == 0
    assert all(torch.equal(value, before[key]) for key, value in agent.world.state_dict().items())
    values = agent.world.validation(agent.tensors(synthetic_batch()), horizon=3)
    assert values["terminal_examples"] == 1
    assert values["validation_transitions"] == 6
    assert np.isfinite(list(values.values())).all()
    path = tmp_path / "world.pt"
    agent.save(path)
    restored = ImaginedAgent.load(path, replay_signature(config), agent.settings)
    assert restored.imagined_transitions == 12 and restored.model_updates == 1
    assert all(torch.equal(value, restored.world.state_dict()[key]) for key, value in agent.world.state_dict().items())
    with pytest.raises(FileExistsError):
        agent.save(path)
    path.write_bytes(path.read_bytes() + b"damage")
    with pytest.raises(ValueError, match="integrity"):
        ImaginedAgent.load(path, replay_signature(config), agent.settings)
    with pytest.raises(ValueError, match="incompatible"):
        ImaginedAgent.load(tmp_path / "ppo.zip", replay_signature(config), agent.settings)


def test_reward_support_and_latent_reset_dont_invent_terminal_prizes(tmp_path):
    agent, _ = small_agent(tmp_path)
    agent.world.reward_high.fill_(0.1)
    agent.world.reward_low.fill_(-2)
    with torch.no_grad():
        agent.world.reward[-1].bias.fill_(100)
        predictions = agent.world.predict_reward(torch.randn(3, agent.world.features))
    assert predictions.max() <= 0.100001
    embeddings = torch.randn(2, agent.world.embed)
    scenario = torch.eye(2)
    state = {key: torch.ones_like(value) * 100 for key, value in agent.world.initial(2).items()}
    after, _, _ = agent.world.observe(embeddings, torch.ones(2, dtype=torch.long), state,
                                       scenario, torch.ones(2, dtype=torch.bool), sample=False)
    assert not after["deter"].any()
    after_again, _, _ = agent.world.observe(embeddings, torch.zeros(2, dtype=torch.long),
                                            agent.world.initial(2), scenario,
                                            torch.ones(2, dtype=torch.bool), sample=False)
    assert torch.equal(after["stoch"], after_again["stoch"])


@pytest.mark.parametrize("balanced", [False, True])
def test_real_prior_prediction_objective_has_gradients_and_respects_padding(tmp_path, balanced):
    agent, _ = small_agent(tmp_path)
    agent.settings.update(balance_terminal_losses=balanced, prior_prediction_weight=0.5, reconstruction_weight=5,
                          terminal_reward_weight=5, open_loop_training_horizon=3)
    batch = synthetic_batch()
    loss, metrics, _, _ = agent.world.loss(agent.tensors(batch), sample=False)
    assert metrics["prior_prediction"] > 0
    assert metrics["open_loop_prediction"] > 0
    loss.backward()
    assert agent.world.prior_net[-1].weight.grad.abs().sum() > 0
    changed = {key: value.copy() for key, value in batch.items()}
    changed["rewards"][:, 0] = 1000  # Burn-in has zero supervised mask.
    repeated, _, _, _ = agent.world.loss(agent.tensors(changed), sample=False)
    assert torch.allclose(loss, repeated)
    metrics, _, _ = agent.update_model(batch)
    assert np.isfinite(list(metrics.values())).all()


def test_terminal_reward_weight_changes_only_supervised_terminal_targets(tmp_path):
    agent, _ = small_agent(tmp_path)
    batch = agent.tensors(synthetic_batch())
    _, original, _, _ = agent.world.loss(batch, sample=False)
    agent.settings["terminal_reward_weight"] = 5
    _, weighted, _, _ = agent.world.loss(batch, sample=False)
    assert weighted["reward"] > original["reward"]
    assert weighted["reconstruction"] == original["reconstruction"]
    batch["mask"][batch["terminated"]] = 0
    _, weighted, _, _ = agent.world.loss(batch, sample=False)
    agent.settings["terminal_reward_weight"] = 1
    _, original, _, _ = agent.world.loss(batch, sample=False)
    assert weighted["reward"] == original["reward"]


def test_prediction_head_repair_preserves_visual_dynamics_actor_and_real_counts(tmp_path):
    agent, _ = small_agent(tmp_path)
    agent.settings.update(balance_terminal_losses=True, terminal_reward_weight=20, terminal_weight=20)
    world = {k: v.clone() for k, v in agent.world.state_dict().items()
             if not k.startswith(('reward.', 'continuation.'))}
    actor = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    metrics = agent.update_prediction_heads(synthetic_batch())
    assert np.isfinite(list(metrics.values())).all()
    assert agent.model_updates == 1
    assert agent.real_decisions == agent.actor_updates == agent.imagined_transitions == 0
    assert all(torch.equal(v, agent.world.state_dict()[k]) for k, v in world.items())
    assert all(torch.equal(v, agent.actor.state_dict()[k]) for k, v in actor.items())
    assert all(p.grad is None for p in agent.world.encoder.parameters())
    empty = synthetic_batch()
    empty['mask'][:] = 0
    with pytest.raises(ValueError, match='supervised'):
        agent.update_prediction_heads(empty)


def test_decoder_repair_preserves_latent_dynamics_heads_and_gradient_flags(tmp_path):
    agent, _ = small_agent(tmp_path)
    agent.settings.update(prior_prediction_weight=.5, open_loop_training_horizon=3)
    protected = {k: v.clone() for k, v in agent.world.state_dict().items()
                 if not k.startswith(('decoder.', 'decode_input.'))}
    decoder = {k: v.clone() for k, v in agent.world.decoder.state_dict().items()}
    metrics = agent.update_visual_decoder(synthetic_batch())
    assert np.isfinite(list(metrics.values())).all()
    assert all(torch.equal(v, agent.world.state_dict()[k]) for k, v in protected.items())
    assert any(not torch.equal(v, agent.world.decoder.state_dict()[k]) for k, v in decoder.items())
    assert all(p.requires_grad for p in agent.world.parameters())
    assert agent.model_updates == 1
    assert agent.real_decisions == agent.actor_updates == agent.imagined_transitions == 0


def test_prior_repair_changes_forecast_without_changing_observed_features(tmp_path):
    agent, _ = small_agent(tmp_path)
    agent.settings.update(prior_prediction_weight=.5, open_loop_training_horizon=3)
    protected = {k: v.clone() for k, v in agent.world.state_dict().items()
                 if not k.startswith('prior_net.')}
    batch = synthetic_batch()
    torch.manual_seed(81)
    features = agent.world.posterior_features(agent.tensors(batch))
    metrics = agent.update_prior_dynamics(batch)
    torch.manual_seed(81)
    assert torch.equal(features, agent.world.posterior_features(agent.tensors(batch)))
    assert all(torch.equal(v, agent.world.state_dict()[k]) for k, v in protected.items())
    assert np.isfinite(list(metrics.values())).all()
    assert all(p.requires_grad for p in agent.world.parameters())
    assert agent.model_updates == 1 and agent.actor_updates == agent.real_decisions == 0


@pytest.mark.parametrize("balanced", [False, True])
def test_prior_prediction_cannot_observe_its_target_image(tmp_path, balanced):
    agent, _ = small_agent(tmp_path)
    agent.settings["prior_prediction_weight"] = 0.5
    agent.settings["balance_terminal_losses"] = balanced
    outputs = []
    handle = agent.world.decode_input.register_forward_hook(
        lambda module, inputs, result: outputs.append(result.detach().clone()))
    batch = synthetic_batch()
    agent.world.loss(agent.tensors(batch), sample=False)
    original = outputs[-1].reshape(2, 4, -1)[:, -1].clone()
    batch["images"][:, -1] = 255 - batch["images"][:, -1]
    agent.world.loss(agent.tensors(batch), sample=False)
    changed = outputs[-1].reshape(2, 4, -1)[:, -1]
    handle.remove()
    assert torch.equal(original, changed)


@pytest.mark.parametrize("balanced", [False, True])
def test_open_loop_training_cannot_observe_future_target_images(tmp_path, balanced):
    agent, _ = small_agent(tmp_path)
    agent.settings["open_loop_training_horizon"] = 3
    agent.settings["balance_terminal_losses"] = balanced
    outputs = []
    handle = agent.world.decode_input.register_forward_hook(
        lambda module, inputs, result: outputs.append(result.detach().clone()))
    batch = synthetic_batch()
    agent.world.loss(agent.tensors(batch), sample=False)
    original = outputs[-1].clone()
    batch["images"][:, -1] = 255 - batch["images"][:, -1]
    agent.world.loss(agent.tensors(batch), sample=False)
    handle.remove()
    assert torch.equal(original, outputs[-1])


def test_lambda_returns_terminate_and_symlog_roundtrips():
    rewards = torch.tensor([[1.0], [2.0]])
    returns = lambda_returns(rewards, torch.tensor([[0.0], [0.9]]),
                             torch.tensor([[100.0], [3.0]]), torch.tensor([3.0]))
    assert returns[:, 0].tolist() == pytest.approx([1, 4.7])
    inputs = torch.tensor([-20.0, -0.1, 0.0, 0.1, 20.0])
    assert torch.allclose(symexp(symlog(inputs)), inputs)


def test_replay_chunk_alignment_reward_labels_and_boundaries(tmp_path):
    config = configuration(tmp_path)
    signature = replay_signature(config)
    recorder = ReplayRecorder(tmp_path / "records", signature, chunk_steps=2, max_bytes=65536)
    recorder.begin(np.zeros((16, 16), np.uint8), 1)
    for t in range(3):
        recorder.append(t, np.full((16, 16), t + 1, np.uint8), 0.3, 0.1, False, t == 2)
    chunks = [load_chunk(p, signature) for p in sorted((tmp_path / "records").glob("*.npz"))]
    assert len(chunks) == 2
    assert chunks[0]["images"][:, 0, 0].tolist() == [0, 1, 2]
    assert chunks[1]["images"][:, 0, 0].tolist() == [2, 3]
    assert chunks[0]["actions"].tolist() == [0, 1]
    assert chunks[1]["actions"].tolist() == [2]
    assert chunks[0]["rewards"].tolist() == pytest.approx([0.2, 0.2])
    assert chunks[0]["is_first"].tolist() == [True, False, False]
    assert chunks[1]["is_first"].tolist() == [False, False]
    assert chunks[1]["truncated"].tolist() == [True]
    assert not chunks[1]["terminated"].any()
    assert chunks[0]["metadata"]["episode_id"] == chunks[1]["metadata"]["episode_id"]
    other = dict(signature, game_id="RMGE01")
    with pytest.raises(ValueError, match="signature"):
        load_chunk(next((tmp_path / "records").glob("*.npz")), other)


def test_replay_splits_episodes_not_chunks_and_masks_padding(tmp_path):
    signature = replay_signature(configuration(tmp_path))
    recorder = ReplayRecorder(tmp_path / "records", signature, chunk_steps=2, max_bytes=1000000)
    for episode in range(30):
        recorder.begin(np.zeros((16, 16), np.uint8), episode % 2)
        recorder.episode_id = f"episode-{episode}"
        for t in range(3):
            recorder.append(0, np.full((16, 16), t + 1, np.uint8), -0.1, 0, t == 2, False)
    replay = SequenceReplay(tmp_path / "records", signature)
    summary = replay.refresh()
    assert summary["transitions"] == 90
    training = {x["episode_id"] for x in replay.index if not x["validation"]}
    validation = {x["episode_id"] for x in replay.index if x["validation"]}
    assert training and validation and not training.intersection(validation)
    batch = replay.sample(8, 8, 2, validation=True, terminal_fraction=1)
    assert batch["images"].shape == (8, 11, 16, 16)
    assert batch["mask"].sum() == 8
    assert batch["terminated"][batch["mask"].astype(bool)].all()
    assert set(batch) == {"images", "actions", "rewards", "terminated", "truncated", "is_first", "mask", "scenario"}


def test_quality_gate_rejects_bad_model_and_missing_terminal_coverage():
    summary = {"transitions": 4096, "train_episodes": 10, "validation_episodes": 4,
               "train_scenarios": {"spinning": 300}, "validation_scenarios": {"spinning": 100}}
    metrics = {"posterior_image_mse": 0.001, "open_loop_image_mse": 0.001,
               "reward_mae": 0.1, "terminal_reward_mae": 0.1, "continuation_brier": 0.01, "terminal_brier": 0.1,
               "persistence_image_mse": 0.02, "terminal_examples": 3}
    assert quality_gate(summary, {"spinning": metrics}, {}, ["spinning"])["ready"]
    assert not quality_gate(summary, {"spinning": dict(metrics, terminal_examples=0)}, {}, ["spinning"])["ready"]
    assert not quality_gate(summary, {"spinning": dict(metrics, reward_mae=float("nan"))}, {}, ["spinning"])["ready"]
    assert not quality_gate(summary, {"spinning": metrics}, {}, ["spinning", "double_time"])["ready"]


@pytest.mark.parametrize("command", ["train", "train-parallel", "train-supervised", "train-world-model", "fit-world-model"])
def test_training_hold_precedes_any_environment_creation(tmp_path, monkeypatch, command):
    from smg_rl.cli import main

    config = configuration(tmp_path)
    (tmp_path / ".local").mkdir()
    (tmp_path / ".local/training-hold.json").write_text(json.dumps({"reason": "user hold"}))
    with pytest.raises(ValueError, match="on hold"):
        require_training_enabled(config)
    monkeypatch.setattr("smg_rl.cli.load_config", lambda _: config)
    monkeypatch.setattr("sys.argv", ["smg-rl", command])
    with pytest.raises(ValueError, match="on hold"):
        main()


def test_world_policy_uses_pixels_and_scenario_only(tmp_path):
    agent, _ = small_agent(tmp_path)
    obs = {"image": np.zeros((2, 4, 16, 16), np.uint8), "scenario": np.eye(2, dtype=np.float32)}
    torch.manual_seed(9)
    first, _, _ = agent.act(obs)
    torch.manual_seed(9)
    second, _, _ = agent.act({**obs, "position": np.ones((2, 3)) * 12345, "health": np.zeros(2)})
    assert np.array_equal(first, second)
    assert ((first >= 0) & (first < len(ACTIONS))).all()


def test_replay_rejects_non_real_source_and_bounds_its_owned_disk(tmp_path):
    signature = replay_signature(configuration(tmp_path))
    recorder = ReplayRecorder(tmp_path / "records", signature, chunk_steps=1, max_bytes=20000)
    recorder.begin(np.zeros((16, 16), np.uint8), 0)
    for index in range(20):
        recorder.append(0, np.full((16, 16), index, np.uint8), 0, 0, False, False)
    files = list((tmp_path / "records").glob("*.npz"))
    assert sum(p.stat().st_size for p in files) <= 20000
    path = files[0]
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["source"] = "imagined"
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="source"):
        load_chunk(path, signature)


def test_default_visual_geometry_forward_is_finite(tmp_path):
    torch.set_num_threads(1)
    config = load_config("config/world-model.yaml")
    agent = ImaginedAgent(replay_signature(config), config["world_model"])
    batch = synthetic_batch()
    batch["images"] = np.zeros((2, 5, 84, 84), np.uint8)
    with torch.no_grad():
        loss, _, _, _ = agent.world.loss(agent.tensors(batch), sample=False)
    assert torch.isfinite(loss)


def test_online_workflow_counts_only_real_decisions_and_resumes_with_mock_environment(tmp_path, monkeypatch):
    from smg_rl.world_training import train_world_model

    config = configuration(tmp_path)
    config["parallel"]["workers"] = 2
    config["world_model"].update(deterministic_size=16, stochastic_size=4, embedding_size=8,
                                  hidden_size=16, channels=4, collect_vector_steps=1, checkpoint_steps=2)
    closed = []

    class MockVector:
        def __init__(self, *args, **kwargs):
            pass

        def seed(self, seed):
            pass

        def reset(self):
            return {"image": np.zeros((2, 4, 16, 16), np.uint8), "scenario": np.eye(2, dtype=np.float32)}

        def step(self, actions):
            assert actions.shape == (2,)
            return self.reset(), np.zeros(2), np.ones(2, dtype=bool), [{"best_progress": 0.01}] * 2

        def close(self):
            closed.append(True)

    monkeypatch.setattr("smg_rl.world_training.MeasuredSubprocVecEnv", MockVector)
    first = train_world_model(config, steps=5)
    status = json.loads((first / "status.json").read_text())
    assert status["state"] == "completed" and status["real_decisions"] == 6
    assert status["imagined_transitions"] == 0 and status["model_updates"] == 0
    assert not status["diagnostics"]["imagination_gate"]["ready"]
    second = train_world_model(config, steps=2, resume=first / "world-final.pt")
    status = json.loads((second / "status.json").read_text())
    assert status["real_decisions"] == 8 and status["imagined_transitions"] == 0
    assert len(closed) == 2
