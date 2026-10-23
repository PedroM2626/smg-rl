"""Compact Dreamer-inspired visual RSSM and imagined discrete actor/critic.

This is an original Gaussian-RSSM research baseline, not DreamerV3. Real data,
model updates and imagined transitions always have separate counters.
"""

import json
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal, kl_divergence
from torch.nn import functional as F

from smg_rl.checkpoints import digest


def symlog(x):
    return x.sign() * x.abs().log1p()


def symexp(x):
    return x.sign() * x.abs().expm1()


def mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


def weighted_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def lambda_returns(rewards, discounts, next_values, bootstrap, mixing=0.95):
    """Time-major returns; padding/terminal masking belongs in discounts/weights."""
    result, value = [], bootstrap
    for t in reversed(range(len(rewards))):
        value = rewards[t] + discounts[t] * ((1 - mixing) * next_values[t] + mixing * value)
        result.append(value)
    return torch.stack(result[::-1])


class VisualWorldModel(nn.Module):
    def __init__(self, image_size, scenarios, actions, settings):
        super().__init__()
        self.size, self.scenarios, self.actions = image_size, scenarios, actions
        self.deter = int(settings.get("deterministic_size", 256))
        self.stoch = int(settings.get("stochastic_size", 32))
        self.embed = int(settings.get("embedding_size", 128))
        hidden = int(settings.get("hidden_size", 256))
        channels = int(settings.get("channels", 16))
        if min(self.deter, self.stoch, self.embed, hidden, channels, scenarios, actions) < 1 or image_size < 16:
            raise ValueError("World-model dimensions must be positive and images at least 16 pixels.")
        self.features = self.deter + self.stoch + scenarios
        self.encoder = nn.Sequential(nn.Conv2d(1, channels, 4, 2, 1), nn.SiLU(),
                                     nn.Conv2d(channels, channels * 2, 4, 2, 1), nn.SiLU(),
                                     nn.Conv2d(channels * 2, channels * 4, 4, 2, 1), nn.SiLU(),
                                     nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
                                     nn.Linear(channels * 4 * 9, self.embed), nn.LayerNorm(self.embed))
        self.gru = nn.GRUCell(self.stoch + actions + scenarios, self.deter)
        self.prior_net = mlp(self.deter + scenarios, 2 * self.stoch, hidden)
        self.posterior_net = mlp(self.deter + self.embed + scenarios, 2 * self.stoch, hidden)
        self.decode_input = nn.Linear(self.features, channels * 4 * 6 * 6)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1), nn.SiLU(),
                                     nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1), nn.SiLU(),
                                     nn.ConvTranspose2d(channels, 1, 4, 2, 1))
        self.reward = mlp(self.features, 1, hidden)
        self.continuation = mlp(self.features, 1, hidden)
        ensemble_size = int(settings.get("ensemble_size", 5))
        if ensemble_size < 2:
            raise ValueError("Disagreement exploration requires at least two predictors.")
        self.ensemble = nn.ModuleList([mlp(self.features + actions, self.embed, hidden)
                                       for _ in range(ensemble_size)])
        self.settings = settings
        self.channels = channels
        self.register_buffer("reward_low", torch.tensor(0.0))
        self.register_buffer("reward_high", torch.tensor(0.0))

    def initial(self, batch):
        device = next(self.parameters()).device
        return {"deter": torch.zeros(batch, self.deter, device=device),
                "stoch": torch.zeros(batch, self.stoch, device=device)}

    def feature(self, state, scenario):
        return torch.cat([state["deter"], state["stoch"], scenario], -1)

    def distribution(self, parameters):
        mean, raw_std = parameters.chunk(2, -1)
        return Independent(Normal(mean, 0.1 + F.softplus(raw_std)), 1)

    def encode(self, images):
        return self.encoder(images.float() / 255)

    def observe(self, embedding, previous_action, state, scenario, is_first, sample=True):
        inputs = torch.cat([state["stoch"], F.one_hot(previous_action, self.actions).float(), scenario], -1)
        deter = self.gru(inputs, state["deter"])
        deter = torch.where(is_first[:, None], torch.zeros_like(deter), deter)
        prior = self.distribution(self.prior_net(torch.cat([deter, scenario], -1)))
        posterior = self.distribution(self.posterior_net(torch.cat([deter, embedding, scenario], -1)))
        return {"deter": deter, "stoch": posterior.rsample() if sample else posterior.mean}, prior, posterior

    def imagine_step(self, state, action, scenario, sample=True):
        inputs = torch.cat([state["stoch"], F.one_hot(action, self.actions).float(), scenario], -1)
        deter = self.gru(inputs, state["deter"])
        prior = self.distribution(self.prior_net(torch.cat([deter, scenario], -1)))
        return {"deter": deter, "stoch": prior.rsample() if sample else prior.mean}

    def decode(self, features):
        x = self.decode_input(features).reshape(-1, self.channels * 4, 6, 6)
        return torch.sigmoid(F.interpolate(self.decoder(x), (self.size, self.size), mode="bilinear", align_corners=False))

    def disagreement(self, features, actions):
        inputs = torch.cat([features, F.one_hot(actions, self.actions).float()], -1)
        predictions = torch.stack([head(inputs) for head in self.ensemble])
        return predictions.std(0, unbiased=False).mean(-1)

    def predict_reward(self, features):
        prediction = symexp(self.reward(features).squeeze(-1).clamp(-5, 5))
        # The model cannot invent a star-sized reward before seeing one in real data.
        return prediction.clamp(min=self.reward_low, max=self.reward_high)

    def loss(self, batch, sample=True, auxiliary=True):
        images, actions, scenario, mask = (batch[k] for k in ("images", "actions", "scenario", "mask"))
        b, frames = images.shape[:2]
        embeddings = self.encode(images.reshape(-1, 1, self.size, self.size)).reshape(b, frames, -1)
        state = self.initial(b)
        state, _, _ = self.observe(embeddings[:, 0], torch.zeros(b, dtype=torch.long, device=images.device),
                                    state, scenario, batch["is_first"][:, 0], sample)
        prior_features, context_states, predicted_states, states, dynamics, representation = [], [], [], [], [], []
        for t in range(frames - 1):
            context_states.append(state)
            prior_features.append(self.feature(state, scenario))
            state, prior, posterior = self.observe(embeddings[:, t + 1], actions[:, t], state,
                                                   scenario, batch["is_first"][:, t + 1], sample)
            if auxiliary and self.settings.get("prior_prediction_weight", 0) > 0:
                predicted_states.append({"deter": state["deter"],
                                         "stoch": prior.rsample() if sample else prior.mean})
            detached_posterior = Independent(Normal(posterior.base_dist.loc.detach(),
                                                    posterior.base_dist.scale.detach()), 1)
            detached_prior = Independent(Normal(prior.base_dist.loc.detach(), prior.base_dist.scale.detach()), 1)
            free = float(self.settings.get("free_nats", 1.0))
            dynamics.append(kl_divergence(detached_posterior, prior).clamp_min(free))
            representation.append(kl_divergence(posterior, detached_prior).clamp_min(free))
            states.append(state)
        features = torch.stack([self.feature(s, scenario) for s in states], 1)
        shape = features.shape
        flat = features.reshape(-1, self.features)
        reconstruction = self.decode(flat).reshape(b, frames - 1, 1, self.size, self.size)
        target = images[:, 1:].float()[:, :, None] / 255
        reconstruction_loss = (reconstruction - target).square().mean((2, 3, 4))
        reward_prediction = self.reward(flat).reshape(b, -1)
        reward_loss = (reward_prediction - symlog(batch["rewards"])).square()
        terminal = batch["terminated"].float()
        reward_weights = 1 + terminal * (float(self.settings.get("terminal_reward_weight", 1)) - 1)
        reward_loss = reward_loss * reward_weights
        continuation_logits = self.continuation(flat).reshape(b, -1)
        continuation_loss = F.binary_cross_entropy_with_logits(continuation_logits, 1 - terminal, reduction="none")
        continuation_loss *= 1 + terminal * (float(self.settings.get("terminal_weight", 5)) - 1)
        ensemble_input = torch.cat([torch.stack(prior_features, 1).detach(), F.one_hot(actions, self.actions).float()], -1)
        ensemble_loss = 0
        for head in self.ensemble:
            error = (head(ensemble_input) - embeddings[:, 1:].detach()).square().mean(-1)
            bootstrap = (torch.rand_like(mask) < 0.8).float() if sample else torch.ones_like(mask)
            ensemble_loss = ensemble_loss + weighted_mean(error, mask * bootstrap)
        ensemble_loss = ensemble_loss / len(self.ensemble)
        terms = {"reconstruction": weighted_mean(reconstruction_loss, mask),
                 "reward": weighted_mean(reward_loss, mask),
                 "continuation": weighted_mean(continuation_loss, mask),
                 "dynamics_kl": weighted_mean(torch.stack(dynamics, 1), mask),
                 "representation_kl": weighted_mean(torch.stack(representation, 1), mask),
                 "ensemble": ensemble_loss}
        image_weight = float(self.settings.get("reconstruction_weight", 1))
        loss = (image_weight * terms["reconstruction"] + terms["reward"] + terms["continuation"]
                + 0.5 * terms["dynamics_kl"] + 0.1 * terms["representation_kl"] + terms["ensemble"])
        if predicted_states:
            # Train the prior used in imagination against real next observations.
            # Posterior-only reconstruction otherwise permits a teacher-forcing gap.
            predicted = torch.stack([self.feature(s, scenario) for s in predicted_states], 1)
            predicted = predicted.reshape(-1, self.features)
            prior_images = self.decode(predicted).reshape_as(reconstruction)
            prior_image_loss = (prior_images - target).square().mean((2, 3, 4))
            prior_reward_loss = (self.reward(predicted).reshape(b, -1) - symlog(batch["rewards"])).square()
            prior_continuation = F.binary_cross_entropy_with_logits(
                self.continuation(predicted).reshape(b, -1), 1 - terminal, reduction="none")
            prior_continuation *= 1 + terminal * (float(self.settings.get("terminal_weight", 5)) - 1)
            terms["prior_prediction"] = weighted_mean(
                image_weight * prior_image_loss + reward_weights * prior_reward_loss + prior_continuation, mask)
            loss = loss + float(self.settings["prior_prediction_weight"]) * terms["prior_prediction"]
        horizon = int(self.settings.get("open_loop_training_horizon", 0)) if auxiliary else 0
        if horizon:
            begin = mask.argmax(1)
            indices = torch.arange(frames - 1, device=images.device)[None]
            endings = batch["terminated"] & mask.bool()
            last = (indices * endings).amax(1)
            begin = torch.where(endings.any(1), torch.maximum(begin, last - horizon + 1), begin)
            rows = torch.arange(b, device=images.device)
            predicted_state = {key: torch.stack([s[key] for s in context_states], 1)[rows, begin]
                               for key in state}
            errors, weights = [], []
            for offset in range(min(horizon, frames - 1)):
                position = (begin + offset).clamp_max(frames - 2)
                valid = (begin + offset < frames - 1).float() * mask[rows, position]
                predicted_state = self.imagine_step(predicted_state, actions[rows, position], scenario, sample)
                predicted = self.feature(predicted_state, scenario)
                expected = images[rows, position + 1].float()[:, None] / 255
                image_error = (self.decode(predicted) - expected).square().mean((1, 2, 3))
                reward_error = (self.reward(predicted).squeeze(-1) - symlog(batch["rewards"][rows, position])).square()
                continuation_error = F.binary_cross_entropy_with_logits(
                    self.continuation(predicted).squeeze(-1), 1 - terminal[rows, position], reduction="none")
                continuation_error *= 1 + terminal[rows, position] * (float(self.settings.get("terminal_weight", 5)) - 1)
                errors.append(image_weight * image_error + reward_weights[rows, position] * reward_error
                              + continuation_error)
                weights.append(valid)
            terms["open_loop_prediction"] = weighted_mean(torch.stack(errors, 1), torch.stack(weights, 1))
            loss = loss + float(self.settings.get("open_loop_training_weight", 0.5)) * terms["open_loop_prediction"]
        seeds = {key: torch.stack([s[key] for s in states], 1).detach() for key in state}
        valid = mask.bool() & ~batch["terminated"]
        seeds = {key: value[valid] for key, value in seeds.items()}
        scenarios = scenario[:, None].expand(-1, shape[1], -1)[valid].detach()
        metrics = {key: float(value.detach()) for key, value in terms.items()}
        metrics.update(loss=float(loss.detach()), posterior_image_mse=float(weighted_mean(reconstruction_loss, mask).detach()))
        with torch.no_grad():
            terminal_mask = mask * terminal
            brier = (continuation_logits.sigmoid() - (1 - terminal)).square()
            metrics.update(reward_mae=float(weighted_mean((self.predict_reward(flat).reshape(b, -1) - batch["rewards"]).abs(), mask)),
                           terminal_reward_mae=float(weighted_mean((self.predict_reward(flat).reshape(b, -1) - batch["rewards"]).abs(), terminal_mask)),
                           continuation_brier=float(weighted_mean(brier, mask)),
                           terminal_brier=float(weighted_mean(brier, terminal_mask)),
                           terminal_examples=int(terminal_mask.sum()), validation_transitions=int(mask.sum()))
        return loss, metrics, seeds, scenarios

    @torch.no_grad()
    def validation(self, batch, horizon=8, stochastic=True):
        # Validate sampled latent dynamics as used by imagination, not only a
        # potentially over-optimistic mean-latent reconstruction.
        _, metrics, _, _ = self.loss(batch, sample=stochastic, auxiliary=False)
        images, scenario = batch["images"], batch["scenario"]
        b, frames = images.shape[:2]
        embeddings = self.encode(images.reshape(-1, 1, self.size, self.size)).reshape(b, frames, -1)
        state = self.initial(b)
        zero = torch.zeros(b, dtype=torch.long, device=images.device)
        state, _, _ = self.observe(embeddings[:, 0], zero, state, scenario, batch["is_first"][:, 0], stochastic)
        image_errors, persistence_errors, reward_errors, weights = [], [], [], []
        # Each sequence starts open-loop prediction at its first supervised transition.
        begin = batch["mask"].argmax(1)
        indices = torch.arange(frames - 1, device=images.device)[None]
        terminal = batch["terminated"] & batch["mask"].bool()
        terminal_index = (indices * terminal).amax(1)
        begin = torch.where(terminal.any(1), torch.maximum(begin, terminal_index - horizon + 1), begin)
        for t in range(min(frames - 1, int((begin + horizon).max()))):
            active = (t >= begin) & (t < begin + horizon) & batch["mask"][:, t].bool()
            predicted = self.imagine_step(state, batch["actions"][:, t], scenario, stochastic)
            features = self.feature(predicted, scenario)
            target = images[:, t + 1].float()[:, None] / 255
            image_errors.append((self.decode(features) - target).square().mean((1, 2, 3)))
            reference = images[torch.arange(b, device=images.device), begin].float()[:, None] / 255
            persistence_errors.append((reference - target).square().mean((1, 2, 3)))
            reward_errors.append((self.predict_reward(features) - batch["rewards"][:, t]).abs())
            weights.append(active.float())
            observed, _, _ = self.observe(embeddings[:, t + 1], batch["actions"][:, t], state,
                                          scenario, batch["is_first"][:, t + 1], stochastic)
            state = {key: torch.where((t >= begin)[:, None], predicted[key], observed[key]) for key in state}
        mask = torch.stack(weights, 1)
        metrics.update(open_loop_image_mse=float(weighted_mean(torch.stack(image_errors, 1), mask)),
                       persistence_image_mse=float(weighted_mean(torch.stack(persistence_errors, 1), mask)),
                       open_loop_reward_mae=float(weighted_mean(torch.stack(reward_errors, 1), mask)))
        return metrics


class ImaginedAgent:
    def __init__(self, signature, settings, device="cpu"):
        self.signature, self.settings, self.device = signature, settings, torch.device(device)
        self.world = VisualWorldModel(signature["image_size"], len(signature["scenario_names"]),
                                      len(signature["action_names"]), settings).to(self.device)
        hidden = int(settings.get("hidden_size", 256))
        self.actor = mlp(self.world.features, self.world.actions, hidden).to(self.device)
        self.critic = mlp(self.world.features, 1, hidden).to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.model_optimizer = torch.optim.Adam(self.world.parameters(), lr=settings.get("model_lr", 3e-4))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=settings.get("actor_lr", 8e-5))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=settings.get("critic_lr", 8e-5))
        self.model_updates = self.actor_updates = self.imagined_transitions = self.real_decisions = 0
        self.replay_rng_state = None

    def tensors(self, batch):
        return {key: torch.as_tensor(value, device=self.device) for key, value in batch.items()}

    def update_model(self, batch):
        batch = self.tensors(batch)
        valid = batch["mask"].bool()
        if not valid.any():
            raise ValueError("A world-model batch needs supervised real transitions.")
        with torch.no_grad():
            self.world.reward_low.copy_(torch.minimum(self.world.reward_low, batch["rewards"][valid].min()))
            self.world.reward_high.copy_(torch.maximum(self.world.reward_high, batch["rewards"][valid].max()))
        self.model_optimizer.zero_grad(set_to_none=True)
        loss, metrics, seeds, scenarios = self.world.loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite world-model loss.")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.world.parameters(), 100, error_if_nonfinite=True)
        self.model_optimizer.step()
        self.model_updates += 1
        metrics["model_gradient_norm"] = float(norm)
        return metrics, seeds, scenarios

    def update_imagination(self, seeds, scenarios):
        maximum = int(self.settings.get("imagination_batch", 128))
        selection = torch.randperm(len(scenarios), device=self.device)[:maximum]
        state = {key: value[selection].detach() for key, value in seeds.items()}
        scenario = scenarios[selection].detach()
        if not len(scenario):
            raise ValueError("Imagination requires real posterior seed states.")
        horizon = int(self.settings.get("imagination_horizon", 8))
        features, actions, rewards, discounts = [], [], [], []
        with torch.no_grad():
            for _ in range(horizon):
                feature = self.world.feature(state, scenario)
                action = Categorical(logits=self.actor(feature)).sample()
                bonus = self.world.disagreement(feature, action).clamp(0, 1)
                state = self.world.imagine_step(state, action, scenario)
                next_feature = self.world.feature(state, scenario)
                reward = self.world.predict_reward(next_feature) + self.settings.get("disagreement_scale", 0.01) * bonus
                discount = self.settings.get("gamma", 0.997) * self.world.continuation(next_feature).squeeze(-1).sigmoid()
                features.append(feature)
                actions.append(action)
                rewards.append(reward)
                discounts.append(discount)
            features, actions, rewards, discounts = map(torch.stack, (features, actions, rewards, discounts))
            last = self.world.feature(state, scenario)
            values = self.target_critic(features).squeeze(-1)
            bootstrap = self.target_critic(last).squeeze(-1)
            next_values = torch.cat([values[1:], bootstrap[None]])
            returns = lambda_returns(rewards, discounts, next_values, bootstrap,
                                     self.settings.get("lambda", 0.95)).clamp(-100, 100)
            weights = torch.cat([torch.ones_like(discounts[:1]), discounts[:-1]]).cumprod(0)
            advantage = returns - values
            advantage = (advantage / advantage.std(unbiased=False).clamp_min(1)).clamp(-10, 10)
        distribution = Categorical(logits=self.actor(features.detach()))
        actor_loss = -weighted_mean(distribution.log_prob(actions) * advantage
                                    + self.settings.get("entropy", 0.003) * distribution.entropy(), weights)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100, error_if_nonfinite=True)
        self.actor_optimizer.step()
        predictions = self.critic(features.detach()).squeeze(-1)
        critic_loss = weighted_mean(F.smooth_l1_loss(predictions, returns, reduction="none"), weights)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100, error_if_nonfinite=True)
        self.critic_optimizer.step()
        with torch.no_grad():
            for target, current in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.lerp_(current, 0.01)
        self.actor_updates += 1
        self.imagined_transitions += horizon * len(scenario)
        return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                "imagined_reward": float(rewards.mean()), "imagined_transitions": self.imagined_transitions}

    @torch.no_grad()
    def act(self, observation, state=None, previous_actions=None, episode_start=None, epsilon=0.05):
        pixels = np.asarray(observation["image"])
        if pixels.ndim != 4 or pixels.dtype != np.uint8:
            raise ValueError("World-model acting requires batched uint8 visual observations.")
        b = len(pixels)
        state = self.world.initial(b) if state is None else state
        previous = torch.zeros(b, dtype=torch.long, device=self.device) if previous_actions is None else previous_actions
        first = torch.ones(b, dtype=torch.bool, device=self.device) if episode_start is None else torch.as_tensor(episode_start, device=self.device, dtype=torch.bool)
        scenario = torch.as_tensor(observation["scenario"], device=self.device)
        embedding = self.world.encode(torch.as_tensor(pixels[:, -1:], device=self.device))
        state, _, _ = self.world.observe(embedding, previous, state, scenario, first)
        actions = Categorical(logits=self.actor(self.world.feature(state, scenario))).sample()
        random_actions = torch.randint(self.world.actions, (b,), device=self.device)
        actions = torch.where(torch.rand(b, device=self.device) < epsilon, random_actions, actions)
        return actions.cpu().numpy(), state, actions

    def save(self, path):
        path = Path(path)
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError("World-model checkpoints are immutable.")
        pending = path.with_name("." + uuid.uuid4().hex + ".pt")
        marker = path.with_suffix(".incomplete")
        marker.touch(exist_ok=False)
        modules = ("world", "actor", "critic", "target_critic", "model_optimizer", "actor_optimizer", "critic_optimizer")
        try:
            torch.save({"modules": {key: getattr(self, key).state_dict() for key in modules},
                        "counters": {key: getattr(self, key) for key in ("real_decisions", "model_updates", "actor_updates", "imagined_transitions")},
                        "torch_rng": torch.get_rng_state(),
                        "replay_rng_state": self.replay_rng_state,
                        "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []}, pending)
            pending.replace(path)
            metadata = {"schema": 1, "signature": self.signature, "settings": self.settings, "sha256": digest(path)}
            manifest = path.with_suffix(".json")
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            temporary.replace(manifest)
            marker.unlink()
        finally:
            pending.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, signature, settings, device="cpu"):
        path = Path(path)
        if path.suffix != ".pt":
            raise ValueError("A world-model .pt checkpoint is required; PPO ZIP checkpoints are incompatible.")
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (path.with_suffix(".incomplete").exists() or metadata.get("schema") != 1
                or metadata["signature"] != signature or metadata["settings"] != settings
                or metadata["sha256"] != digest(path)):
            raise ValueError("World-model checkpoint signature, settings or integrity mismatch.")
        agent = cls(signature, settings, device)
        data = torch.load(path, map_location=agent.device, weights_only=True)
        for key, state in data["modules"].items():
            getattr(agent, key).load_state_dict(state)
        for key, value in data["counters"].items():
            setattr(agent, key, int(value))
        torch.set_rng_state(data["torch_rng"].cpu())
        agent.replay_rng_state = data.get("replay_rng_state")
        if data["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in data["cuda_rng"]])
        return agent
