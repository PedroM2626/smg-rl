"""NGU-inspired episodic kNN novelty multiplied by per-scenario RND novelty.

This is not a reproduction of NGU's distributed Q-learning/meta-controller.
An inverse-dynamics encoder learns controllable visual features. A frozen copy
is used throughout each episode so stored kNN embeddings stay comparable.
"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Moments:
    def __init__(self):
        self.count, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self):
        return self.m2 / max(self.count - 1, 1)


class Encoder(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512, dimension),
            nn.Tanh(),
        )

    def forward(self, image):
        return self.network(image)


class EpisodicMemory:
    def __init__(self, neighbors=10, capacity=4000):
        self.k, self.capacity = neighbors, capacity
        self.reset()

    def reset(self):
        self.values = None
        self.count = 0
        self.cursor = 0
        self.distance_scale = None

    def reward(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        if self.values is None:
            self.values = np.empty((self.capacity, embedding.size), dtype=np.float32)
        if not self.count:
            novelty = 1.0
        else:
            distances = np.square(self.values[:self.count] - embedding).sum(axis=1)
            nearest = np.partition(distances, min(self.k, len(distances)) - 1)[: self.k]
            mean_distance = float(nearest.mean())
            if self.distance_scale is None:
                self.distance_scale = max(mean_distance, 1e-6)
            else:
                self.distance_scale = 0.99 * self.distance_scale + 0.01 * mean_distance
            normalized = np.maximum(nearest / max(self.distance_scale, 1e-6) - 0.008, 0)
            kernel = 0.0001 / (normalized + 0.0001)
            similarity = float(np.sqrt(kernel.sum()) + 0.001)
            novelty = 0.0 if similarity > 3.0 else min(1.0, 1.0 / similarity)
        self.values[self.cursor] = embedding
        self.cursor = (self.cursor + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        return novelty


class IntrinsicReward:
    def __init__(self, action_count, config):
        dimension = config["embedding_size"]
        self.device = torch.device(config.get("device", "cpu"))
        self.update_every = int(config.get("update_every", 1))
        if self.update_every < 1:
            raise ValueError("Curiosity update_every must be positive.")
        self.pending = []
        self.updates = 0
        self.encoder = Encoder(dimension)
        self.inverse = nn.Linear(2 * dimension, action_count)
        self.episodic_encoder = deepcopy(self.encoder).eval()
        self.target = Encoder(dimension).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Encoder(dimension)
        for module in (self.encoder, self.inverse, self.episodic_encoder, self.target, self.predictor):
            module.to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.inverse.parameters())
            + list(self.predictor.parameters()),
            lr=1e-4,
        )
        self.memory = EpisodicMemory(config["neighbors"], config["capacity"])
        self.error_stats = Moments()
        self.rnd_clip = config["rnd_clip"]

    def tensor(self, image):
        return torch.from_numpy(np.asarray(image).copy()).to(self.device).float()[None, None] / 255.0

    def reset(self, image):
        self.flush_pending()
        self.episodic_encoder.load_state_dict(self.encoder.state_dict())
        self.memory.reset()
        with torch.no_grad():
            self.memory.reward(self.episodic_encoder(self.tensor(image))[0].cpu().numpy())

    def step(self, before, after, action):
        if self.update_every > 1:
            return self._batched_step(before, after, action)
        before, after = self.tensor(before), self.tensor(after)
        with torch.no_grad():
            embedding = self.episodic_encoder(after)[0].cpu().numpy()
            target = self.target(after)
        prediction = self.predictor(after)
        rnd_loss = F.mse_loss(prediction, target)
        error = float(rnd_loss.detach())
        # Normalize using past errors, before updating the estimator.
        scale = max(self.error_stats.variance**0.5, 1e-4)
        lifelong = float(np.clip(1 + (error - self.error_stats.mean) / scale, 1, self.rnd_clip))
        episodic = self.memory.reward(embedding)
        self.error_stats.update(error)
        inverse_logits = self.inverse(torch.cat((self.encoder(before), self.encoder(after)), dim=1))
        inverse_loss = F.cross_entropy(inverse_logits, torch.tensor([action], device=self.device))
        self.optimizer.zero_grad()
        (rnd_loss + inverse_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.inverse.parameters())
            + list(self.predictor.parameters()),
            1.0,
        )
        self.optimizer.step()
        self.updates += 1
        return episodic * lifelong, {
            "episodic_novelty": episodic,
            "rnd_error": error,
            "lifelong_novelty": lifelong,
            "curiosity_update": 1,
            "curiosity_batch_size": 1,
        }

    def _batched_step(self, before, after, action):
        with torch.no_grad():
            tensor = self.tensor(after)
            embedding = self.episodic_encoder(tensor)[0].cpu().numpy()
            target = self.target(tensor)
            error = float(F.mse_loss(self.predictor(tensor), target))
        scale = max(self.error_stats.variance**0.5, 1e-4)
        lifelong = float(np.clip(1 + (error - self.error_stats.mean) / scale, 1, self.rnd_clip))
        episodic = self.memory.reward(embedding)
        self.error_stats.update(error)
        self.pending.append((torch.from_numpy(np.asarray(before).copy()),
                             torch.from_numpy(np.asarray(after).copy()), int(action),
                             target[0].detach().cpu()))
        count = len(self.pending) if len(self.pending) >= self.update_every else 0
        if count:
            self.flush_pending()
        return episodic * lifelong, {
            "episodic_novelty": episodic, "rnd_error": error,
            "lifelong_novelty": lifelong, "curiosity_update": int(bool(count)),
            "curiosity_batch_size": count,
        }

    def flush_pending(self):
        if not self.pending:
            return
        before, after, actions, targets = zip(*self.pending)
        before = torch.stack(before).to(self.device).float()[:, None] / 255.0
        after = torch.stack(after).to(self.device).float()[:, None] / 255.0
        targets = torch.stack(targets).to(self.device)
        labels = torch.tensor(actions, device=self.device)
        rnd_loss = F.mse_loss(self.predictor(after), targets)
        inverse_loss = F.cross_entropy(
            self.inverse(torch.cat((self.encoder(before), self.encoder(after)), dim=1)), labels
        )
        self.optimizer.zero_grad()
        (rnd_loss + inverse_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.inverse.parameters())
            + list(self.predictor.parameters()), 1.0,
        )
        self.optimizer.step()
        self.updates += 1
        self.pending.clear()

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "inverse": self.inverse.state_dict(),
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "moments": vars(self.error_stats).copy(),
            "pending": self.pending.copy(),
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        for key in ("encoder", "inverse", "target", "predictor", "optimizer"):
            getattr(self, key).load_state_dict(state[key])
        vars(self.error_stats).update(state["moments"])
        self.pending = state.get("pending", []).copy()
        self.updates = state.get("updates", 0)
