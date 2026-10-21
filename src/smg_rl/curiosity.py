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
        self.values = []
        self.distance_scale = None

    def reward(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        if not self.values:
            novelty = 1.0
        else:
            distances = np.square(np.stack(self.values) - embedding).sum(axis=1)
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
        if len(self.values) == self.capacity:
            self.values.pop(0)
        self.values.append(embedding.copy())
        return novelty


class IntrinsicReward:
    def __init__(self, action_count, config):
        dimension = config["embedding_size"]
        self.encoder = Encoder(dimension)
        self.inverse = nn.Linear(2 * dimension, action_count)
        self.episodic_encoder = deepcopy(self.encoder).eval()
        self.target = Encoder(dimension).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Encoder(dimension)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.inverse.parameters())
            + list(self.predictor.parameters()),
            lr=1e-4,
        )
        self.memory = EpisodicMemory(config["neighbors"], config["capacity"])
        self.error_stats = Moments()
        self.rnd_clip = config["rnd_clip"]

    @staticmethod
    def tensor(image):
        return torch.from_numpy(np.asarray(image).copy()).float()[None, None] / 255.0

    def reset(self, image):
        self.episodic_encoder.load_state_dict(self.encoder.state_dict())
        self.memory.reset()
        with torch.no_grad():
            self.memory.reward(self.episodic_encoder(self.tensor(image))[0].numpy())

    def step(self, before, after, action):
        before, after = self.tensor(before), self.tensor(after)
        with torch.no_grad():
            embedding = self.episodic_encoder(after)[0].numpy()
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
        inverse_loss = F.cross_entropy(inverse_logits, torch.tensor([action]))
        self.optimizer.zero_grad()
        (rnd_loss + inverse_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.inverse.parameters())
            + list(self.predictor.parameters()),
            1.0,
        )
        self.optimizer.step()
        return episodic * lifelong, {
            "episodic_novelty": episodic,
            "rnd_error": error,
            "lifelong_novelty": lifelong,
        }

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "inverse": self.inverse.state_dict(),
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "moments": vars(self.error_stats).copy(),
        }

    def load_state_dict(self, state):
        for key in ("encoder", "inverse", "target", "predictor", "optimizer"):
            getattr(self, key).load_state_dict(state[key])
        vars(self.error_stats).update(state["moments"])
