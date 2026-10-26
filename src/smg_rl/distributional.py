"""Original model-free Rainbow C51 and IQN networks and distributional losses."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    def __init__(self, inputs, outputs, sigma=.5):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(outputs, inputs).uniform_(-1 / math.sqrt(inputs), 1 / math.sqrt(inputs)))
        self.bias_mu = nn.Parameter(torch.empty(outputs).uniform_(-1 / math.sqrt(inputs), 1 / math.sqrt(inputs)))
        self.weight_sigma = nn.Parameter(torch.full((outputs, inputs), sigma / math.sqrt(inputs)))
        self.bias_sigma = nn.Parameter(torch.full((outputs,), sigma / math.sqrt(outputs)))
        self.register_buffer('weight_noise', torch.zeros(outputs, inputs), persistent=False)
        self.register_buffer('bias_noise', torch.zeros(outputs), persistent=False)
        self.reset_noise()

    @torch.no_grad()
    def reset_noise(self):
        def factor(size):
            sample = torch.randn(size, device=self.weight_mu.device)
            return sample.sign() * sample.abs().sqrt()
        left, right = factor(self.weight_mu.shape[1]), factor(self.weight_mu.shape[0])
        self.weight_noise.copy_(right[:, None] * left[None])
        self.bias_noise.copy_(right)

    def forward(self, inputs):
        weight, bias = self.weight_mu, self.bias_mu
        if self.training:
            weight = weight + self.weight_sigma * self.weight_noise
            bias = bias + self.bias_sigma * self.bias_noise
        return F.linear(inputs, weight, bias)


class DistributionalQ(nn.Module):
    def __init__(self, signature, settings):
        super().__init__()
        self.algorithm = settings['algorithm']
        if self.algorithm not in ('rainbow', 'iqn'):
            raise ValueError('Choose rainbow or iqn explicitly.')
        self.actions, self.scenarios = len(signature['action_names']), len(signature['scenario_names'])
        self.atoms = int(settings.get('atoms', 51))
        hidden, channels = int(settings.get('hidden', 512)), int(settings.get('channels', 32))
        size, stack = signature['image_size'], settings['frame_stack']
        if min(hidden, channels, stack) < 1 or size < 40 or self.atoms < 2:
            raise ValueError('Invalid distributional image/network dimensions.')
        self.conv = nn.Sequential(nn.Conv2d(stack, channels, 8, 4), nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 4, 2), nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, 3, 1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            features = self.conv(torch.zeros(1, stack, size, size)).shape[-1]
        self.encode = nn.Sequential(nn.Linear(features, hidden), nn.ReLU())
        output = self.atoms if self.algorithm == 'rainbow' else 1
        self.value = nn.Sequential(NoisyLinear(hidden + self.scenarios, hidden), nn.ReLU(), NoisyLinear(hidden, output))
        self.advantage = nn.Sequential(NoisyLinear(hidden + self.scenarios, hidden), nn.ReLU(), NoisyLinear(hidden, self.actions * output))
        if self.algorithm == 'iqn':
            embedding = int(settings.get('cosine_embedding', 64))
            self.register_buffer('frequencies', torch.arange(1, embedding + 1).float() * math.pi)
            self.quantile_embedding = nn.Sequential(nn.Linear(embedding, hidden), nn.ReLU())
        else:
            low, high = settings['v_min'], settings['v_max']
            if not math.isfinite(low + high) or low >= high:
                raise ValueError('C51 support must have finite ordered endpoints.')
            self.register_buffer('support', torch.linspace(low, high, self.atoms))

    def reset_noise(self):
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, images, scenario, taus=None):
        features = self.encode(self.conv(images.float() / 255.))
        if self.algorithm == 'iqn':
            if taus is None or taus.ndim != 2 or taus.shape[0] != len(images):
                raise ValueError('IQN requires explicit per-sample quantiles.')
            embeddings = self.quantile_embedding(torch.cos(taus[:, :, None] * self.frequencies))
            features = features[:, None] * embeddings
            features = torch.cat((features, scenario[:, None].expand(-1, taus.shape[1], -1)), -1)
            value, advantage = self.value(features), self.advantage(features)
            return value + advantage - advantage.mean(-1, keepdim=True)  # B,N,A
        features = torch.cat((features, scenario), -1)
        value = self.value(features).reshape(-1, 1, self.atoms)
        advantage = self.advantage(features).reshape(-1, self.actions, self.atoms)
        return value + advantage - advantage.mean(1, keepdim=True)  # B,A,Z logits

    def expected(self, images, scenario, quantiles=32, deterministic=False):
        if self.algorithm == 'rainbow':
            return (self(images, scenario).softmax(-1) * self.support).sum(-1)
        if deterministic:
            taus = ((torch.arange(quantiles, device=images.device) + .5) / quantiles).expand(len(images), -1)
        else:
            taus = torch.rand(len(images), quantiles, device=images.device)
        return self(images, scenario, taus).mean(1)


def categorical_projection(probabilities, rewards, discounts, support):
    """Project Bellman distributions, preserving mass at exact integer bins."""
    atoms = len(support)
    locations = (rewards[:, None] + discounts[:, None] * support).clamp(support[0], support[-1])
    bins = ((locations - support[0]) / (support[-1] - support[0]) * (atoms - 1)).clamp(0, atoms - 1)
    lower, upper = bins.floor().long(), bins.ceil().long()
    result = torch.zeros_like(probabilities)
    result.scatter_add_(1, lower, probabilities * (upper - bins + (lower == upper)))
    result.scatter_add_(1, upper, probabilities * (bins - lower))
    return result


def quantile_huber(predictions, targets, taus, kappa=1.):
    """Per-sample IQN loss: sum prediction quantiles, mean target quantiles."""
    difference = targets[:, None] - predictions[:, :, None]
    absolute = difference.abs()
    huber = torch.where(absolute <= kappa, .5 * difference.square(), kappa * (absolute - .5 * kappa))
    weights = (taus[:, :, None] - (difference.detach() < 0).float()).abs()
    return (weights * huber / kappa).mean(-1).sum(-1)
