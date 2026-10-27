"""Genetic evolution of every deterministic parameter on the visual action path."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, vmap


class VisualActionPath(nn.Module):
    def __init__(self, network, quantiles):
        super().__init__()
        self.conv, self.encode = network.conv, network.encode
        self.quantile_embedding, self.advantage = network.quantile_embedding, network.advantage
        self.register_buffer('frequencies', network.frequencies)
        self.quantiles = quantiles

    def forward(self, images, scenario):
        features = self.encode(self.conv(images.float() / 255.))
        taus = (torch.arange(self.quantiles, device=images.device) + .5) / self.quantiles
        embedding = self.quantile_embedding(torch.cos(taus[:, None] * self.frequencies))
        features = features[:, None] * embedding[None]
        features = torch.cat((features, scenario[:, None].expand(-1, self.quantiles, -1)), -1)
        return self.advantage(features).mean(1)


class GeneticVisualPolicy:
    """No backpropagation: CNN/projection/action parameters are independent genes."""

    def __init__(self, agent):
        if agent.online.algorithm != 'iqn':
            raise ValueError('Visual genetic transfer currently requires an IQN source.')
        self.network = agent.online.eval().requires_grad_(False)
        self.device = agent.device
        self.model = VisualActionPath(self.network, agent.settings['quantiles']).eval()
        self.actions, self.hidden = self.network.advantage[-1].weight_mu.shape
        self.layout, parts, scales, offset = [], [], [], 0
        for name, parameter in self.model.named_parameters():
            if name.endswith(('weight_sigma', 'bias_sigma')):
                continue  # Evaluation-mode noise amplitudes cannot affect action selection.
            part = parameter.detach().cpu().numpy().astype(np.float32).reshape(-1).copy()
            visual = name.startswith(('conv.', 'encode.'))
            head = name.startswith('advantage.2.')
            fan_in = int(np.prod(parameter.shape[1:])) if parameter.ndim > 1 else parameter.numel()
            scale = 1. if head else max(float(np.sqrt(np.mean(part.astype(float)**2))), 1 / math.sqrt(fan_in))
            self.layout.append({'name': name, 'shape': list(parameter.shape), 'start': offset,
                                'end': offset + len(part), 'visual': visual, 'mutation_scale': scale})
            parts.append(part)
            scales.append(np.full(len(part), scale, np.float32))
            offset += len(part)
        self.source, self.mutation_scale = np.concatenate(parts), np.concatenate(scales)
        self.layout_sha256 = hashlib.sha256(json.dumps(self.layout, sort_keys=True).encode()).hexdigest()
        self.visual_parameters = sum(item['end'] - item['start'] for item in self.layout if item['visual'])
        self._cache = {}

    def lift_head(self, head_genome):
        head_genome = np.asarray(head_genome, np.float32)
        if head_genome.shape != (self.actions * self.hidden + self.actions,) or not np.isfinite(head_genome).all():
            raise ValueError('An exact finite ancestral action-head genome is required.')
        result = self.source.copy()
        for item in self.layout:
            if item['name'] == 'advantage.2.weight_mu':
                result[item['start']:item['end']] = head_genome[:-self.actions]
            elif item['name'] == 'advantage.2.bias_mu':
                result[item['start']:item['end']] = head_genome[-self.actions:]
        return result

    def visual_delta_l2(self, genome):
        return math.sqrt(sum(float(np.square((genome[item['start']:item['end']] -
            self.source[item['start']:item['end']]).astype(float)).sum()) for item in self.layout if item['visual']))

    @torch.no_grad()
    def logits(self, observation, genomes):
        active = {}
        for genome in genomes:
            key = id(genome)
            if key in self._cache and self._cache[key][0] is genome:
                active[key] = self._cache[key]
                continue
            if genome.shape != self.source.shape or not np.isfinite(genome).all():
                raise ValueError('Visual genome dimensions/finite values failed.')
            values = torch.as_tensor(genome.copy(), device=self.device)
            parameters = {item['name']: values[item['start']:item['end']].reshape(item['shape']) for item in self.layout}
            active[key] = (genome, parameters)  # Strong reference prevents recycled object-id collisions.
        self._cache = active  # Bound GPU genotype storage to currently evaluated candidates.
        parameters = {item['name']: torch.stack([active[id(g)][1][item['name']] for g in genomes]) for item in self.layout}
        images = torch.as_tensor(observation['image'], device=self.device)
        scenarios = torch.as_tensor(observation['scenario'], device=self.device)

        def one(weights, image, scenario):
            return functional_call(self.model, weights, (image[None], scenario[None]), strict=False)[0]

        return vmap(one)(parameters, images, scenarios)


def initial_visual_population(policy, ancestor, source_hashes, population_size):
    """Lift ancestral head genomes and add visual variation, without reusing fitness."""
    from smg_rl.genetic import resume_population

    ancestor = Path(ancestor)
    heads, rng, generation, _, final = resume_population(
        ancestor, source_hashes, population_size, policy.actions * policy.hidden + policy.actions)
    saved = torch.load(ancestor / 'champion.pt', map_location='cpu', weights_only=True)
    if not all(torch.equal(v.cpu(), saved['source_online'][k]) for k, v in policy.network.state_dict().items()):
        raise ValueError('Ancestral head features do not match the visual evolution base.')
    population = np.stack([policy.lift_head(head) for head in heads])
    population[0] = policy.lift_head(saved['genome'].numpy())
    policy.source = population[0].copy()
    for i in range(1, population_size):
        noise = rng.normal(0, .02, policy.source.shape) * policy.mutation_scale
        for item in policy.layout:
            if item['name'].startswith('advantage.2.'):
                noise[item['start']:item['end']] = 0  # Preserve ancestral head variation initially.
        population[i] = np.clip(population[i] + noise, -5, 5).astype(np.float32)
    return population, rng, {'parent_mode': 'head', 'parent_generation': generation,
                            'parent_champion_fitness_world_units': final['champion']['fitness_world_units'],
                            'reused_fitness_trials': 0}


def save_population(folder, name, population, metadata):
    from smg_rl.checkpoints import digest
    from smg_rl.genetic import checked_json

    folder = Path(folder)
    pending = folder / f'{name}.tmp'
    destination = folder / f'{name}.npz'
    with pending.open('wb') as stream:
        np.savez_compressed(stream, genomes=population)
    pending.replace(destination)
    record = {**metadata, 'sha256': digest(destination), 'shape': list(population.shape)}
    checked_json(folder / f'{name}.json', record)
    return record


def resume_visual_population(policy, folder, source_hashes, population_size):
    from smg_rl.checkpoints import digest

    folder = Path(folder)
    final = json.loads((folder / 'summary.json').read_text())
    meta = json.loads((folder / 'population-final.json').read_text())
    if (final['state'] != 'completed' or final['error'] or final.get('policy_mode') != 'full_visual'
            or final['source_hashes'] != source_hashes or final['gene_layout_sha256'] != policy.layout_sha256
            or final['population'] != population_size or digest(folder / 'population-final.npz') != meta['sha256']
            or digest(folder / 'champion.pt') != final['champion_sha256']):
        raise ValueError('Full-visual population resume integrity/identity failed.')
    with np.load(folder / 'population-final.npz', allow_pickle=False) as data:
        population = data['genomes'].copy()
    if population.shape != (population_size, len(policy.source)) or not np.isfinite(population).all():
        raise ValueError('Full-visual pending population dimensions/finite values failed.')
    baseline = np.load(folder / 'baseline-genome.npy', allow_pickle=False)
    if (baseline.shape != policy.source.shape or not np.isfinite(baseline).all()
            or hashlib.sha256(baseline.tobytes()).hexdigest() != final['baseline_genome_sha256']):
        raise ValueError('Full-visual ancestral baseline identity failed.')
    policy.source = baseline.copy()
    rng = np.random.default_rng()
    rng.bit_generator.state = meta['rng_state']
    inherited = {}
    for line in (folder / 'trials.jsonl').read_text().splitlines():
        record = json.loads(line)
        if record['phase'] == 'evolution' and record['generation'] == meta['generation'] and record['fitness_eligible']:
            i = record['individual']
            expected = hashlib.sha256(population[i].tobytes()).hexdigest()
            if i in inherited or expected != record['genome_sha256']:
                raise ValueError('Full-visual inherited pending trial identity failed.')
            inherited[i] = {**record, 'inherited_from_run': str(folder)}
    return population, rng, meta['generation'], inherited, final
