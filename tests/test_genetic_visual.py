import numpy as np
import pytest
import torch

from smg_rl.config import load_config
from smg_rl.dqn import DistributionalAgent
from smg_rl.genetic import GeneticHead, next_population
from smg_rl.genetic_visual import GeneticVisualPolicy
from smg_rl.replay import replay_signature


def small_agent():
    config = load_config('config/impala-iqn-smg1.yaml')
    config['environment']['image_size'] = 40
    config['dqn'].update(channels=2, hidden=16, quantiles=4)
    torch.set_num_threads(1)
    return DistributionalAgent(replay_signature(config), config['dqn'])


def test_visual_genome_reproduces_ancestral_head_and_uses_distinct_cnn_per_individual():
    agent = small_agent()
    head = GeneticHead(agent)
    policy = GeneticVisualPolicy(agent)
    observation = {'image': np.random.default_rng(11).integers(0, 256, (3, 4, 40, 40), dtype=np.uint8),
                   'scenario': np.ones((3, 1), np.float32)}
    lifted = policy.lift_head(head.source)
    reference = head.logits(observation, [head.source] * 3)
    actual = policy.logits(observation, [lifted] * 3)
    assert torch.allclose(reference, actual, atol=1e-6, rtol=1e-5)
    assert policy.visual_parameters > 0 and len(lifted) > len(head.source)
    before = {k: v.clone() for k, v in policy.network.state_dict().items()}
    mutant = lifted.copy()
    layer = next(item for item in policy.layout if item['name'].startswith('conv.') and len(item['shape']) == 4)
    mutant[layer['start']:layer['end']] += .25
    mixed = policy.logits(observation, [lifted, mutant, lifted])
    assert torch.allclose(mixed[[0, 2]], actual[[0, 2]], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(mixed[1], actual[1], atol=1e-6, rtol=1e-5)
    assert policy.visual_delta_l2(mutant) > 0 and policy.visual_delta_l2(lifted) == 0
    assert all(torch.equal(v, before[k]) for k, v in policy.network.state_dict().items())
    assert not mixed.requires_grad and all(p.grad is None for p in policy.network.parameters())
    assert len(policy._cache) == 2
    policy.logits({k: v[:1] for k, v in observation.items()}, [lifted])
    assert len(policy._cache) == 1


def test_scaled_mutation_changes_visual_genes_and_preserves_elites():
    policy = GeneticVisualPolicy(small_agent())
    population = np.stack([policy.source.copy() for _ in range(6)])
    children = next_population(population, np.arange(6), np.random.default_rng(11), elite=1,
                               mutation_scale=policy.mutation_scale)
    assert np.array_equal(children[0], population[-1])
    assert all(policy.visual_delta_l2(child) > 0 for child in children[1:])
    assert all(item['mutation_scale'] == 1 for item in policy.layout if item['name'].startswith('advantage.2.'))
    assert not any(item['name'].endswith(('weight_sigma', 'bias_sigma')) for item in policy.layout)
    with pytest.raises(ValueError, match='Mutation scales'):
        next_population(population, np.arange(6), np.random.default_rng(11), mutation_scale=np.zeros(2))


def test_visual_population_checkpoint_roundtrip_preserves_rng_and_rejects_corruption(tmp_path):
    import json

    from smg_rl.checkpoints import digest
    from smg_rl.genetic import genome_digest
    from smg_rl.genetic_visual import resume_visual_population, save_population

    policy = GeneticVisualPolicy(small_agent())
    population = np.stack([policy.source.copy() for _ in range(6)])
    rng = np.random.default_rng(11)
    save_population(tmp_path, 'population-final', population,
        {'generation': 3, 'rng_state': rng.bit_generator.state})
    np.save(tmp_path / 'baseline-genome.npy', policy.source, allow_pickle=False)
    (tmp_path / 'champion.pt').write_bytes(b'fixture, not a trained checkpoint')
    final = {'state': 'completed', 'error': None, 'policy_mode': 'full_visual',
             'source_hashes': {'fixture': 'source'}, 'gene_layout_sha256': policy.layout_sha256,
             'population': 6, 'champion_sha256': digest(tmp_path / 'champion.pt'),
             'baseline_genome_sha256': genome_digest(policy.source)}
    (tmp_path / 'summary.json').write_text(json.dumps(final))
    (tmp_path / 'trials.jsonl').write_text(json.dumps({'phase': 'evolution', 'generation': 3,
        'individual': 0, 'fitness_eligible': True, 'genome_sha256': genome_digest(population[0])}) + '\n')
    restored, restored_rng, generation, inherited, _ = resume_visual_population(policy, tmp_path, {'fixture': 'source'}, 6)
    assert np.array_equal(restored, population) and restored_rng.bit_generator.state == rng.bit_generator.state
    assert generation == 3 and list(inherited) == [0]
    with (tmp_path / 'population-final.npz').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        resume_visual_population(policy, tmp_path, {'fixture': 'source'}, 6)


def test_visual_dispatch_refuses_duplicate_state_before_pausing_iqn(tmp_path, monkeypatch):
    from scripts import run_visual_genetic_trial

    state = tmp_path / 'existing.json'
    state.write_text('{}')
    monkeypatch.setattr(run_visual_genetic_trial, 'STATE', state)
    with pytest.raises(ValueError, match='never duplicate'):
        run_visual_genetic_trial.supervise()
