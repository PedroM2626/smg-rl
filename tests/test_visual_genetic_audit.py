import pytest

from scripts.audit_visual_genetic_result import visual_trial_summary


def test_visual_result_distinguishes_unchanged_elite_from_mutants_and_partial_records():
    baseline = {'fitness_eligible': True, 'fitness_world_units': 10., 'genome_sha256': 'baseline',
                'visual_genome_delta_l2': 0.}
    mutant = {'fitness_eligible': True, 'fitness_world_units': 9., 'genome_sha256': 'mutant',
              'visual_genome_delta_l2': .3}
    partial = {'fitness_eligible': False, 'fitness_world_units': 20., 'genome_sha256': 'partial',
               'visual_genome_delta_l2': .4}
    report = visual_trial_summary([baseline, mutant, partial], 'baseline')
    assert report['completed_mutated_visual_trials'] == report['completed_baseline_trials'] == 1
    assert report['best_complete'] == baseline
    assert report['best_mutated_visual'] == mutant
    with pytest.raises(ValueError, match='No eligible physical'):
        visual_trial_summary([baseline, partial], 'baseline')
