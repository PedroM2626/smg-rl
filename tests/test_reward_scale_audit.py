import csv
import importlib.util
import json
from pathlib import Path

import pytest


def test_counterfactual_preserves_terminal_rewards_and_finds_bad_stationary_credit(tmp_path):
    spec = importlib.util.spec_from_file_location('audit_reward_scale', Path('scripts/audit_reward_scale.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / 'run'
    episodes = run / 'worker-00' / 'sweet_sweet'
    episodes.mkdir(parents=True)
    fields = ['start_kind', 'outcome', 'regional_shaping_reward', 'reward', 'x', 'y', 'z', 'grounded', 'dead', 'step']
    rows = [
        ['full_start', 'running', 0, 0, 0, 0, 0, 'True', 'False', 0],
        ['full_start', 'running', .01, .01, 1, 0, 0, 'True', 'False', 1],
        ['full_start', 'running', .02, .02, 1, 0, 0, 'True', 'False', 2],
        ['full_start', 'death', 0, -2, 2, 0, 0, 'False', 'True', 3],
    ]
    with (episodes / 'episode-000001.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'blocks': [{'run': str(run)}]}))
    result = module.audit(state)
    assert result['completed'] == 1 and result['partial_excluded'] == 0
    assert result['by_outcome']['death']['mean_observed_return'] == pytest.approx(-1.97)
    assert result['by_outcome']['death']['mean_counterfactual_return'] == pytest.approx(-1.7)
    assert result['positive_reward_on_identical_xyz_airborne_or_dead'][0]['step'] == '2'
