import csv
from types import SimpleNamespace

import pytest

from scripts.review_regional_six import audit_episode


def test_completion_review_keeps_assisted_baseline_out_of_new_distance_and_closes_failure(tmp_path):
    path = tmp_path / 'episode.csv'
    initial = {'action': 'reset', 'variation': 'sweet_sweet', 'start_kind': 'curriculum',
        'frontier_state': 'verified-anchor.state', 'outcome': 'running', 'intrinsic_reward': 0,
        'progress_reward': 0, 'emulated_frames': 0, 'complete': False, 'dead': False,
        'scene_nerve': '0x806A1BD0', 'start_relative_route_progress': 0,
        'shaping_potential': .2, 'regional_shaping_reward': 0, 'reward': 0,
        'route_prefix_exited': False, 'best_progress': .8, 'health_lost': 0,
        'damage_penalty': 0, 'grounded': True}
    final = {**initial, 'action': 'w', 'outcome': 'death', 'dead': True,
             'emulated_frames': 12, 'start_relative_route_progress': .05,
             'shaping_potential': 0, 'regional_shaping_reward': -.2, 'reward': -2.201}
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=initial)
        writer.writeheader()
        writer.writerows([initial, final])
    config = {'game': {'id': 'RMGE01'}, 'regional_reward': {'gamma_per_frame': .997 ** (1 / 12)}}
    result = audit_episode(path, config, SimpleNamespace(total_length=100))
    assert result['maximum_start_relative_distance'] == pytest.approx(5)
    assert not result['full_start_star'] and result['terminal_shaping_closure_error'] == 0
    assert result['discounted_return'] < 0
    final['regional_shaping_reward'] = 0  # A partial reward left unclosed must be rejected.
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=initial)
        writer.writeheader()
        writer.writerows([initial, final])
    with pytest.raises(AssertionError):
        audit_episode(path, config, SimpleNamespace(total_length=100))
