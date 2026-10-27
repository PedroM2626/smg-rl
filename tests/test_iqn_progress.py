import csv

import pytest

from smg_rl.iqn_progress import episode_projection


def write_trace(path, kind='full_start'):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'x', 'y', 'z', 'grounded', 'dead', 'complete', 'scene_nerve',
                         'start_kind', 'outcome', 'route_prefix_progress'])
        writer.writerows([[0, 0, 0, 0, True, False, False, '0x806A1BD0', kind, 'running', 0],
                          [1, 20, 0, 0, True, False, False, '0x806A1BD0', kind, 'running', 1],
                          [2, 40, 0, 0, False, False, False, '0x806A1BD0', kind, 'interrupted', 1]])


def test_goal_projection_is_not_opening_fraction_or_airborne_maximum(tmp_path):
    path = tmp_path / 'fixture.csv'
    write_trace(path)
    report = episode_projection(path, {'start': [0, 0, 0], 'goal': [100, 0, 0]}, 50)
    assert report['geometric_projection_percent'] == 20
    assert report['mapped_opening_fraction'] == 1
    assert report['partial_episode'] and not report['ram_star']


def test_assisted_starts_are_not_global_full_start_progress(tmp_path):
    path = tmp_path / 'fixture.csv'
    write_trace(path, 'curriculum')
    assert episode_projection(path, {}, 50) == {'excluded': 'assisted_or_missing_start'}


def test_native_completion_requires_the_correct_ram_scene(tmp_path):
    path = tmp_path / 'fixture.csv'
    write_trace(path)
    content = path.read_text().replace('False,0x806A1BD0,full_start,interrupted',
                                        'True,0x806A1BD0,full_start,interrupted')
    path.write_text(content)
    with pytest.raises(ValueError, match='native completion RAM'):
        episode_projection(path, {'start': [0, 0, 0], 'goal': [100, 0, 0]}, 50)


def test_ram_completion_is_not_discarded_when_geometry_is_invalid(tmp_path):
    path = tmp_path / 'fixture.csv'
    write_trace(path)
    content = path.read_text().replace('2,40,0,0,False,False,False,0x806A1BD0',
                                        '2,500,0,0,False,False,True,0x806A1BD8')
    path.write_text(content)
    assert episode_projection(path, {'start': [0, 0, 0], 'goal': [100, 0, 0]}, 50) == {
        'excluded': 'nonfinite_or_discontinuous_position', 'ram_star': True}
