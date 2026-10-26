import csv
import json
from copy import deepcopy

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from smg_rl import training_metrics
from smg_rl.config import load_config
from smg_rl.training_metrics import MetricsExporter, episode_metrics


def row(x=0., y=0., z=0., reward=0., advance=0., action='none', dead='False', complete='False'):
    return dict(x=x, y=y, z=z, reward=reward, start_relative_route_progress=advance,
                action=action, dead=dead, complete=complete, time=100.)


def test_backtracking_distance_is_not_net_displacement_or_grounded_progress():
    rows = [row(action='reset'), row(3., z=4., advance=.1, reward=.1), row(advance=.1),
            row(9000., reward=-2., advance=.1, dead='True')]
    metrics = episode_metrics(rows, 1000., 2500.)
    assert metrics['path_distance_xz'] == pytest.approx(10.)
    assert metrics['net_displacement_xz'] == 0
    assert metrics['valid_route_advance'] == 100.
    assert metrics['opening_advance_percent'] == 10.
    assert metrics['episode_reward'] == pytest.approx(-1.9)
    assert metrics['excluded_position_samples'] == 1


def test_jumps_and_discontinuities_cannot_be_confused_with_forward_progress():
    rows = [row(action='reset'), row(y=5.), row(), row(9000.), row(9000., complete='True')]
    metrics = episode_metrics(rows, 1000., 2500.)
    assert metrics['path_distance_xyz'] == 10.
    assert metrics['path_distance_xz'] == metrics['valid_route_advance'] == 0
    assert metrics['excluded_position_samples'] == 2
    assert 'net_displacement_xz' not in metrics  # A teleport cannot become claimed net travel.


def test_export_backfills_both_axes_and_resume_does_not_duplicate_attempts(tmp_path, monkeypatch):
    config = deepcopy(load_config('config/impala-iqn-smg1.yaml'))
    config['paths']['tensorboard'] = str(tmp_path / 'tensorboard')
    monkeypatch.setattr(training_metrics, 'load_config', lambda _: config)
    run = tmp_path / 'run'
    folder = run / 'worker-00' / 'sweet_sweet'
    folder.mkdir(parents=True)
    ledger = []
    for attempt in (1, 2):
        rows = [row(action='reset'), row(x=attempt, reward=.1, advance=.1),
                row(x=attempt, reward=-2., advance=.1, dead='True')]
        with (folder / f'episode-{attempt:06d}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        ledger.append(dict(mission='sweet_sweet', start_kind='full_start', worker=0,
                           worker_episode=attempt, attempt=attempt, real_decisions=attempt * 10, return_=-1.9))
        ledger[-1]['return'] = ledger[-1].pop('return_')
    (run / 'attempts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in ledger))
    training = {'phase': 'main_running', 'games': {'smg1': {'jobs': [{'run': str(run), 'config': 'fixture'}]}}}
    state = tmp_path / 'export.json'
    exporter = MetricsExporter(state)
    assert exporter.export(training) == 2
    assert exporter.export(training) == 0
    exporter.close()
    restored = MetricsExporter(state)
    assert restored.export(training) == 0
    restored.close()
    events = EventAccumulator(str(tmp_path / 'tensorboard' / 'telemetry' / 'RMGE01' / 'simple-impala')).Reload()
    reward = events.Scalars('by_attempt/sweet_sweet/full_start/episode_reward')
    assert len(reward) == 2 and [e.step for e in reward] == [1, 2]
    assert reward[-1].value == pytest.approx(-1.9)
    distance = events.Scalars('by_real_decision/sweet_sweet/full_start/path_distance_xz')
    assert [e.step for e in distance] == [10, 20] and distance[-1].value == 2.


def test_metric_export_refuses_unbounded_watching():
    with pytest.raises(ValueError, match='finite'):
        training_metrics.export_metrics(watch=True)


def test_export_refuses_attempt_gaps_instead_of_inventing_history(tmp_path, monkeypatch):
    config = deepcopy(load_config('config/impala-iqn-smg1.yaml'))
    config['paths']['tensorboard'] = str(tmp_path / 'tensorboard')
    monkeypatch.setattr(training_metrics, 'load_config', lambda _: config)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'attempts.jsonl').write_text(json.dumps({'mission': 'sweet_sweet', 'start_kind': 'full_start',
                                                 'attempt': 2}) + '\n')
    exporter = MetricsExporter(tmp_path / 'state.json')
    try:
        with pytest.raises(ValueError, match='Missing attempt'):
            exporter.export({'phase': 'main_running', 'games': {'smg1': {
                'jobs': [{'run': str(run), 'config': 'fixture'}]}}})
    finally:
        exporter.close()
