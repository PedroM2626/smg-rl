from pathlib import Path

from scripts.run_density_frozen_review import parse_run


def test_parse_run_only_accepts_a_structured_run_record(tmp_path):
    path = tmp_path / 'log'
    path.write_text('noise\n{not data}\n' + repr({'evaluation_run': 'safe/run'}) + '\n')
    assert parse_run(path) == Path('safe/run')
    path.write_text('no run\n')
    assert parse_run(path) is None
