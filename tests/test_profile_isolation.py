from smg_rl.parallel import runtime_directory


def test_simultaneous_game_runs_cannot_share_worker_profile_even_with_identical_names(tmp_path):
    timestamp = "parallel-20261022-125705-842348"
    galaxy2 = {"game": {"id": "SB4E01"}, "paths": {"user": str(tmp_path / ".local/native-user")}}
    galaxy1 = {"game": {"id": "RMGE01"}, "paths": {"user": str(tmp_path / ".local/smg1-native-user")}}
    first = runtime_directory(galaxy2, tmp_path / "runs" / timestamp)
    second = runtime_directory(galaxy1, tmp_path / "runs/smg1" / timestamp)
    assert first != second
    assert first.parent.name == "SB4E01" and second.parent.name == "RMGE01"
    assert first.name == second.name == timestamp
