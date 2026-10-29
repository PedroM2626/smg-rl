from scripts.run_density_four_worker_continuation import segment_seconds


def test_continuation_reserves_cleanup_and_does_not_extend_budget():
    assert segment_seconds(0, 10000, maximum=300) == 300
    assert segment_seconds(7000, 10000) == 80
    assert segment_seconds(7100, 10000) == 0
    assert segment_seconds(0, 200) == 80
    assert segment_seconds(0, -10) == 0
