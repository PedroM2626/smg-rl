from datetime import datetime, timedelta

from scripts.run_human_route_density_recovery import DEADLINE, segment_seconds


def test_recovery_reserves_time_and_never_exceeds_cap_or_deadline():
    early = DEADLINE - timedelta(hours=4)
    assert segment_seconds(early, 1209.09) == 600
    assert segment_seconds(early, 17000) == 0
    assert segment_seconds(early, 16900) == 80
    assert segment_seconds(DEADLINE - timedelta(seconds=200), 0) == 80
    assert segment_seconds(DEADLINE, 0) == 0
    assert segment_seconds(early, 0, maximum=300) == 300
    assert isinstance(DEADLINE, datetime)
