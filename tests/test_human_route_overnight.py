from datetime import datetime, timedelta, timezone

from scripts.run_human_route_overnight import remaining_budget


def test_remaining_budget_reserves_cleanup_and_caps_each_block():
    deadline = datetime(2026, 10, 29, 10, 0, tzinfo=timezone.utc)
    assert remaining_budget(deadline - timedelta(hours=2), deadline=deadline) == 1800
    assert remaining_budget(deadline - timedelta(seconds=200), deadline=deadline) == 80
    assert remaining_budget(deadline - timedelta(seconds=100), deadline=deadline) == 0
    assert remaining_budget(deadline + timedelta(seconds=1), deadline=deadline) == 0
    assert remaining_budget(deadline - timedelta(hours=1), deadline=deadline, maximum=300) == 300
