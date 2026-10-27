import pytest

from smg_rl.goal_distance import GroundedGoalDistance


def tracker(initial=(0, 0, 0), max_step=1000):
    return GroundedGoalDistance((0, 0, 0), (100, 0, 0), initial, max_step)


def test_only_new_grounded_closest_distance_is_paid_once():
    score = tracker()
    assert score.update((20, 0, 0), True) == pytest.approx(0.2)
    assert score.update((20, 0, 0), True) == 0
    assert score.update((10, 0, 0), True) == 0
    assert score.update((20, 0, 0), True) == 0
    assert score.update((30, 0, 0), True) == pytest.approx(0.1)


def test_airborne_death_and_assisted_reset_do_not_get_free_credit():
    score = tracker(initial=(50, 0, 0))
    assert score.update((60, 0, 0), False) == 0
    assert score.update((60, 0, 0), True) == pytest.approx(0.1)
    assert score.update((70, 0, 0), True, dead=True) == 0
    assert score.update((80, 0, 0), True, complete=True) == 0


def test_discontinuity_and_nonfinite_samples_fail_closed():
    score = tracker(max_step=10)
    assert score.update((11, 0, 0), True) == 0
    assert score.invalid
    assert score.update((1, 0, 0), True) == 0
    score = tracker()
    assert score.update((float('nan'), 0, 0), True) == 0
    assert score.invalid


def test_goal_distance_is_not_same_as_projection_or_course_progress():
    goal = (100, 0, 0)
    projected_far_off_course = GroundedGoalDistance((0, 0, 0), goal, (0, 0, 0), 20000)
    reachable_closer = GroundedGoalDistance((0, 0, 0), goal, (0, 0, 0), 20000)
    assert projected_far_off_course.update((90, 0, 1000), True) == 0
    assert reachable_closer.update((60, 0, 0), True) == pytest.approx(0.6)
