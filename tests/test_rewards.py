import pytest

from smg_rl.rewards import ProgressReward


def course():
    reward = ProgressReward((0, -1150, 500), (0, -450, -10000))
    reward.reset((0, -1150, 500))
    return reward


def test_backtracking_cannot_farm_progress():
    reward = course()
    delta, _ = reward.update((0, -1150, -500))
    assert delta > 0
    assert reward.update((0, -1150, 500))[0] == 0
    assert reward.update((0, -1150, -500))[0] == 0
    assert reward.update((0, -1150, -1000))[0] > 0


def test_falling_and_death_cannot_earn_progress():
    assert course().update((0, -2500, -500))[0] == 0
    assert course().update((0, -1150, -500), dead=True)[0] == 0


def test_teleport_does_not_earn_progress():
    assert course().update((0, -450, -10000))[0] == 0


def test_height_and_sideways_movement_do_not_create_forward_progress():
    assert course().update((100, -800, 500))[0] == 0


def test_projection_is_bounded():
    reward = course()
    assert reward.fraction((0, 0, -20000)) == 1
    assert reward.fraction((0, 0, 20000)) == 0


def test_rejects_degenerate_route():
    with pytest.raises(ValueError):
        ProgressReward((0, 0, 0), (0, 1, 0))
