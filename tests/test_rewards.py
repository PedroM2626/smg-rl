import pytest

from smg_rl.rewards import ExplorationBudget, ProgressReward


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


def test_intrinsic_budget_bounds_repeated_animation_reward():
    budget = ExplorationBudget(limit=0.5)
    paid = sum(budget.pay(0.02, 0)[0] for _ in range(3000))
    assert paid == pytest.approx(0.5)
    assert budget.remaining == 0


def test_no_progress_bonus_fades_and_legacy_config_is_unchanged():
    budget = ExplorationBudget(limit=1, decay_start=30, decay_end=120)
    assert budget.pay(0.1, 30) == pytest.approx((0.1, 1))
    assert budget.pay(0.1, 75) == pytest.approx((0.05, 0.5))
    assert budget.pay(0.1, 120) == (0, 0)
    assert budget.pay(0.1, 0) == pytest.approx((0.1, 1))
    assert ExplorationBudget().pay(0.02, 1000) == (0.02, 1)


def test_stationary_timeout_is_costly_instead_of_positive_living_reward():
    budget = ExplorationBudget(limit=0.5, decay_start=30, decay_end=120)
    revised = sum(-0.001 + budget.pay(0.005, t)[0] for t in range(1, 181)) - 2
    legacy = 180 * (-0.0001 + 0.02)
    assert legacy > 3 and revised < -1.5
