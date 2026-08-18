"""Tests for reproducible experiment execution."""

from epistemic_action.experiment import run_condition


def test_condition_is_reproducible() -> None:
    """Identical seeds and settings should return identical aggregates."""
    first = run_condition(
        clue_reliability=0.8,
        clue_cost=0.05,
        episodes=500,
        seed=11,
        information_weight=1.0,
    )
    second = run_condition(
        clue_reliability=0.8,
        clue_cost=0.05,
        episodes=500,
        seed=11,
        information_weight=1.0,
    )
    assert first == second


def test_reliable_clue_improves_epistemic_accuracy() -> None:
    """With a reliable cheap clue, epistemic accuracy should exceed greedy accuracy."""
    greedy, epistemic = run_condition(
        clue_reliability=0.9,
        clue_cost=0.0,
        episodes=5_000,
        seed=19,
        information_weight=1.0,
    )
    assert epistemic.accuracy > greedy.accuracy
    assert epistemic.clue_rate == 1.0
