"""Tests for finite-horizon evidence acquisition."""

import math

from epistemic_action.evidence import DecisionCosts
from epistemic_action.sequential import (
    optimal_value,
    select_lookahead_source,
    select_myopic_source,
    sequential_sources,
)
from epistemic_action.sequential_experiment import run_condition


def test_horizon_one_matches_myopic_policy() -> None:
    """One-step exact planning should reduce to myopic VoI selection."""
    sources = sequential_sources()
    costs = DecisionCosts(false_approve=5.0, false_reject=1.0)

    myopic = select_myopic_source(0.05, sources, costs, escalation_cost=0.40)
    lookahead = select_lookahead_source(
        0.05,
        sources,
        costs,
        escalation_cost=0.40,
        steps_remaining=1,
    )

    assert myopic is None
    assert lookahead is None


def test_two_step_planner_values_screening_option() -> None:
    """Two-step planning should buy a screen that has no positive one-step value."""
    sources = sequential_sources()
    costs = DecisionCosts(false_approve=5.0, false_reject=1.0)

    source = select_lookahead_source(
        0.05,
        sources,
        costs,
        escalation_cost=0.40,
        steps_remaining=2,
    )

    assert source is not None
    assert source.name == "screen"


def test_two_step_value_exceeds_stopping_value() -> None:
    """The exact planner should gain utility from the two-stage acquisition policy."""
    sources = sequential_sources()
    costs = DecisionCosts(false_approve=5.0, false_reject=1.0)

    one_step = optimal_value(
        0.05,
        sources,
        costs,
        escalation_cost=0.40,
        steps_remaining=1,
    )
    two_step = optimal_value(
        0.05,
        sources,
        costs,
        escalation_cost=0.40,
        steps_remaining=2,
    )

    assert math.isclose(one_step, -0.25, abs_tol=1e-12)
    assert two_step > one_step


def test_sequential_condition_is_reproducible() -> None:
    """Identical seeds should produce identical aggregate outcomes."""
    first = run_condition(
        prior_suspicious=0.05,
        false_approve_cost=5.0,
        escalation_cost=0.40,
        horizon=2,
        episodes=2_000,
        seed=31,
    )
    second = run_condition(
        prior_suspicious=0.05,
        false_approve_cost=5.0,
        escalation_cost=0.40,
        horizon=2,
        episodes=2_000,
        seed=31,
    )

    assert first == second


def test_lookahead_reduces_total_loss_against_myopic() -> None:
    """The default two-step benchmark should reward non-myopic screening."""
    results = run_condition(
        prior_suspicious=0.05,
        false_approve_cost=5.0,
        escalation_cost=0.40,
        horizon=2,
        episodes=20_000,
        seed=23,
    )
    by_policy = {result.policy: result for result in results}

    assert by_policy["myopic_voi"].mean_queries == 0.0
    assert by_policy["lookahead"].mean_queries > 1.0
    assert by_policy["lookahead"].mean_total_loss < by_policy["myopic_voi"].mean_total_loss
    assert by_policy["lookahead"].mean_total_loss < by_policy["information_gain"].mean_total_loss
