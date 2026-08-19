"""Tests for the transparent Active Inference comparison."""

import math

from epistemic_action.active_inference import (
    build_abcd_model,
    expected_state_information_gain,
    select_sophisticated_efe_source,
    select_standard_efe_source,
)
from epistemic_action.evidence import DecisionCosts
from epistemic_action.sequential import sequential_sources


def test_information_gain_is_positive_for_informative_source() -> None:
    """The screen should reduce hidden-state uncertainty at the default prior."""
    screen, _ = sequential_sources()
    assert expected_state_information_gain(0.05, screen) > 0.0


def test_standard_efe_can_stop_at_default_precision() -> None:
    """Open-loop EFE should be allowed to prefer acting immediately."""
    sources = sequential_sources()
    costs = DecisionCosts(false_approve=5.0, false_reject=1.0)

    source = select_standard_efe_source(
        0.05,
        sources,
        costs,
        escalation_cost=0.40,
        preference_precision=5.0,
        steps_remaining=2,
    )

    assert source is None


def test_sophisticated_efe_values_screening_at_default_precision() -> None:
    """Observation-contingent EFE should recover the cheap screening action."""
    sources = sequential_sources()
    costs = DecisionCosts(false_approve=5.0, false_reject=1.0)

    source = select_sophisticated_efe_source(
        0.05,
        sources,
        costs,
        escalation_cost=0.40,
        preference_precision=5.0,
        steps_remaining=2,
    )

    assert source is not None
    assert source.name == "screen"


def test_abcd_priors_are_normalized() -> None:
    """D priors should be proper distributions."""
    model = build_abcd_model(
        prior_suspicious=0.05,
        sources=sequential_sources(),
        costs=DecisionCosts(false_approve=5.0, false_reject=1.0),
        escalation_cost=0.40,
        preference_precision=2.0,
    )

    assert math.isclose(sum(model.D_transaction), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(model.D_context), 1.0, abs_tol=1e-12)


def test_abcd_transition_columns_are_normalized() -> None:
    """Each B transition column should sum to one."""
    model = build_abcd_model(
        prior_suspicious=0.05,
        sources=sequential_sources(),
        costs=DecisionCosts(false_approve=5.0, false_reject=1.0),
        escalation_cost=0.40,
        preference_precision=2.0,
    )

    for control in range(len(model.controls)):
        for current_state in range(2):
            assert math.isclose(
                sum(
                    model.B_transaction[next_state][current_state][control]
                    for next_state in range(2)
                ),
                1.0,
                abs_tol=1e-12,
            )
        for current_context in range(len(model.contexts)):
            assert math.isclose(
                sum(
                    model.B_context[next_context][current_context][control]
                    for next_context in range(len(model.contexts))
                ),
                1.0,
                abs_tol=1e-12,
            )


def test_preference_precision_maps_costs_to_log_preferences() -> None:
    """C should encode the benchmark losses and evidence costs consistently."""
    precision = 3.0
    model = build_abcd_model(
        prior_suspicious=0.05,
        sources=sequential_sources(),
        costs=DecisionCosts(false_approve=5.0, false_reject=1.0),
        escalation_cost=0.40,
        preference_precision=precision,
    )

    assert math.isclose(model.C_terminal[2], -15.0)
    assert math.isclose(model.C_terminal[3], -3.0)
    assert math.isclose(model.C_terminal[4], -1.2)
    assert math.isclose(model.C_cost[1], -0.06)
    assert math.isclose(model.C_cost[2], -0.60)
