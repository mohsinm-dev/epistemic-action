"""Tests for correlated and miscalibrated evidence stress models."""

import math

from epistemic_action.evidence import DecisionCosts, Signal
from epistemic_action.stress import joint_signal_probability, posterior_from_pair, symmetric_source
from epistemic_action.stress_experiment import run_condition


def test_joint_distribution_normalizes() -> None:
    """All four signal-pair outcomes should sum to one for each latent state."""
    from epistemic_action.evidence import State

    source = symmetric_source("test", 0.85)
    for state in State:
        total = sum(
            joint_signal_probability(source, first, second, state, correlation=0.6)
            for first in Signal
            for second in Signal
        )
        assert math.isclose(total, 1.0, abs_tol=1e-12)


def test_zero_correlation_matches_naive_independence() -> None:
    """Correlation-aware inference should reduce to independence at rho=0."""
    source = symmetric_source("test", 0.85)
    aware = posterior_from_pair(
        0.1,
        source,
        Signal.FLAGGED,
        Signal.FLAGGED,
        correlation=0.0,
    )
    naive = posterior_from_pair(
        0.1,
        source,
        Signal.FLAGGED,
        Signal.FLAGGED,
        correlation=0.0,
    )
    assert math.isclose(aware, naive, abs_tol=1e-12)


def test_perfect_correlation_makes_duplicate_signal_redundant() -> None:
    """At rho=1, repeating the same channel should add no extra evidence."""
    from epistemic_action.evidence import posterior_suspicious

    source = symmetric_source("test", 0.85)
    single = posterior_suspicious(0.1, source, Signal.FLAGGED)
    paired = posterior_from_pair(
        0.1,
        source,
        Signal.FLAGGED,
        Signal.FLAGGED,
        correlation=1.0,
    )
    assert math.isclose(single, paired, abs_tol=1e-12)


def test_false_independence_overcounts_repeated_positive_evidence() -> None:
    """Ignoring strong dependence should make repeated flags too persuasive."""
    source = symmetric_source("test", 0.85)
    naive = posterior_from_pair(
        0.02,
        source,
        Signal.FLAGGED,
        Signal.FLAGGED,
        correlation=0.0,
    )
    aware = posterior_from_pair(
        0.02,
        source,
        Signal.FLAGGED,
        Signal.FLAGGED,
        correlation=0.8,
    )
    assert naive > aware
    assert naive > 1.0 / 6.0
    assert aware < 1.0 / 6.0


def test_high_correlation_hurts_naive_decision_loss() -> None:
    """A naive independence model should lose more under strong dependence."""
    results = run_condition(
        prior_suspicious=0.02,
        true_accuracy=0.85,
        assumed_accuracy=0.85,
        correlation=0.8,
        false_approve_cost=5.0,
        episodes=20_000,
        seed=17,
    )
    by_model = {result.model: result for result in results}
    assert by_model["naive_independent"].mean_decision_loss > by_model[
        "correlation_aware"
    ].mean_decision_loss
    assert by_model["naive_independent"].brier_score > by_model[
        "correlation_aware"
    ].brier_score


def test_miscalibration_changes_posterior_even_with_correct_correlation() -> None:
    """Wrong source reliability should remain a distinct model error."""
    calibrated = run_condition(
        prior_suspicious=0.05,
        true_accuracy=0.85,
        assumed_accuracy=0.85,
        correlation=0.5,
        false_approve_cost=5.0,
        episodes=5_000,
        seed=23,
    )
    optimistic = run_condition(
        prior_suspicious=0.05,
        true_accuracy=0.85,
        assumed_accuracy=0.95,
        correlation=0.5,
        false_approve_cost=5.0,
        episodes=5_000,
        seed=23,
    )
    calibrated_model = next(r for r in calibrated if r.model == "correlation_aware")
    optimistic_model = next(r for r in optimistic if r.model == "correlation_aware")
    assert optimistic_model.mean_posterior_shift > calibrated_model.mean_posterior_shift


def test_stress_condition_is_reproducible() -> None:
    """Identical seeds and parameters should produce identical aggregates."""
    kwargs = dict(
        prior_suspicious=0.05,
        true_accuracy=0.85,
        assumed_accuracy=0.85,
        correlation=0.5,
        false_approve_cost=5.0,
        episodes=2_000,
        seed=31,
    )
    assert run_condition(**kwargs) == run_condition(**kwargs)
