"""Tests for the deterministic planner-disagreement campaign."""

import math

from epistemic_action.disagreement_experiment import run_condition


def _default_results(preference_precision: float) -> dict[str, object]:
    """Return the corrected option-value condition keyed by policy."""
    results = run_condition(
        prior_suspicious=0.05,
        screen_accuracy=0.70,
        screen_cost=0.05,
        review_accuracy=0.95,
        review_cost=0.20,
        false_approve_cost=5.0,
        escalation_cost=0.40,
        preference_precision=preference_precision,
        horizon=2,
    )
    return {result.policy: result for result in results}


def test_corrected_condition_has_genuine_option_value() -> None:
    """Bayes should screen even though myopic VoI stops."""
    results = _default_results(preference_precision=5.0)

    assert results["myopic_voi"].first_action == "stop"
    assert results["bayes_lookahead"].first_action == "screen"
    assert math.isclose(results["myopic_voi"].expected_total_loss, 0.25, abs_tol=1e-12)
    assert math.isclose(results["bayes_lookahead"].expected_total_loss, 0.212, abs_tol=1e-12)


def test_standard_efe_misses_default_option_value() -> None:
    """Open-loop EFE should stop under the default precision."""
    results = _default_results(preference_precision=5.0)

    assert results["standard_efe"].first_action == "stop"
    assert results["standard_efe"].expected_total_loss > results["bayes_lookahead"].expected_total_loss


def test_sophisticated_efe_matches_bayes_at_default_precision() -> None:
    """Observation-contingent EFE should recover the Bayes policy here."""
    results = _default_results(preference_precision=5.0)

    assert results["sophisticated_efe"].first_action == "screen"
    assert math.isclose(
        results["sophisticated_efe"].expected_total_loss,
        results["bayes_lookahead"].expected_total_loss,
        abs_tol=1e-12,
    )


def test_low_precision_sophisticated_efe_over_explores() -> None:
    """A strong relative epistemic term should prefer costly review and add regret."""
    results = _default_results(preference_precision=0.5)

    assert results["bayes_lookahead"].first_action == "screen"
    assert results["sophisticated_efe"].first_action == "review"
    assert results["sophisticated_efe"].regret_vs_bayes > 0.0
