"""Tests for evidence valuation and one-step policies."""

import math
import random

from epistemic_action.evidence import (
    DecisionCosts,
    EvidenceSource,
    Signal,
    expected_information_gain,
    expected_value_of_information,
    posterior_suspicious,
)
from epistemic_action.evidence_experiment import run_condition
from epistemic_action.policies import InformationGainPolicy, ValueOfInformationPolicy


def test_flagged_signal_increases_suspicion_for_informative_source() -> None:
    """A positive likelihood ratio should move the posterior upward."""
    source = EvidenceSource("test", sensitivity=0.9, specificity=0.9, cost=0.0)
    posterior = posterior_suspicious(0.1, source, Signal.FLAGGED)
    assert posterior > 0.1


def test_uninformative_source_has_zero_information_gain() -> None:
    """A source independent of state should provide no expected information."""
    source = EvidenceSource("noise", sensitivity=0.5, specificity=0.5, cost=0.0)
    assert math.isclose(expected_information_gain(0.2, source), 0.0, abs_tol=1e-12)


def test_information_gain_ignores_acquisition_cost() -> None:
    """Entropy reduction alone may choose an expensive but informative source."""
    cheap = EvidenceSource("cheap", sensitivity=0.8, specificity=0.8, cost=0.0)
    expensive = EvidenceSource("expensive", sensitivity=0.99, specificity=0.99, cost=100.0)
    policy = InformationGainPolicy()

    selected = policy.select_source(
        prior_suspicious=0.2,
        sources=(cheap, expensive),
        costs=DecisionCosts(),
        rng=random.Random(1),
    )

    assert selected is expensive


def test_value_of_information_rejects_overpriced_evidence() -> None:
    """Evidence with acquisition cost above its decision value should be skipped."""
    source = EvidenceSource("expensive", sensitivity=0.99, specificity=0.99, cost=100.0)
    costs = DecisionCosts(false_approve=5.0, false_reject=1.0)
    assert expected_value_of_information(0.1, source, costs) < 0.0

    selected = ValueOfInformationPolicy().select_source(
        prior_suspicious=0.1,
        sources=(source,),
        costs=costs,
        rng=random.Random(1),
    )
    assert selected is None


def test_value_of_information_selects_decision_relevant_source() -> None:
    """The policy should prefer a cheaper source when it has higher net value."""
    cheap = EvidenceSource("cheap", sensitivity=0.9, specificity=0.9, cost=0.05)
    expensive = EvidenceSource("expensive", sensitivity=0.99, specificity=0.99, cost=2.0)
    costs = DecisionCosts(false_approve=5.0, false_reject=1.0)

    selected = ValueOfInformationPolicy().select_source(
        prior_suspicious=0.1,
        sources=(cheap, expensive),
        costs=costs,
        rng=random.Random(1),
    )
    assert selected is cheap


def test_condition_is_reproducible() -> None:
    """Identical seeds and settings should return identical aggregates."""
    first = run_condition(
        prior_suspicious=0.1,
        false_approve_cost=5.0,
        episodes=1_000,
        seed=17,
    )
    second = run_condition(
        prior_suspicious=0.1,
        false_approve_cost=5.0,
        episodes=1_000,
        seed=17,
    )
    assert first == second


def test_value_of_information_can_reduce_total_loss() -> None:
    """Under asymmetric error costs, useful evidence should beat acting immediately."""
    results = run_condition(
        prior_suspicious=0.1,
        false_approve_cost=5.0,
        episodes=20_000,
        seed=23,
    )
    by_policy = {result.policy: result for result in results}
    assert (
        by_policy["value_of_information"].mean_total_loss
        < by_policy["greedy"].mean_total_loss
    )
