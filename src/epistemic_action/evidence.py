"""Binary decision environment with optional evidence acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random


_EPSILON = 1e-12


class State(Enum):
    """Latent state of a synthetic transaction."""

    LEGITIMATE = "legitimate"
    SUSPICIOUS = "suspicious"


class Signal(Enum):
    """Binary observation returned by an evidence source."""

    CLEAR = "clear"
    FLAGGED = "flagged"


class Decision(Enum):
    """Terminal action taken by the decision maker."""

    APPROVE = "approve"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """A conditionally independent binary evidence source.

    ``sensitivity`` is ``P(flagged | suspicious)`` and ``specificity`` is
    ``P(clear | legitimate)``. The model intentionally assumes conditional
    independence so that later experiments can violate that assumption.
    """

    name: str
    sensitivity: float
    specificity: float
    cost: float

    def __post_init__(self) -> None:
        """Validate source parameters."""
        if not self.name:
            raise ValueError("name must not be empty")
        if not 0.0 <= self.sensitivity <= 1.0:
            raise ValueError("sensitivity must be in [0.0, 1.0]")
        if not 0.0 <= self.specificity <= 1.0:
            raise ValueError("specificity must be in [0.0, 1.0]")
        if self.cost < 0.0:
            raise ValueError("cost must be non-negative")

    def probability(self, signal: Signal, state: State) -> float:
        """Return ``P(signal | state)`` under this source model."""
        if state is State.SUSPICIOUS:
            flagged = self.sensitivity
        else:
            flagged = 1.0 - self.specificity

        return flagged if signal is Signal.FLAGGED else 1.0 - flagged

    def sample(self, state: State, rng: random.Random) -> Signal:
        """Sample one observation conditional on the latent state."""
        flagged_probability = self.probability(Signal.FLAGGED, state)
        return Signal.FLAGGED if rng.random() < flagged_probability else Signal.CLEAR


@dataclass(frozen=True, slots=True)
class DecisionCosts:
    """Losses for incorrect terminal decisions."""

    false_approve: float = 5.0
    false_reject: float = 1.0

    def __post_init__(self) -> None:
        """Validate decision losses."""
        if self.false_approve < 0.0 or self.false_reject < 0.0:
            raise ValueError("decision costs must be non-negative")


def validate_probability(probability: float, *, name: str) -> None:
    """Raise when a probability lies outside the closed unit interval."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0]")


def posterior_suspicious(
    prior_suspicious: float,
    source: EvidenceSource,
    signal: Signal,
) -> float:
    """Bayes-update ``P(suspicious)`` after observing one source."""
    validate_probability(prior_suspicious, name="prior_suspicious")

    suspicious_likelihood = source.probability(signal, State.SUSPICIOUS)
    legitimate_likelihood = source.probability(signal, State.LEGITIMATE)

    numerator = suspicious_likelihood * prior_suspicious
    evidence = numerator + legitimate_likelihood * (1.0 - prior_suspicious)
    if evidence <= 0.0:
        raise ValueError("signal has zero probability under the model")

    return numerator / evidence


def probability_signal(
    prior_suspicious: float,
    source: EvidenceSource,
    signal: Signal,
) -> float:
    """Return the posterior-predictive probability of an evidence signal."""
    validate_probability(prior_suspicious, name="prior_suspicious")
    return (
        prior_suspicious * source.probability(signal, State.SUSPICIOUS)
        + (1.0 - prior_suspicious) * source.probability(signal, State.LEGITIMATE)
    )


def binary_entropy(probability: float) -> float:
    """Return binary entropy in bits."""
    validate_probability(probability, name="probability")
    if probability <= _EPSILON or probability >= 1.0 - _EPSILON:
        return 0.0

    complement = 1.0 - probability
    return -(
        probability * math.log2(probability)
        + complement * math.log2(complement)
    )


def expected_information_gain(
    prior_suspicious: float,
    source: EvidenceSource,
) -> float:
    """Return expected entropy reduction from acquiring ``source``."""
    prior_entropy = binary_entropy(prior_suspicious)
    expected_posterior_entropy = 0.0

    for signal in Signal:
        signal_probability = probability_signal(prior_suspicious, source, signal)
        if signal_probability <= 0.0:
            continue

        posterior = posterior_suspicious(prior_suspicious, source, signal)
        expected_posterior_entropy += signal_probability * binary_entropy(posterior)

    return max(0.0, prior_entropy - expected_posterior_entropy)


def expected_decision_utility(
    prior_suspicious: float,
    decision: Decision,
    costs: DecisionCosts,
) -> float:
    """Return expected utility of a terminal decision before evidence cost."""
    validate_probability(prior_suspicious, name="prior_suspicious")

    if decision is Decision.APPROVE:
        return -prior_suspicious * costs.false_approve

    return -(1.0 - prior_suspicious) * costs.false_reject


def best_decision(
    prior_suspicious: float,
    costs: DecisionCosts,
) -> Decision:
    """Choose the terminal decision with maximum expected utility."""
    approve_utility = expected_decision_utility(
        prior_suspicious,
        Decision.APPROVE,
        costs,
    )
    reject_utility = expected_decision_utility(
        prior_suspicious,
        Decision.REJECT,
        costs,
    )
    return Decision.APPROVE if approve_utility >= reject_utility else Decision.REJECT


def best_decision_utility(prior_suspicious: float, costs: DecisionCosts) -> float:
    """Return the expected utility of the optimal immediate decision."""
    decision = best_decision(prior_suspicious, costs)
    return expected_decision_utility(prior_suspicious, decision, costs)


def expected_utility_after_evidence(
    prior_suspicious: float,
    source: EvidenceSource,
    costs: DecisionCosts,
) -> float:
    """Return expected optimal decision utility after observing ``source``."""
    expected_utility = 0.0

    for signal in Signal:
        signal_probability = probability_signal(prior_suspicious, source, signal)
        if signal_probability <= 0.0:
            continue

        posterior = posterior_suspicious(prior_suspicious, source, signal)
        expected_utility += signal_probability * best_decision_utility(posterior, costs)

    return expected_utility


def expected_value_of_information(
    prior_suspicious: float,
    source: EvidenceSource,
    costs: DecisionCosts,
) -> float:
    """Return one-step net value of acquiring ``source``.

    This is expected value of sample information: the improvement in optimal
    terminal decision utility after observing the source, minus acquisition cost.
    A positive value means acquiring the source is preferable to acting now.
    """
    utility_now = best_decision_utility(prior_suspicious, costs)
    utility_after = expected_utility_after_evidence(prior_suspicious, source, costs)
    return utility_after - utility_now - source.cost


def decision_loss(
    state: State,
    decision: Decision,
    costs: DecisionCosts,
) -> float:
    """Return realized loss for a terminal decision."""
    if state is State.SUSPICIOUS and decision is Decision.APPROVE:
        return costs.false_approve
    if state is State.LEGITIMATE and decision is Decision.REJECT:
        return costs.false_reject
    return 0.0


def default_sources() -> tuple[EvidenceSource, ...]:
    """Return synthetic evidence sources used in the benchmark.

    The numbers are deliberately synthetic and should not be interpreted as
    empirical estimates for real fraud or finance systems.
    """
    return (
        EvidenceSource("history", sensitivity=0.70, specificity=0.80, cost=0.05),
        EvidenceSource("identity", sensitivity=0.80, specificity=0.90, cost=0.10),
        EvidenceSource("device", sensitivity=0.85, specificity=0.85, cost=0.08),
        EvidenceSource("fraud_signal", sensitivity=0.95, specificity=0.95, cost=0.50),
        EvidenceSource("manual_review", sensitivity=0.99, specificity=0.99, cost=3.00),
    )
