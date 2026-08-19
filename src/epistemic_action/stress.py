"""Stress models for correlated and miscalibrated binary evidence."""

from __future__ import annotations

from dataclasses import dataclass
import random

from epistemic_action.evidence import EvidenceSource, Signal, State


@dataclass(frozen=True, slots=True)
class CorrelatedPair:
    """Two evidence channels with identical marginals and controllable dependence.

    With probability ``correlation`` both channels reuse the same sampled signal.
    Otherwise they are sampled independently. This mixture preserves the source
    marginal distribution while introducing positive dependence.
    """

    source: EvidenceSource
    correlation: float

    def __post_init__(self) -> None:
        """Validate the dependence parameter."""
        if not 0.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must be in [0.0, 1.0]")

    def sample(self, state: State, rng: random.Random) -> tuple[Signal, Signal]:
        """Sample two correlated observations from the latent state."""
        if rng.random() < self.correlation:
            signal = self.source.sample(state, rng)
            return signal, signal
        return self.source.sample(state, rng), self.source.sample(state, rng)


def joint_signal_probability(
    source: EvidenceSource,
    first: Signal,
    second: Signal,
    state: State,
    correlation: float,
) -> float:
    """Return ``P(first, second | state)`` for the shared-noise mixture."""
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [0.0, 1.0]")

    independent = source.probability(first, state) * source.probability(second, state)
    shared = source.probability(first, state) if first is second else 0.0
    return correlation * shared + (1.0 - correlation) * independent


def posterior_from_pair(
    prior_suspicious: float,
    source: EvidenceSource,
    first: Signal,
    second: Signal,
    *,
    correlation: float,
) -> float:
    """Bayes-update ``P(suspicious)`` from two possibly correlated signals."""
    if not 0.0 <= prior_suspicious <= 1.0:
        raise ValueError("prior_suspicious must be in [0.0, 1.0]")

    suspicious_likelihood = joint_signal_probability(
        source,
        first,
        second,
        State.SUSPICIOUS,
        correlation,
    )
    legitimate_likelihood = joint_signal_probability(
        source,
        first,
        second,
        State.LEGITIMATE,
        correlation,
    )

    numerator = suspicious_likelihood * prior_suspicious
    evidence = numerator + legitimate_likelihood * (1.0 - prior_suspicious)
    if evidence <= 0.0:
        raise ValueError("signal pair has zero probability under the model")

    return numerator / evidence


def symmetric_source(name: str, accuracy: float) -> EvidenceSource:
    """Construct a zero-cost source with equal sensitivity and specificity."""
    if not 0.5 <= accuracy <= 1.0:
        raise ValueError("accuracy must be in [0.5, 1.0]")
    return EvidenceSource(
        name=name,
        sensitivity=accuracy,
        specificity=accuracy,
        cost=0.0,
    )
