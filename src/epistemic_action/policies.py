"""One-step evidence-acquisition policies for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import random

from epistemic_action.evidence import (
    DecisionCosts,
    EvidenceSource,
    expected_information_gain,
    expected_value_of_information,
)


@dataclass(frozen=True, slots=True)
class GreedyPolicy:
    """Act immediately without acquiring evidence."""

    name: str = "greedy"

    def select_source(
        self,
        *,
        prior_suspicious: float,
        sources: tuple[EvidenceSource, ...],
        costs: DecisionCosts,
        rng: random.Random,
    ) -> EvidenceSource | None:
        """Return no source because this policy never gathers evidence."""
        del prior_suspicious, sources, costs, rng
        return None


@dataclass(frozen=True, slots=True)
class RandomEvidencePolicy:
    """Acquire one uniformly random source as a sanity baseline."""

    name: str = "random"

    def select_source(
        self,
        *,
        prior_suspicious: float,
        sources: tuple[EvidenceSource, ...],
        costs: DecisionCosts,
        rng: random.Random,
    ) -> EvidenceSource | None:
        """Choose one source uniformly at random."""
        del prior_suspicious, costs
        if not sources:
            return None
        return rng.choice(sources)


@dataclass(frozen=True, slots=True)
class InformationGainPolicy:
    """Acquire the source with the largest expected entropy reduction."""

    name: str = "information_gain"

    def select_source(
        self,
        *,
        prior_suspicious: float,
        sources: tuple[EvidenceSource, ...],
        costs: DecisionCosts,
        rng: random.Random,
    ) -> EvidenceSource | None:
        """Select the most informative source, intentionally ignoring its cost."""
        del costs, rng
        if not sources:
            return None

        return max(
            sources,
            key=lambda source: expected_information_gain(prior_suspicious, source),
        )


@dataclass(frozen=True, slots=True)
class ValueOfInformationPolicy:
    """Acquire evidence only when its expected decision value exceeds its cost."""

    name: str = "value_of_information"

    def select_source(
        self,
        *,
        prior_suspicious: float,
        sources: tuple[EvidenceSource, ...],
        costs: DecisionCosts,
        rng: random.Random,
    ) -> EvidenceSource | None:
        """Choose the source with highest positive one-step net value."""
        del rng
        if not sources:
            return None

        source_values = [
            (expected_value_of_information(prior_suspicious, source, costs), source)
            for source in sources
        ]
        best_value, best_source = max(source_values, key=lambda item: item[0])
        return best_source if best_value > 0.0 else None
