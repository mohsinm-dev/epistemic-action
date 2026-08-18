"""Agents and belief calculations for the binary clue experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from epistemic_action.environment import Door, Episode, Outcome, score_episode


_EPSILON = 1e-12


def binary_entropy(probability: float) -> float:
    """Return binary entropy in bits."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0.0, 1.0]")

    if probability <= _EPSILON or probability >= 1.0 - _EPSILON:
        return 0.0

    complement = 1.0 - probability
    return -(
        probability * math.log2(probability)
        + complement * math.log2(complement)
    )


def posterior_left(
    prior_left: float,
    observation: Door,
    clue_reliability: float,
) -> float:
    """Update ``P(left is rewarding)`` after observing a noisy clue."""
    if not 0.0 <= prior_left <= 1.0:
        raise ValueError("prior_left must be in [0.0, 1.0]")
    if not 0.5 <= clue_reliability <= 1.0:
        raise ValueError("clue_reliability must be in [0.5, 1.0]")

    if observation is Door.LEFT:
        likelihood_left = clue_reliability
        likelihood_right = 1.0 - clue_reliability
    else:
        likelihood_left = 1.0 - clue_reliability
        likelihood_right = clue_reliability

    numerator = likelihood_left * prior_left
    evidence = numerator + likelihood_right * (1.0 - prior_left)
    if evidence <= 0.0:
        raise ValueError("observation has zero probability under the model")

    return numerator / evidence


def expected_information_gain(prior_left: float, clue_reliability: float) -> float:
    """Return expected entropy reduction from requesting the clue, in bits."""
    if not 0.0 <= prior_left <= 1.0:
        raise ValueError("prior_left must be in [0.0, 1.0]")
    if not 0.5 <= clue_reliability <= 1.0:
        raise ValueError("clue_reliability must be in [0.5, 1.0]")

    prior_entropy = binary_entropy(prior_left)
    expected_posterior_entropy = 0.0

    for observation in Door:
        if observation is Door.LEFT:
            probability_observation = (
                prior_left * clue_reliability
                + (1.0 - prior_left) * (1.0 - clue_reliability)
            )
        else:
            probability_observation = (
                prior_left * (1.0 - clue_reliability)
                + (1.0 - prior_left) * clue_reliability
            )

        posterior = posterior_left(prior_left, observation, clue_reliability)
        expected_posterior_entropy += probability_observation * binary_entropy(posterior)

    return prior_entropy - expected_posterior_entropy


def expected_reward_after_clue(prior_left: float, clue_reliability: float) -> float:
    """Return expected task reward when observing the clue before choosing a door."""
    expected_reward = 0.0

    for observation in Door:
        if observation is Door.LEFT:
            probability_observation = (
                prior_left * clue_reliability
                + (1.0 - prior_left) * (1.0 - clue_reliability)
            )
        else:
            probability_observation = (
                prior_left * (1.0 - clue_reliability)
                + (1.0 - prior_left) * clue_reliability
            )

        posterior = posterior_left(prior_left, observation, clue_reliability)
        expected_reward += probability_observation * max(posterior, 1.0 - posterior)

    return expected_reward


def _best_door(probability_left: float, rng: random.Random) -> Door:
    """Choose the most probable door and break exact ties reproducibly."""
    if probability_left > 0.5:
        return Door.LEFT
    if probability_left < 0.5:
        return Door.RIGHT
    return Door.LEFT if rng.random() < 0.5 else Door.RIGHT


@dataclass(frozen=True, slots=True)
class GreedyAgent:
    """Choose the door with highest current expected reward; never inspect the clue."""

    prior_left: float = 0.5

    def run_episode(
        self,
        episode: Episode,
        *,
        clue_reliability: float,
        clue_cost: float,
        rng: random.Random,
    ) -> Outcome:
        """Run one episode without gathering information."""
        del clue_reliability
        choice = _best_door(self.prior_left, rng)
        return score_episode(
            episode,
            choice,
            used_clue=False,
            clue_cost=clue_cost,
        )


@dataclass(frozen=True, slots=True)
class EpistemicAgent:
    """Trade task utility against expected information gain before acting.

    This is intentionally an active-inference-inspired planner rather than a full
    variational active inference implementation. The information term makes the
    exploration mechanism explicit and inspectable.
    """

    information_weight: float = 1.0
    prior_left: float = 0.5

    def __post_init__(self) -> None:
        """Validate agent hyperparameters."""
        if self.information_weight < 0.0:
            raise ValueError("information_weight must be non-negative")

    def should_request_clue(self, clue_reliability: float, clue_cost: float) -> bool:
        """Return whether the clue policy dominates acting immediately."""
        direct_value = max(self.prior_left, 1.0 - self.prior_left)
        clue_value = (
            expected_reward_after_clue(self.prior_left, clue_reliability)
            - clue_cost
            + self.information_weight
            * expected_information_gain(self.prior_left, clue_reliability)
        )
        return clue_value > direct_value

    def run_episode(
        self,
        episode: Episode,
        *,
        clue_reliability: float,
        clue_cost: float,
        rng: random.Random,
    ) -> Outcome:
        """Run one episode, optionally observing the clue before choosing."""
        use_clue = self.should_request_clue(clue_reliability, clue_cost)
        belief_left = self.prior_left

        if use_clue:
            belief_left = posterior_left(
                self.prior_left,
                episode.clue,
                clue_reliability,
            )

        choice = _best_door(belief_left, rng)
        return score_episode(
            episode,
            choice,
            used_clue=use_clue,
            clue_cost=clue_cost,
        )
