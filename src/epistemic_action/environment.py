"""Binary information-seeking environment used by the experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random


class Door(Enum):
    """The two possible rewarding doors and clue observations."""

    LEFT = "left"
    RIGHT = "right"

    def other(self) -> "Door":
        """Return the opposite door."""
        return Door.RIGHT if self is Door.LEFT else Door.LEFT


@dataclass(frozen=True, slots=True)
class Episode:
    """One latent world state and its pre-sampled noisy clue."""

    rewarding_door: Door
    clue: Door


@dataclass(frozen=True, slots=True)
class Outcome:
    """Observed result from one episode."""

    correct: bool
    used_clue: bool
    net_reward: float


def sample_episode(rng: random.Random, clue_reliability: float) -> Episode:
    """Sample an episode with a uniformly random rewarding door.

    Args:
        rng: Pseudorandom generator used for reproducibility.
        clue_reliability: Probability that the clue identifies the rewarding door.

    Returns:
        A sampled episode.

    Raises:
        ValueError: If ``clue_reliability`` is outside ``[0.5, 1.0]``.
    """
    if not 0.5 <= clue_reliability <= 1.0:
        raise ValueError("clue_reliability must be in [0.5, 1.0]")

    rewarding_door = Door.LEFT if rng.random() < 0.5 else Door.RIGHT
    clue_is_correct = rng.random() < clue_reliability
    clue = rewarding_door if clue_is_correct else rewarding_door.other()
    return Episode(rewarding_door=rewarding_door, clue=clue)


def score_episode(
    episode: Episode,
    choice: Door,
    *,
    used_clue: bool,
    clue_cost: float,
) -> Outcome:
    """Score a door choice, subtracting clue cost when information was requested."""
    if clue_cost < 0.0:
        raise ValueError("clue_cost must be non-negative")

    correct = choice is episode.rewarding_door
    reward = 1.0 if correct else 0.0
    if used_clue:
        reward -= clue_cost

    return Outcome(correct=correct, used_clue=used_clue, net_reward=reward)
