"""Tests for belief updates and information-seeking decisions."""

import math

from epistemic_action.agents import (
    EpistemicAgent,
    expected_information_gain,
    posterior_left,
)
from epistemic_action.environment import Door


def test_uninformative_clue_has_zero_information_gain() -> None:
    """A chance-level clue should not reduce uncertainty."""
    assert math.isclose(expected_information_gain(0.5, 0.5), 0.0, abs_tol=1e-12)


def test_perfect_clue_removes_binary_uncertainty() -> None:
    """A perfect clue should provide one bit from a uniform prior."""
    assert math.isclose(expected_information_gain(0.5, 1.0), 1.0, abs_tol=1e-12)


def test_bayesian_update_tracks_clue_reliability() -> None:
    """With a uniform prior, the posterior should match clue reliability."""
    assert math.isclose(posterior_left(0.5, Door.LEFT, 0.8), 0.8)
    assert math.isclose(posterior_left(0.5, Door.RIGHT, 0.8), 0.2)


def test_epistemic_agent_rejects_useless_clue() -> None:
    """The agent should not pay for a clue carrying no information."""
    agent = EpistemicAgent(information_weight=1.0)
    assert not agent.should_request_clue(clue_reliability=0.5, clue_cost=0.01)


def test_epistemic_agent_uses_reliable_free_clue() -> None:
    """A reliable free clue should dominate acting immediately."""
    agent = EpistemicAgent(information_weight=1.0)
    assert agent.should_request_clue(clue_reliability=0.9, clue_cost=0.0)


def test_high_clue_cost_can_remove_information_seeking() -> None:
    """Information seeking should stop when its total value is below its cost."""
    agent = EpistemicAgent(information_weight=1.0)
    assert not agent.should_request_clue(clue_reliability=0.6, clue_cost=0.5)
