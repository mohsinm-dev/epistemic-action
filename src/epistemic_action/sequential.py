"""Finite-horizon evidence acquisition under asymmetric decision costs."""

from __future__ import annotations

from enum import Enum

from epistemic_action.evidence import (
    Decision,
    DecisionCosts,
    EvidenceSource,
    Signal,
    State,
    decision_loss,
    expected_decision_utility,
    expected_information_gain,
    posterior_suspicious,
    probability_signal,
)


_TOLERANCE = 1e-12


class TerminalAction(Enum):
    """Terminal action available after evidence acquisition stops."""

    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


def expected_terminal_utility(
    prior_suspicious: float,
    action: TerminalAction,
    costs: DecisionCosts,
    escalation_cost: float,
) -> float:
    """Return expected utility of a terminal action."""
    if escalation_cost < 0.0:
        raise ValueError("escalation_cost must be non-negative")

    if action is TerminalAction.APPROVE:
        return expected_decision_utility(prior_suspicious, Decision.APPROVE, costs)
    if action is TerminalAction.REJECT:
        return expected_decision_utility(prior_suspicious, Decision.REJECT, costs)
    return -escalation_cost


def best_terminal_action(
    prior_suspicious: float,
    costs: DecisionCosts,
    escalation_cost: float,
) -> TerminalAction:
    """Choose approve, reject, or escalate by maximum expected utility."""
    actions = (
        TerminalAction.APPROVE,
        TerminalAction.REJECT,
        TerminalAction.ESCALATE,
    )
    return max(
        actions,
        key=lambda action: expected_terminal_utility(
            prior_suspicious,
            action,
            costs,
            escalation_cost,
        ),
    )


def best_terminal_utility(
    prior_suspicious: float,
    costs: DecisionCosts,
    escalation_cost: float,
) -> float:
    """Return expected utility of the best terminal action."""
    action = best_terminal_action(prior_suspicious, costs, escalation_cost)
    return expected_terminal_utility(
        prior_suspicious,
        action,
        costs,
        escalation_cost,
    )


def realized_terminal_loss(
    state: State,
    action: TerminalAction,
    costs: DecisionCosts,
    escalation_cost: float,
) -> float:
    """Return realized terminal loss for one episode."""
    if action is TerminalAction.ESCALATE:
        return escalation_cost
    decision = Decision.APPROVE if action is TerminalAction.APPROVE else Decision.REJECT
    return decision_loss(state, decision, costs)


def expected_terminal_utility_after_source(
    prior_suspicious: float,
    source: EvidenceSource,
    costs: DecisionCosts,
    escalation_cost: float,
) -> float:
    """Return one-step expected terminal utility after querying ``source``."""
    expected_utility = -source.cost

    for signal in Signal:
        signal_probability = probability_signal(prior_suspicious, source, signal)
        if signal_probability <= 0.0:
            continue
        posterior = posterior_suspicious(prior_suspicious, source, signal)
        expected_utility += signal_probability * best_terminal_utility(
            posterior,
            costs,
            escalation_cost,
        )

    return expected_utility


def myopic_net_value(
    prior_suspicious: float,
    source: EvidenceSource,
    costs: DecisionCosts,
    escalation_cost: float,
) -> float:
    """Return one-step value of querying instead of stopping now."""
    return expected_terminal_utility_after_source(
        prior_suspicious,
        source,
        costs,
        escalation_cost,
    ) - best_terminal_utility(prior_suspicious, costs, escalation_cost)


def select_myopic_source(
    prior_suspicious: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
) -> EvidenceSource | None:
    """Choose the source with highest positive one-step decision value."""
    best_source: EvidenceSource | None = None
    best_value = 0.0

    for source in sources:
        value = myopic_net_value(
            prior_suspicious,
            source,
            costs,
            escalation_cost,
        )
        if value > best_value + _TOLERANCE:
            best_value = value
            best_source = source

    return best_source


def select_information_gain_source(
    prior_suspicious: float,
    sources: tuple[EvidenceSource, ...],
) -> EvidenceSource | None:
    """Choose the source with maximum expected entropy reduction, ignoring cost."""
    best_source: EvidenceSource | None = None
    best_gain = 0.0

    for source in sources:
        gain = expected_information_gain(prior_suspicious, source)
        if gain > best_gain + _TOLERANCE:
            best_gain = gain
            best_source = source

    return best_source


def _query_value(
    prior_suspicious: float,
    source_index: int,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    steps_remaining: int,
) -> float:
    """Return expected utility of querying one source then planning optimally."""
    source = sources[source_index]
    remaining_sources = sources[:source_index] + sources[source_index + 1 :]
    expected_utility = -source.cost

    for signal in Signal:
        signal_probability = probability_signal(prior_suspicious, source, signal)
        if signal_probability <= 0.0:
            continue
        posterior = posterior_suspicious(prior_suspicious, source, signal)
        expected_utility += signal_probability * optimal_value(
            posterior,
            remaining_sources,
            costs,
            escalation_cost,
            steps_remaining - 1,
        )

    return expected_utility


def optimal_value(
    prior_suspicious: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    steps_remaining: int,
) -> float:
    """Return exact finite-horizon value under the stated evidence model."""
    if steps_remaining < 0:
        raise ValueError("steps_remaining must be non-negative")

    best_value = best_terminal_utility(prior_suspicious, costs, escalation_cost)
    if steps_remaining == 0 or not sources:
        return best_value

    for source_index in range(len(sources)):
        query_value = _query_value(
            prior_suspicious,
            source_index,
            sources,
            costs,
            escalation_cost,
            steps_remaining,
        )
        if query_value > best_value + _TOLERANCE:
            best_value = query_value

    return best_value


def select_lookahead_source(
    prior_suspicious: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    steps_remaining: int,
) -> EvidenceSource | None:
    """Choose the first query from the exact finite-horizon optimal policy."""
    if steps_remaining < 0:
        raise ValueError("steps_remaining must be non-negative")
    if steps_remaining == 0 or not sources:
        return None

    best_value = best_terminal_utility(prior_suspicious, costs, escalation_cost)
    best_source: EvidenceSource | None = None

    for source_index, source in enumerate(sources):
        query_value = _query_value(
            prior_suspicious,
            source_index,
            sources,
            costs,
            escalation_cost,
            steps_remaining,
        )
        if query_value > best_value + _TOLERANCE:
            best_value = query_value
            best_source = source

    return best_source


def sequential_sources() -> tuple[EvidenceSource, ...]:
    """Return sources that expose genuine non-myopic option value.

    At the default prior and decision costs, neither source has positive one-step
    value. A two-step planner still buys the cheap screen because a clear result
    lets it stop while a flagged result can make the stronger review worthwhile.
    """
    return (
        EvidenceSource("screen", sensitivity=0.70, specificity=0.70, cost=0.05),
        EvidenceSource("review", sensitivity=0.95, specificity=0.95, cost=0.20),
    )
