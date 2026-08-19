"""Transparent active-inference planning for the sequential evidence task.

This module keeps the generative model and planning assumptions explicit. It
implements a small factorized A/B/C/D-style model plus two negative-EFE planners:

* ``standard_efe`` scores open-loop query sequences using predicted beliefs that
  are propagated without conditioning on anticipated observations.
* ``sophisticated_efe`` recursively branches on anticipated observations and
  updates beliefs before selecting the next action.

The implementation is specialized to the static hidden-state evidence task and
is intentionally not a drop-in reimplementation of ``pymdp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
import math

from epistemic_action.evidence import (
    DecisionCosts,
    EvidenceSource,
    Signal,
    State,
    posterior_suspicious,
    probability_signal,
)
from epistemic_action.sequential import (
    TerminalAction,
    best_terminal_action,
    expected_terminal_utility,
)

_TOLERANCE = 1e-12

_CONTEXTS = ("idle", "screen", "review", "approve", "reject", "escalate")
_CONTROLS = ("screen", "review", "approve", "reject", "escalate")
_EVIDENCE_OUTCOMES = ("none", "clear", "flagged")
_TERMINAL_OUTCOMES = (
    "none",
    "correct",
    "false_approve",
    "false_reject",
    "escalated",
)
_COST_OUTCOMES = ("none", "screen", "review")


@dataclass(frozen=True, slots=True)
class ABCDModel:
    """Factorized discrete generative model for the benchmark.

    The hidden transaction state is static. A second controlled context factor
    records which query or terminal action is currently active, allowing the
    observation likelihood to remain action-independent once context is part of
    the hidden-state representation. Source availability is fully observed task
    memory and is enforced by the planner's action mask rather than represented
    as an additional hidden factor.
    """

    contexts: tuple[str, ...]
    controls: tuple[str, ...]
    evidence_outcomes: tuple[str, ...]
    terminal_outcomes: tuple[str, ...]
    cost_outcomes: tuple[str, ...]
    A_evidence: tuple[tuple[tuple[float, ...], ...], ...]
    A_terminal: tuple[tuple[tuple[float, ...], ...], ...]
    A_cost: tuple[tuple[tuple[float, ...], ...], ...]
    B_transaction: tuple[tuple[tuple[float, ...], ...], ...]
    B_context: tuple[tuple[tuple[float, ...], ...], ...]
    C_evidence: tuple[float, ...]
    C_terminal: tuple[float, ...]
    C_cost: tuple[float, ...]
    D_transaction: tuple[float, ...]
    D_context: tuple[float, ...]


def _freeze3(values: list[list[list[float]]]) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Convert a mutable rank-3 list to nested tuples."""
    return tuple(tuple(tuple(column) for column in row) for row in values)


def build_abcd_model(
    *,
    prior_suspicious: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    preference_precision: float,
) -> ABCDModel:
    """Build an A/B/C/D-style model for the two-source sequential task.

    ``preference_precision`` maps task utility into log-preference units. A
    larger value makes instrumental outcomes dominate the epistemic term.
    """
    if not 0.0 <= prior_suspicious <= 1.0:
        raise ValueError("prior_suspicious must be in [0.0, 1.0]")
    if escalation_cost < 0.0:
        raise ValueError("escalation_cost must be non-negative")
    if preference_precision <= 0.0:
        raise ValueError("preference_precision must be positive")

    by_name = {source.name: source for source in sources}
    if set(by_name) != {"screen", "review"}:
        raise ValueError("A/B/C/D benchmark requires exactly 'screen' and 'review' sources")

    n_states = 2
    n_contexts = len(_CONTEXTS)
    n_controls = len(_CONTROLS)

    a_evidence = [
        [[0.0 for _ in range(n_contexts)] for _ in range(n_states)]
        for _ in _EVIDENCE_OUTCOMES
    ]
    a_terminal = [
        [[0.0 for _ in range(n_contexts)] for _ in range(n_states)]
        for _ in _TERMINAL_OUTCOMES
    ]
    a_cost = [
        [[0.0 for _ in range(n_contexts)] for _ in range(n_states)]
        for _ in _COST_OUTCOMES
    ]

    states = (State.LEGITIMATE, State.SUSPICIOUS)
    for state_index, state in enumerate(states):
        for context_index, context in enumerate(_CONTEXTS):
            if context in by_name:
                source = by_name[context]
                a_evidence[1][state_index][context_index] = source.probability(
                    Signal.CLEAR,
                    state,
                )
                a_evidence[2][state_index][context_index] = source.probability(
                    Signal.FLAGGED,
                    state,
                )
            else:
                a_evidence[0][state_index][context_index] = 1.0

            if context == "approve":
                outcome = "correct" if state is State.LEGITIMATE else "false_approve"
            elif context == "reject":
                outcome = "correct" if state is State.SUSPICIOUS else "false_reject"
            elif context == "escalate":
                outcome = "escalated"
            else:
                outcome = "none"
            a_terminal[_TERMINAL_OUTCOMES.index(outcome)][state_index][context_index] = 1.0

            cost_outcome = context if context in {"screen", "review"} else "none"
            a_cost[_COST_OUTCOMES.index(cost_outcome)][state_index][context_index] = 1.0

    b_transaction = [
        [[0.0 for _ in range(n_controls)] for _ in range(n_states)]
        for _ in range(n_states)
    ]
    for control_index in range(n_controls):
        for state_index in range(n_states):
            b_transaction[state_index][state_index][control_index] = 1.0

    b_context = [
        [[0.0 for _ in range(n_controls)] for _ in range(n_contexts)]
        for _ in range(n_contexts)
    ]
    for control_index, control in enumerate(_CONTROLS):
        next_context = _CONTEXTS.index(control)
        for current_context in range(n_contexts):
            b_context[next_context][current_context][control_index] = 1.0

    screen = by_name["screen"]
    review = by_name["review"]
    beta = preference_precision

    return ABCDModel(
        contexts=_CONTEXTS,
        controls=_CONTROLS,
        evidence_outcomes=_EVIDENCE_OUTCOMES,
        terminal_outcomes=_TERMINAL_OUTCOMES,
        cost_outcomes=_COST_OUTCOMES,
        A_evidence=_freeze3(a_evidence),
        A_terminal=_freeze3(a_terminal),
        A_cost=_freeze3(a_cost),
        B_transaction=_freeze3(b_transaction),
        B_context=_freeze3(b_context),
        C_evidence=(0.0, 0.0, 0.0),
        C_terminal=(
            0.0,
            0.0,
            -beta * costs.false_approve,
            -beta * costs.false_reject,
            -beta * escalation_cost,
        ),
        C_cost=(0.0, -beta * screen.cost, -beta * review.cost),
        D_transaction=(1.0 - prior_suspicious, prior_suspicious),
        D_context=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _binary_entropy_nats(probability: float) -> float:
    """Return binary entropy in nats."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0.0, 1.0]")
    if probability <= 0.0 or probability >= 1.0:
        return 0.0

    complement = 1.0 - probability
    return -(
        probability * math.log(probability)
        + complement * math.log(complement)
    )


@lru_cache(maxsize=None)
def expected_state_information_gain(
    prior_suspicious: float,
    source: EvidenceSource,
) -> float:
    """Return expected Bayesian surprise about the hidden state, in nats."""
    prior_entropy = _binary_entropy_nats(prior_suspicious)
    posterior_entropy = 0.0

    for signal in Signal:
        signal_probability = probability_signal(prior_suspicious, source, signal)
        if signal_probability <= 0.0:
            continue
        posterior = posterior_suspicious(prior_suspicious, source, signal)
        posterior_entropy += signal_probability * _binary_entropy_nats(posterior)

    return max(0.0, prior_entropy - posterior_entropy)


def _terminal_preference_score(
    prior_suspicious: float,
    action: TerminalAction,
    costs: DecisionCosts,
    escalation_cost: float,
    preference_precision: float,
) -> float:
    """Return expected log-preference score for one terminal action."""
    return preference_precision * expected_terminal_utility(
        prior_suspicious,
        action,
        costs,
        escalation_cost,
    )


def standard_policy_score(
    prior_suspicious: float,
    query_sequence: tuple[EvidenceSource, ...],
    terminal_action: TerminalAction,
    costs: DecisionCosts,
    escalation_cost: float,
    preference_precision: float,
) -> float:
    """Return a specialized open-loop negative-EFE score.

    The predictive hidden-state belief remains unchanged across anticipated query
    steps. This mirrors the standard mean-field planning distinction we want to
    test in the static-state benchmark.
    """
    score = _terminal_preference_score(
        prior_suspicious,
        terminal_action,
        costs,
        escalation_cost,
        preference_precision,
    )

    for source in query_sequence:
        score -= preference_precision * source.cost
        score += expected_state_information_gain(prior_suspicious, source)

    return score


@lru_cache(maxsize=None)
def _best_standard_policy(
    prior_suspicious: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    preference_precision: float,
    steps_remaining: int,
) -> tuple[tuple[EvidenceSource, ...], TerminalAction, float]:
    """Return the highest-scoring open-loop policy."""
    if steps_remaining < 0:
        raise ValueError("steps_remaining must be non-negative")

    terminal_action = best_terminal_action(prior_suspicious, costs, escalation_cost)
    best_sequence: tuple[EvidenceSource, ...] = ()
    best_action = terminal_action
    best_score = standard_policy_score(
        prior_suspicious,
        (),
        terminal_action,
        costs,
        escalation_cost,
        preference_precision,
    )

    max_queries = min(steps_remaining, len(sources))
    for length in range(1, max_queries + 1):
        for sequence in itertools.permutations(sources, length):
            for action in TerminalAction:
                score = standard_policy_score(
                    prior_suspicious,
                    sequence,
                    action,
                    costs,
                    escalation_cost,
                    preference_precision,
                )
                if score > best_score + _TOLERANCE:
                    best_sequence = sequence
                    best_action = action
                    best_score = score

    return best_sequence, best_action, best_score


def select_standard_efe_source(
    prior_suspicious: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    preference_precision: float,
    steps_remaining: int,
) -> EvidenceSource | None:
    """Return the first query of the best open-loop negative-EFE policy."""
    sequence, _, _ = _best_standard_policy(
        prior_suspicious,
        sources,
        costs,
        escalation_cost,
        preference_precision,
        steps_remaining,
    )
    return sequence[0] if sequence else None


@lru_cache(maxsize=None)
def sophisticated_efe_value(
    prior_suspicious: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    preference_precision: float,
    steps_remaining: int,
) -> float:
    """Return a recursive, observation-contingent negative-EFE value.

    Future beliefs are updated under each anticipated observation before the
    next action is selected. This is a transparent sophisticated-inference-style
    recursion, specialized to this benchmark.
    """
    if steps_remaining < 0:
        raise ValueError("steps_remaining must be non-negative")

    best_action = best_terminal_action(prior_suspicious, costs, escalation_cost)
    best_value = _terminal_preference_score(
        prior_suspicious,
        best_action,
        costs,
        escalation_cost,
        preference_precision,
    )
    if steps_remaining == 0 or not sources:
        return best_value

    for source_index, source in enumerate(sources):
        remaining = sources[:source_index] + sources[source_index + 1 :]
        value = (
            -preference_precision * source.cost
            + expected_state_information_gain(prior_suspicious, source)
        )
        for signal in Signal:
            signal_probability = probability_signal(prior_suspicious, source, signal)
            if signal_probability <= 0.0:
                continue
            posterior = posterior_suspicious(prior_suspicious, source, signal)
            value += signal_probability * sophisticated_efe_value(
                posterior,
                remaining,
                costs,
                escalation_cost,
                preference_precision,
                steps_remaining - 1,
            )

        if value > best_value + _TOLERANCE:
            best_value = value

    return best_value


def select_sophisticated_efe_source(
    prior_suspicious: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    preference_precision: float,
    steps_remaining: int,
) -> EvidenceSource | None:
    """Choose the first query under recursive sophisticated EFE."""
    if steps_remaining < 0:
        raise ValueError("steps_remaining must be non-negative")
    if steps_remaining == 0 or not sources:
        return None

    best_action = best_terminal_action(prior_suspicious, costs, escalation_cost)
    best_value = _terminal_preference_score(
        prior_suspicious,
        best_action,
        costs,
        escalation_cost,
        preference_precision,
    )
    best_source: EvidenceSource | None = None

    for source_index, source in enumerate(sources):
        remaining = sources[:source_index] + sources[source_index + 1 :]
        value = (
            -preference_precision * source.cost
            + expected_state_information_gain(prior_suspicious, source)
        )
        for signal in Signal:
            signal_probability = probability_signal(prior_suspicious, source, signal)
            if signal_probability <= 0.0:
                continue
            posterior = posterior_suspicious(prior_suspicious, source, signal)
            value += signal_probability * sophisticated_efe_value(
                posterior,
                remaining,
                costs,
                escalation_cost,
                preference_precision,
                steps_remaining - 1,
            )

        if value > best_value + _TOLERANCE:
            best_value = value
            best_source = source

    return best_source
