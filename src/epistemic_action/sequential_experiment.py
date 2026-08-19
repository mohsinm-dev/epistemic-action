"""Compare myopic and finite-horizon evidence acquisition policies."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
from pathlib import Path
import random
from statistics import fmean

from epistemic_action.evidence import DecisionCosts, EvidenceSource, State, posterior_suspicious
from epistemic_action.sequential import (
    TerminalAction,
    best_terminal_action,
    realized_terminal_loss,
    select_information_gain_source,
    select_lookahead_source,
    select_myopic_source,
    sequential_sources,
)


@dataclass(frozen=True, slots=True)
class Episode:
    """One sampled latent state."""

    state: State


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Realized trajectory outcome from one sequential policy."""

    correct: bool
    terminal_action: str
    terminal_loss: float
    evidence_cost: float
    total_loss: float
    queries: int
    queried_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Result:
    """Aggregate metrics for one sequential policy."""

    policy: str
    prior_suspicious: float
    false_approve_cost: float
    escalation_cost: float
    horizon: int
    episodes: int
    accuracy: float
    mean_terminal_loss: float
    mean_evidence_cost: float
    mean_total_loss: float
    mean_queries: float
    escalation_rate: float


def _sample_episodes(*, count: int, suspicious_rate: float, seed: int) -> list[Episode]:
    """Sample shared latent states for all policies."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 <= suspicious_rate <= 1.0:
        raise ValueError("suspicious_rate must be in [0.0, 1.0]")

    rng = random.Random(seed)
    return [
        Episode(State.SUSPICIOUS if rng.random() < suspicious_rate else State.LEGITIMATE)
        for _ in range(count)
    ]


def _select_source(
    *,
    policy: str,
    posterior: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    steps_remaining: int,
) -> EvidenceSource | None:
    """Select the next source under the requested policy."""
    if policy == "greedy":
        return None
    if policy == "information_gain":
        return select_information_gain_source(posterior, sources)
    if policy == "myopic_voi":
        return select_myopic_source(posterior, sources, costs, escalation_cost)
    if policy == "lookahead":
        return select_lookahead_source(
            posterior,
            sources,
            costs,
            escalation_cost,
            steps_remaining,
        )
    raise ValueError(f"unknown policy: {policy}")


def _run_episode(
    episode: Episode,
    *,
    policy: str,
    prior_suspicious: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    horizon: int,
    rng: random.Random,
) -> EpisodeOutcome:
    """Run one finite-horizon evidence-acquisition trajectory."""
    if horizon < 0:
        raise ValueError("horizon must be non-negative")

    posterior = prior_suspicious
    remaining_sources = sources
    evidence_cost = 0.0
    queried_sources: list[str] = []

    for step in range(horizon):
        source = _select_source(
            policy=policy,
            posterior=posterior,
            sources=remaining_sources,
            costs=costs,
            escalation_cost=escalation_cost,
            steps_remaining=horizon - step,
        )
        if source is None:
            break

        signal = source.sample(episode.state, rng)
        posterior = posterior_suspicious(posterior, source, signal)
        evidence_cost += source.cost
        queried_sources.append(source.name)
        remaining_sources = tuple(candidate for candidate in remaining_sources if candidate != source)

    terminal_action = best_terminal_action(posterior, costs, escalation_cost)
    terminal_loss = realized_terminal_loss(
        episode.state,
        terminal_action,
        costs,
        escalation_cost,
    )
    correct = (
        terminal_action is TerminalAction.APPROVE and episode.state is State.LEGITIMATE
    ) or (
        terminal_action is TerminalAction.REJECT and episode.state is State.SUSPICIOUS
    )

    return EpisodeOutcome(
        correct=correct,
        terminal_action=terminal_action.value,
        terminal_loss=terminal_loss,
        evidence_cost=evidence_cost,
        total_loss=terminal_loss + evidence_cost,
        queries=len(queried_sources),
        queried_sources=tuple(queried_sources),
    )


def _summarize(
    *,
    policy: str,
    prior_suspicious: float,
    costs: DecisionCosts,
    escalation_cost: float,
    horizon: int,
    outcomes: list[EpisodeOutcome],
) -> Result:
    """Aggregate trajectory outcomes."""
    return Result(
        policy=policy,
        prior_suspicious=prior_suspicious,
        false_approve_cost=costs.false_approve,
        escalation_cost=escalation_cost,
        horizon=horizon,
        episodes=len(outcomes),
        accuracy=fmean(float(outcome.correct) for outcome in outcomes),
        mean_terminal_loss=fmean(outcome.terminal_loss for outcome in outcomes),
        mean_evidence_cost=fmean(outcome.evidence_cost for outcome in outcomes),
        mean_total_loss=fmean(outcome.total_loss for outcome in outcomes),
        mean_queries=fmean(outcome.queries for outcome in outcomes),
        escalation_rate=fmean(
            float(outcome.terminal_action == TerminalAction.ESCALATE.value)
            for outcome in outcomes
        ),
    )


def run_condition(
    *,
    prior_suspicious: float,
    false_approve_cost: float,
    escalation_cost: float,
    horizon: int,
    episodes: int,
    seed: int,
    sources: tuple[EvidenceSource, ...] | None = None,
) -> list[Result]:
    """Evaluate all sequential policies on identical latent states."""
    if sources is None:
        sources = sequential_sources()

    costs = DecisionCosts(false_approve=false_approve_cost, false_reject=1.0)
    sampled_episodes = _sample_episodes(
        count=episodes,
        suspicious_rate=prior_suspicious,
        seed=seed,
    )
    policies = ("greedy", "information_gain", "myopic_voi", "lookahead")

    results: list[Result] = []
    for policy_index, policy in enumerate(policies):
        policy_rng = random.Random(seed + 10_000 * (policy_index + 1))
        outcomes = [
            _run_episode(
                episode,
                policy=policy,
                prior_suspicious=prior_suspicious,
                sources=sources,
                costs=costs,
                escalation_cost=escalation_cost,
                horizon=horizon,
                rng=policy_rng,
            )
            for episode in sampled_episodes
        ]
        results.append(
            _summarize(
                policy=policy,
                prior_suspicious=prior_suspicious,
                costs=costs,
                escalation_cost=escalation_cost,
                horizon=horizon,
                outcomes=outcomes,
            )
        )

    return results


def write_csv(results: list[Result], output_path: Path) -> None:
    """Write sequential benchmark results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(Result)]

    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "policy": result.policy,
                    "prior_suspicious": result.prior_suspicious,
                    "false_approve_cost": result.false_approve_cost,
                    "escalation_cost": result.escalation_cost,
                    "horizon": result.horizon,
                    "episodes": result.episodes,
                    "accuracy": f"{result.accuracy:.6f}",
                    "mean_terminal_loss": f"{result.mean_terminal_loss:.6f}",
                    "mean_evidence_cost": f"{result.mean_evidence_cost:.6f}",
                    "mean_total_loss": f"{result.mean_total_loss:.6f}",
                    "mean_queries": f"{result.mean_queries:.6f}",
                    "escalation_rate": f"{result.escalation_rate:.6f}",
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compare myopic and finite-horizon evidence acquisition.",
    )
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--prior", type=float, default=0.05)
    parser.add_argument("--false-approve-cost", type=float, default=5.0)
    parser.add_argument("--escalation-cost", type=float, default=0.40)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sequential.csv"),
    )
    return parser


def main() -> None:
    """Run the sequential benchmark and save results."""
    args = _build_parser().parse_args()
    results = run_condition(
        prior_suspicious=args.prior,
        false_approve_cost=args.false_approve_cost,
        escalation_cost=args.escalation_cost,
        horizon=args.horizon,
        episodes=args.episodes,
        seed=args.seed,
    )
    write_csv(results, args.output)
    print(f"wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
