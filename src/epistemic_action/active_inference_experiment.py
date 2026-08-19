"""Compare Bayesian and active-inference planners on the same sequential task."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
from pathlib import Path
import random
from statistics import fmean

from epistemic_action.active_inference import (
    select_sophisticated_efe_source,
    select_standard_efe_source,
)
from epistemic_action.evidence import DecisionCosts, EvidenceSource, State, posterior_suspicious
from epistemic_action.sequential import (
    TerminalAction,
    best_terminal_action,
    realized_terminal_loss,
    select_lookahead_source,
    sequential_sources,
)


@dataclass(frozen=True, slots=True)
class Episode:
    """One shared latent state for all policies."""

    state: State


@dataclass(frozen=True, slots=True)
class Result:
    """Aggregate outcome for one planner and preference precision."""

    policy: str
    preference_precision: float
    prior_suspicious: float
    horizon: int
    episodes: int
    accuracy: float
    mean_terminal_loss: float
    mean_evidence_cost: float
    mean_total_loss: float
    mean_queries: float
    escalation_rate: float


def _sample_episodes(count: int, prior_suspicious: float, seed: int) -> list[Episode]:
    """Sample latent states once so every planner sees identical worlds."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 <= prior_suspicious <= 1.0:
        raise ValueError("prior_suspicious must be in [0.0, 1.0]")

    rng = random.Random(seed)
    return [
        Episode(State.SUSPICIOUS if rng.random() < prior_suspicious else State.LEGITIMATE)
        for _ in range(count)
    ]


def _select_source(
    *,
    policy: str,
    posterior: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    preference_precision: float,
    steps_remaining: int,
) -> EvidenceSource | None:
    """Select the next query under one planner."""
    if policy == "bayes_lookahead":
        return select_lookahead_source(
            posterior,
            sources,
            costs,
            escalation_cost,
            steps_remaining,
        )
    if policy == "standard_efe":
        return select_standard_efe_source(
            posterior,
            sources,
            costs,
            escalation_cost,
            preference_precision,
            steps_remaining,
        )
    if policy == "sophisticated_efe":
        return select_sophisticated_efe_source(
            posterior,
            sources,
            costs,
            escalation_cost,
            preference_precision,
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
    preference_precision: float,
    horizon: int,
    rng: random.Random,
) -> tuple[float, float, float, int, bool, bool]:
    """Run one planner trajectory and return comparable metrics."""
    posterior = prior_suspicious
    remaining_sources = sources
    evidence_cost = 0.0
    queries = 0

    for step in range(horizon):
        source = _select_source(
            policy=policy,
            posterior=posterior,
            sources=remaining_sources,
            costs=costs,
            escalation_cost=escalation_cost,
            preference_precision=preference_precision,
            steps_remaining=horizon - step,
        )
        if source is None:
            break

        signal = source.sample(episode.state, rng)
        posterior = posterior_suspicious(posterior, source, signal)
        evidence_cost += source.cost
        queries += 1
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
    escalated = terminal_action is TerminalAction.ESCALATE
    total_loss = terminal_loss + evidence_cost
    return terminal_loss, evidence_cost, total_loss, queries, correct, escalated


def run_condition(
    *,
    prior_suspicious: float,
    false_approve_cost: float,
    escalation_cost: float,
    horizon: int,
    preference_precision: float,
    episodes: int,
    seed: int,
) -> list[Result]:
    """Evaluate all planners on identical latent states."""
    if horizon < 0:
        raise ValueError("horizon must be non-negative")

    costs = DecisionCosts(false_approve=false_approve_cost, false_reject=1.0)
    sources = sequential_sources()
    sampled = _sample_episodes(episodes, prior_suspicious, seed)
    policies = ("bayes_lookahead", "standard_efe", "sophisticated_efe")

    results: list[Result] = []
    for policy_index, policy in enumerate(policies):
        rng = random.Random(seed + 10_000 * (policy_index + 1))
        outcomes = [
            _run_episode(
                episode,
                policy=policy,
                prior_suspicious=prior_suspicious,
                sources=sources,
                costs=costs,
                escalation_cost=escalation_cost,
                preference_precision=preference_precision,
                horizon=horizon,
                rng=rng,
            )
            for episode in sampled
        ]

        results.append(
            Result(
                policy=policy,
                preference_precision=preference_precision,
                prior_suspicious=prior_suspicious,
                horizon=horizon,
                episodes=episodes,
                accuracy=fmean(float(outcome[4]) for outcome in outcomes),
                mean_terminal_loss=fmean(outcome[0] for outcome in outcomes),
                mean_evidence_cost=fmean(outcome[1] for outcome in outcomes),
                mean_total_loss=fmean(outcome[2] for outcome in outcomes),
                mean_queries=fmean(outcome[3] for outcome in outcomes),
                escalation_rate=fmean(float(outcome[5]) for outcome in outcomes),
            )
        )

    return results


def run_sweep(
    *,
    precisions: list[float],
    prior_suspicious: float,
    false_approve_cost: float,
    escalation_cost: float,
    horizon: int,
    episodes: int,
    seed: int,
) -> list[Result]:
    """Sweep preference precision while holding the task fixed."""
    results: list[Result] = []
    for index, precision in enumerate(precisions):
        results.extend(
            run_condition(
                prior_suspicious=prior_suspicious,
                false_approve_cost=false_approve_cost,
                escalation_cost=escalation_cost,
                horizon=horizon,
                preference_precision=precision,
                episodes=episodes,
                seed=seed + index,
            )
        )
    return results


def write_csv(results: list[Result], output_path: Path) -> None:
    """Write comparison results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(Result)]

    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "policy": result.policy,
                    "preference_precision": result.preference_precision,
                    "prior_suspicious": result.prior_suspicious,
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


def _parse_float_list(value: str) -> list[float]:
    """Parse comma-separated positive floats."""
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("expected positive numeric values")
    return values


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compare exact Bayesian and EFE planners on one evidence task.",
    )
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--prior", type=float, default=0.05)
    parser.add_argument("--false-approve-cost", type=float, default=5.0)
    parser.add_argument("--escalation-cost", type=float, default=0.40)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument(
        "--precisions",
        type=_parse_float_list,
        default=_parse_float_list("0.5,1,2,5,10,20"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/active_inference.csv"),
    )
    return parser


def main() -> None:
    """Run the comparison sweep."""
    args = _build_parser().parse_args()
    results = run_sweep(
        precisions=args.precisions,
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
