"""Benchmark one-step evidence acquisition under asymmetric decision costs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
from pathlib import Path
import random
from statistics import fmean
from typing import Protocol

from epistemic_action.evidence import (
    Decision,
    DecisionCosts,
    EvidenceSource,
    State,
    best_decision,
    decision_loss,
    default_sources,
    posterior_suspicious,
)
from epistemic_action.policies import (
    GreedyPolicy,
    InformationGainPolicy,
    RandomEvidencePolicy,
    ValueOfInformationPolicy,
)


class EvidencePolicy(Protocol):
    """Protocol implemented by one-step evidence-selection policies."""

    name: str

    def select_source(
        self,
        *,
        prior_suspicious: float,
        sources: tuple[EvidenceSource, ...],
        costs: DecisionCosts,
        rng: random.Random,
    ) -> EvidenceSource | None:
        """Choose at most one evidence source before the terminal decision."""


@dataclass(frozen=True, slots=True)
class Episode:
    """One sampled latent transaction state."""

    state: State


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Realized outcome from one policy on one episode."""

    correct: bool
    decision_loss: float
    evidence_cost: float
    total_loss: float
    evidence_source: str | None


@dataclass(frozen=True, slots=True)
class Result:
    """Aggregate metrics for one policy under one condition."""

    policy: str
    prior_suspicious: float
    false_approve_cost: float
    episodes: int
    accuracy: float
    mean_decision_loss: float
    mean_evidence_cost: float
    mean_total_loss: float
    evidence_rate: float
    manual_review_rate: float


def _sample_episodes(*, count: int, suspicious_rate: float, seed: int) -> list[Episode]:
    """Sample a shared latent-state sequence for all policies."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 <= suspicious_rate <= 1.0:
        raise ValueError("suspicious_rate must be in [0.0, 1.0]")

    rng = random.Random(seed)
    return [
        Episode(
            State.SUSPICIOUS if rng.random() < suspicious_rate else State.LEGITIMATE
        )
        for _ in range(count)
    ]


def _run_episode(
    episode: Episode,
    *,
    policy: EvidencePolicy,
    prior_suspicious: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    rng: random.Random,
) -> EpisodeOutcome:
    """Run a one-step evidence policy and terminal decision."""
    source = policy.select_source(
        prior_suspicious=prior_suspicious,
        sources=sources,
        costs=costs,
        rng=rng,
    )

    posterior = prior_suspicious
    evidence_cost = 0.0
    evidence_source: str | None = None

    if source is not None:
        signal = source.sample(episode.state, rng)
        posterior = posterior_suspicious(prior_suspicious, source, signal)
        evidence_cost = source.cost
        evidence_source = source.name

    decision = best_decision(posterior, costs)
    realized_decision_loss = decision_loss(episode.state, decision, costs)
    correct = (
        decision is Decision.APPROVE and episode.state is State.LEGITIMATE
    ) or (
        decision is Decision.REJECT and episode.state is State.SUSPICIOUS
    )

    return EpisodeOutcome(
        correct=correct,
        decision_loss=realized_decision_loss,
        evidence_cost=evidence_cost,
        total_loss=realized_decision_loss + evidence_cost,
        evidence_source=evidence_source,
    )


def _summarize(
    *,
    policy_name: str,
    prior_suspicious: float,
    costs: DecisionCosts,
    outcomes: list[EpisodeOutcome],
) -> Result:
    """Aggregate outcomes into decision and acquisition metrics."""
    return Result(
        policy=policy_name,
        prior_suspicious=prior_suspicious,
        false_approve_cost=costs.false_approve,
        episodes=len(outcomes),
        accuracy=fmean(float(outcome.correct) for outcome in outcomes),
        mean_decision_loss=fmean(outcome.decision_loss for outcome in outcomes),
        mean_evidence_cost=fmean(outcome.evidence_cost for outcome in outcomes),
        mean_total_loss=fmean(outcome.total_loss for outcome in outcomes),
        evidence_rate=fmean(float(outcome.evidence_source is not None) for outcome in outcomes),
        manual_review_rate=fmean(
            float(outcome.evidence_source == "manual_review") for outcome in outcomes
        ),
    )


def run_condition(
    *,
    prior_suspicious: float,
    false_approve_cost: float,
    episodes: int,
    seed: int,
    sources: tuple[EvidenceSource, ...] | None = None,
) -> list[Result]:
    """Evaluate all policies on identical latent states."""
    if sources is None:
        sources = default_sources()

    costs = DecisionCosts(false_approve=false_approve_cost, false_reject=1.0)
    sampled_episodes = _sample_episodes(
        count=episodes,
        suspicious_rate=prior_suspicious,
        seed=seed,
    )
    policies: tuple[EvidencePolicy, ...] = (
        GreedyPolicy(),
        RandomEvidencePolicy(),
        InformationGainPolicy(),
        ValueOfInformationPolicy(),
    )

    results: list[Result] = []
    for policy_index, policy in enumerate(policies):
        # Separate RNG streams keep policy randomness reproducible and independent.
        policy_rng = random.Random(seed + 10_000 * (policy_index + 1))
        outcomes = [
            _run_episode(
                episode,
                policy=policy,
                prior_suspicious=prior_suspicious,
                sources=sources,
                costs=costs,
                rng=policy_rng,
            )
            for episode in sampled_episodes
        ]
        results.append(
            _summarize(
                policy_name=policy.name,
                prior_suspicious=prior_suspicious,
                costs=costs,
                outcomes=outcomes,
            )
        )

    return results


def run_sweep(
    *,
    priors: list[float],
    false_approve_costs: list[float],
    episodes: int,
    seed: int,
) -> list[Result]:
    """Sweep prior uncertainty and asymmetric decision cost."""
    results: list[Result] = []

    for prior_index, prior in enumerate(priors):
        for cost_index, false_approve_cost in enumerate(false_approve_costs):
            condition_seed = seed + prior_index * 1_000 + cost_index
            results.extend(
                run_condition(
                    prior_suspicious=prior,
                    false_approve_cost=false_approve_cost,
                    episodes=episodes,
                    seed=condition_seed,
                )
            )

    return results


def write_csv(results: list[Result], output_path: Path) -> None:
    """Write benchmark results to CSV."""
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
                    "episodes": result.episodes,
                    "accuracy": f"{result.accuracy:.6f}",
                    "mean_decision_loss": f"{result.mean_decision_loss:.6f}",
                    "mean_evidence_cost": f"{result.mean_evidence_cost:.6f}",
                    "mean_total_loss": f"{result.mean_total_loss:.6f}",
                    "evidence_rate": f"{result.evidence_rate:.6f}",
                    "manual_review_rate": f"{result.manual_review_rate:.6f}",
                }
            )


def _parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compare one-step evidence-acquisition policies.",
    )
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--priors",
        type=_parse_float_list,
        default=_parse_float_list("0.02,0.05,0.1,0.2,0.4"),
    )
    parser.add_argument(
        "--false-approve-costs",
        type=_parse_float_list,
        default=_parse_float_list("1,2,5,10,20"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evidence.csv"),
    )
    return parser


def main() -> None:
    """Run the evidence benchmark and save results."""
    args = _build_parser().parse_args()
    results = run_sweep(
        priors=args.priors,
        false_approve_costs=args.false_approve_costs,
        episodes=args.episodes,
        seed=args.seed,
    )
    write_csv(results, args.output)
    print(f"wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
