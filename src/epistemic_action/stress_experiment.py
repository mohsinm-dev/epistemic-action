"""Stress-test correlated and miscalibrated evidence models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
from pathlib import Path
import math
import random
from statistics import fmean

from epistemic_action.evidence import (
    DecisionCosts,
    Signal,
    State,
    best_decision,
    decision_loss,
    posterior_suspicious,
)
from epistemic_action.stress import CorrelatedPair, posterior_from_pair, symmetric_source


@dataclass(frozen=True, slots=True)
class Result:
    """Aggregate metrics for one inference model under one stress condition."""

    model: str
    prior_suspicious: float
    true_accuracy: float
    assumed_accuracy: float
    correlation: float
    episodes: int
    accuracy: float
    mean_decision_loss: float
    brier_score: float
    log_loss: float
    mean_posterior_shift: float


def _clip_probability(probability: float) -> float:
    """Clip probability only for stable logarithms."""
    return min(max(probability, 1e-12), 1.0 - 1e-12)


def _sample_state(prior_suspicious: float, rng: random.Random) -> State:
    """Sample one latent state from the configured prior."""
    return State.SUSPICIOUS if rng.random() < prior_suspicious else State.LEGITIMATE


def _posterior(
    *,
    model: str,
    prior_suspicious: float,
    first: Signal,
    second: Signal,
    true_accuracy: float,
    assumed_accuracy: float,
    correlation: float,
) -> float:
    """Infer posterior probability under the selected model."""
    if model == "single_source":
        assumed_source = symmetric_source("assumed", assumed_accuracy)
        return posterior_suspicious(prior_suspicious, assumed_source, first)

    if model == "naive_independent":
        assumed_source = symmetric_source("assumed", assumed_accuracy)
        return posterior_from_pair(
            prior_suspicious,
            assumed_source,
            first,
            second,
            correlation=0.0,
        )

    if model == "correlation_aware":
        assumed_source = symmetric_source("assumed", assumed_accuracy)
        return posterior_from_pair(
            prior_suspicious,
            assumed_source,
            first,
            second,
            correlation=correlation,
        )

    if model == "oracle":
        true_source = symmetric_source("true", true_accuracy)
        return posterior_from_pair(
            prior_suspicious,
            true_source,
            first,
            second,
            correlation=correlation,
        )

    raise ValueError(f"unknown model: {model}")


def run_condition(
    *,
    prior_suspicious: float,
    true_accuracy: float,
    assumed_accuracy: float,
    correlation: float,
    false_approve_cost: float,
    episodes: int,
    seed: int,
) -> list[Result]:
    """Evaluate inference models on identical correlated evidence pairs."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    costs = DecisionCosts(false_approve=false_approve_cost, false_reject=1.0)
    true_source = symmetric_source("true", true_accuracy)
    pair = CorrelatedPair(true_source, correlation)
    rng = random.Random(seed)

    samples: list[tuple[State, Signal, Signal]] = []
    for _ in range(episodes):
        state = _sample_state(prior_suspicious, rng)
        first, second = pair.sample(state, rng)
        samples.append((state, first, second))

    models = (
        "single_source",
        "naive_independent",
        "correlation_aware",
        "oracle",
    )
    results: list[Result] = []

    for model in models:
        correct_values: list[float] = []
        decision_losses: list[float] = []
        brier_values: list[float] = []
        log_losses: list[float] = []
        shifts: list[float] = []

        for state, first, second in samples:
            posterior = _posterior(
                model=model,
                prior_suspicious=prior_suspicious,
                first=first,
                second=second,
                true_accuracy=true_accuracy,
                assumed_accuracy=assumed_accuracy,
                correlation=correlation,
            )
            decision = best_decision(posterior, costs)
            target = 1.0 if state is State.SUSPICIOUS else 0.0
            probability = _clip_probability(posterior)

            correct_values.append(
                float(
                    (decision.value == "reject" and state is State.SUSPICIOUS)
                    or (decision.value == "approve" and state is State.LEGITIMATE)
                )
            )
            decision_losses.append(decision_loss(state, decision, costs))
            brier_values.append((posterior - target) ** 2)
            log_losses.append(
                -(target * math.log(probability) + (1.0 - target) * math.log(1.0 - probability))
            )
            shifts.append(abs(posterior - prior_suspicious))

        results.append(
            Result(
                model=model,
                prior_suspicious=prior_suspicious,
                true_accuracy=true_accuracy,
                assumed_accuracy=assumed_accuracy,
                correlation=correlation,
                episodes=episodes,
                accuracy=fmean(correct_values),
                mean_decision_loss=fmean(decision_losses),
                brier_score=fmean(brier_values),
                log_loss=fmean(log_losses),
                mean_posterior_shift=fmean(shifts),
            )
        )

    return results


def run_sweep(
    *,
    priors: list[float],
    correlations: list[float],
    assumed_accuracies: list[float],
    true_accuracy: float,
    false_approve_cost: float,
    episodes: int,
    seed: int,
) -> list[Result]:
    """Sweep dependence strength and model calibration."""
    results: list[Result] = []

    for prior_index, prior in enumerate(priors):
        for correlation_index, correlation in enumerate(correlations):
            for accuracy_index, assumed_accuracy in enumerate(assumed_accuracies):
                condition_seed = (
                    seed
                    + prior_index * 10_000
                    + correlation_index * 100
                    + accuracy_index
                )
                results.extend(
                    run_condition(
                        prior_suspicious=prior,
                        true_accuracy=true_accuracy,
                        assumed_accuracy=assumed_accuracy,
                        correlation=correlation,
                        false_approve_cost=false_approve_cost,
                        episodes=episodes,
                        seed=condition_seed,
                    )
                )

    return results


def write_csv(results: list[Result], output_path: Path) -> None:
    """Write stress-test results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(Result)]

    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result.model,
                    "prior_suspicious": result.prior_suspicious,
                    "true_accuracy": result.true_accuracy,
                    "assumed_accuracy": result.assumed_accuracy,
                    "correlation": result.correlation,
                    "episodes": result.episodes,
                    "accuracy": f"{result.accuracy:.6f}",
                    "mean_decision_loss": f"{result.mean_decision_loss:.6f}",
                    "brier_score": f"{result.brier_score:.6f}",
                    "log_loss": f"{result.log_loss:.6f}",
                    "mean_posterior_shift": f"{result.mean_posterior_shift:.6f}",
                }
            )


def _parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated float list."""
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Stress-test evidence inference under dependence and miscalibration.",
    )
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--true-accuracy", type=float, default=0.85)
    parser.add_argument("--false-approve-cost", type=float, default=5.0)
    parser.add_argument(
        "--priors",
        type=_parse_float_list,
        default=_parse_float_list("0.02,0.05,0.1"),
    )
    parser.add_argument(
        "--correlations",
        type=_parse_float_list,
        default=_parse_float_list("0.0,0.2,0.5,0.8"),
    )
    parser.add_argument(
        "--assumed-accuracies",
        type=_parse_float_list,
        default=_parse_float_list("0.85,0.95,0.75"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stress.csv"),
    )
    return parser


def main() -> None:
    """Run the stress sweep and save results."""
    args = _build_parser().parse_args()
    results = run_sweep(
        priors=args.priors,
        correlations=args.correlations,
        assumed_accuracies=args.assumed_accuracies,
        true_accuracy=args.true_accuracy,
        false_approve_cost=args.false_approve_cost,
        episodes=args.episodes,
        seed=args.seed,
    )
    write_csv(results, args.output)
    print(f"wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
