"""Map planner disagreement regions with exact expected trajectory loss."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import fmean

from epistemic_action.active_inference import (
    select_sophisticated_efe_source,
    select_standard_efe_source,
)
from epistemic_action.evidence import (
    DecisionCosts,
    EvidenceSource,
    Signal,
    posterior_suspicious,
    probability_signal,
)
from epistemic_action.sequential import (
    best_terminal_utility,
    select_lookahead_source,
    select_myopic_source,
)


@dataclass(frozen=True, slots=True)
class ExpectedTrajectory:
    """Exact expected cost and query count under a deterministic policy."""

    total_loss: float
    queries: float
    evidence_cost: float


@dataclass(frozen=True, slots=True)
class Result:
    """One planner evaluated under one synthetic condition."""

    policy: str
    prior_suspicious: float
    screen_accuracy: float
    screen_cost: float
    review_accuracy: float
    review_cost: float
    preference_precision: float
    first_action: str
    expected_total_loss: float
    expected_queries: float
    expected_evidence_cost: float
    regret_vs_bayes: float


def _sources(
    screen_accuracy: float,
    screen_cost: float,
    review_accuracy: float,
    review_cost: float,
) -> tuple[EvidenceSource, ...]:
    """Build the two-source synthetic evidence environment."""
    return (
        EvidenceSource(
            "screen",
            sensitivity=screen_accuracy,
            specificity=screen_accuracy,
            cost=screen_cost,
        ),
        EvidenceSource(
            "review",
            sensitivity=review_accuracy,
            specificity=review_accuracy,
            cost=review_cost,
        ),
    )


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
    """Select one query under the requested planner."""
    if not sources or steps_remaining == 0:
        return None
    if policy == "myopic_voi":
        return select_myopic_source(posterior, sources, costs, escalation_cost)
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


def exact_expected_trajectory(
    *,
    policy: str,
    posterior: float,
    sources: tuple[EvidenceSource, ...],
    costs: DecisionCosts,
    escalation_cost: float,
    preference_precision: float,
    steps_remaining: int,
) -> ExpectedTrajectory:
    """Evaluate a policy exactly under the correctly specified evidence model."""
    source = _select_source(
        policy=policy,
        posterior=posterior,
        sources=sources,
        costs=costs,
        escalation_cost=escalation_cost,
        preference_precision=preference_precision,
        steps_remaining=steps_remaining,
    )

    if source is None:
        return ExpectedTrajectory(
            total_loss=-best_terminal_utility(posterior, costs, escalation_cost),
            queries=0.0,
            evidence_cost=0.0,
        )

    remaining = tuple(candidate for candidate in sources if candidate != source)
    expected_loss = source.cost
    expected_queries = 1.0
    expected_evidence_cost = source.cost

    for signal in Signal:
        signal_probability = probability_signal(posterior, source, signal)
        if signal_probability <= 0.0:
            continue
        next_posterior = posterior_suspicious(posterior, source, signal)
        continuation = exact_expected_trajectory(
            policy=policy,
            posterior=next_posterior,
            sources=remaining,
            costs=costs,
            escalation_cost=escalation_cost,
            preference_precision=preference_precision,
            steps_remaining=steps_remaining - 1,
        )
        expected_loss += signal_probability * continuation.total_loss
        expected_queries += signal_probability * continuation.queries
        expected_evidence_cost += signal_probability * continuation.evidence_cost

    return ExpectedTrajectory(
        total_loss=expected_loss,
        queries=expected_queries,
        evidence_cost=expected_evidence_cost,
    )


def run_condition(
    *,
    prior_suspicious: float,
    screen_accuracy: float,
    screen_cost: float,
    review_accuracy: float,
    review_cost: float,
    false_approve_cost: float,
    escalation_cost: float,
    preference_precision: float,
    horizon: int,
) -> list[Result]:
    """Evaluate all planners under one fully specified synthetic condition."""
    sources = _sources(
        screen_accuracy,
        screen_cost,
        review_accuracy,
        review_cost,
    )
    costs = DecisionCosts(false_approve=false_approve_cost, false_reject=1.0)
    policies = (
        "myopic_voi",
        "bayes_lookahead",
        "standard_efe",
        "sophisticated_efe",
    )

    trajectories: dict[str, ExpectedTrajectory] = {}
    first_actions: dict[str, str] = {}
    for policy in policies:
        source = _select_source(
            policy=policy,
            posterior=prior_suspicious,
            sources=sources,
            costs=costs,
            escalation_cost=escalation_cost,
            preference_precision=preference_precision,
            steps_remaining=horizon,
        )
        first_actions[policy] = source.name if source is not None else "stop"
        trajectories[policy] = exact_expected_trajectory(
            policy=policy,
            posterior=prior_suspicious,
            sources=sources,
            costs=costs,
            escalation_cost=escalation_cost,
            preference_precision=preference_precision,
            steps_remaining=horizon,
        )

    bayes_loss = trajectories["bayes_lookahead"].total_loss
    return [
        Result(
            policy=policy,
            prior_suspicious=prior_suspicious,
            screen_accuracy=screen_accuracy,
            screen_cost=screen_cost,
            review_accuracy=review_accuracy,
            review_cost=review_cost,
            preference_precision=preference_precision,
            first_action=first_actions[policy],
            expected_total_loss=trajectories[policy].total_loss,
            expected_queries=trajectories[policy].queries,
            expected_evidence_cost=trajectories[policy].evidence_cost,
            regret_vs_bayes=trajectories[policy].total_loss - bayes_loss,
        )
        for policy in policies
    ]


def run_sweep(
    *,
    priors: list[float],
    screen_accuracies: list[float],
    screen_costs: list[float],
    review_accuracies: list[float],
    review_costs: list[float],
    precisions: list[float],
    false_approve_cost: float,
    escalation_cost: float,
    horizon: int,
) -> list[Result]:
    """Run the deterministic disagreement campaign."""
    results: list[Result] = []
    for prior in priors:
        for screen_accuracy in screen_accuracies:
            for screen_cost in screen_costs:
                for review_accuracy in review_accuracies:
                    for review_cost in review_costs:
                        for precision in precisions:
                            results.extend(
                                run_condition(
                                    prior_suspicious=prior,
                                    screen_accuracy=screen_accuracy,
                                    screen_cost=screen_cost,
                                    review_accuracy=review_accuracy,
                                    review_cost=review_cost,
                                    false_approve_cost=false_approve_cost,
                                    escalation_cost=escalation_cost,
                                    preference_precision=precision,
                                    horizon=horizon,
                                )
                            )
    return results


def summarize(results: list[Result]) -> str:
    """Return a compact summary of first-action disagreement and regret."""
    grouped: dict[tuple[float, float, float, float, float, float], dict[str, Result]] = {}
    for result in results:
        key = (
            result.prior_suspicious,
            result.screen_accuracy,
            result.screen_cost,
            result.review_accuracy,
            result.review_cost,
            result.preference_precision,
        )
        grouped.setdefault(key, {})[result.policy] = result

    lines = [f"conditions: {len(grouped)}"]
    for policy in ("myopic_voi", "standard_efe", "sophisticated_efe"):
        rows = [condition[policy] for condition in grouped.values()]
        agreements = [
            condition[policy].first_action == condition["bayes_lookahead"].first_action
            for condition in grouped.values()
        ]
        lines.append(
            f"{policy}: first-action agreement={fmean(agreements):.3f}, "
            f"mean regret={fmean(row.regret_vs_bayes for row in rows):.6f}, "
            f"max regret={max(row.regret_vs_bayes for row in rows):.6f}"
        )
    return "\n".join(lines)


def write_csv(results: list[Result], output_path: Path) -> None:
    """Write campaign results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(Result)]
    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def _parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Create the campaign CLI."""
    parser = argparse.ArgumentParser(
        description="Map first-action disagreement and exact regret across planners.",
    )
    parser.add_argument("--priors", type=_parse_float_list, default=_parse_float_list("0.01,0.02,0.05,0.1,0.2"))
    parser.add_argument("--screen-accuracies", type=_parse_float_list, default=_parse_float_list("0.6,0.7,0.8"))
    parser.add_argument("--screen-costs", type=_parse_float_list, default=_parse_float_list("0.02,0.05,0.1"))
    parser.add_argument("--review-accuracies", type=_parse_float_list, default=_parse_float_list("0.85,0.95"))
    parser.add_argument("--review-costs", type=_parse_float_list, default=_parse_float_list("0.1,0.2,0.4"))
    parser.add_argument("--precisions", type=_parse_float_list, default=_parse_float_list("0.5,1,2,5,10"))
    parser.add_argument("--false-approve-cost", type=float, default=5.0)
    parser.add_argument("--escalation-cost", type=float, default=0.40)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/disagreement.csv"))
    return parser


def main() -> None:
    """Run the exact planner-disagreement campaign."""
    args = _build_parser().parse_args()
    results = run_sweep(
        priors=args.priors,
        screen_accuracies=args.screen_accuracies,
        screen_costs=args.screen_costs,
        review_accuracies=args.review_accuracies,
        review_costs=args.review_costs,
        precisions=args.precisions,
        false_approve_cost=args.false_approve_cost,
        escalation_cost=args.escalation_cost,
        horizon=args.horizon,
    )
    write_csv(results, args.output)
    print(summarize(results))
    print(f"wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
