"""Run reproducible sweeps for the information-seeking experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
from statistics import fmean

from epistemic_action.agents import EpistemicAgent, GreedyAgent
from epistemic_action.environment import Episode, Outcome, sample_episode


@dataclass(frozen=True, slots=True)
class Result:
    """Aggregate metrics for one agent under one experimental condition."""

    agent: str
    clue_reliability: float
    clue_cost: float
    episodes: int
    accuracy: float
    mean_net_reward: float
    clue_rate: float


def _sample_episodes(
    *,
    count: int,
    clue_reliability: float,
    seed: int,
) -> list[Episode]:
    """Generate a shared episode set so agents face identical latent worlds."""
    if count <= 0:
        raise ValueError("count must be positive")

    rng = random.Random(seed)
    return [sample_episode(rng, clue_reliability) for _ in range(count)]


def _summarize(
    *,
    agent_name: str,
    clue_reliability: float,
    clue_cost: float,
    outcomes: list[Outcome],
) -> Result:
    """Aggregate episode outcomes into comparable metrics."""
    return Result(
        agent=agent_name,
        clue_reliability=clue_reliability,
        clue_cost=clue_cost,
        episodes=len(outcomes),
        accuracy=fmean(float(outcome.correct) for outcome in outcomes),
        mean_net_reward=fmean(outcome.net_reward for outcome in outcomes),
        clue_rate=fmean(float(outcome.used_clue) for outcome in outcomes),
    )


def run_condition(
    *,
    clue_reliability: float,
    clue_cost: float,
    episodes: int,
    seed: int,
    information_weight: float,
) -> list[Result]:
    """Evaluate greedy and epistemic agents on the same sampled episodes."""
    sampled_episodes = _sample_episodes(
        count=episodes,
        clue_reliability=clue_reliability,
        seed=seed,
    )
    agents = {
        "greedy": GreedyAgent(),
        "epistemic": EpistemicAgent(information_weight=information_weight),
    }

    results: list[Result] = []
    for agent_index, (agent_name, agent) in enumerate(agents.items()):
        # Decision randomness is independent from environment sampling and stable per agent.
        decision_rng = random.Random(seed + 10_000 + agent_index)
        outcomes = [
            agent.run_episode(
                episode,
                clue_reliability=clue_reliability,
                clue_cost=clue_cost,
                rng=decision_rng,
            )
            for episode in sampled_episodes
        ]
        results.append(
            _summarize(
                agent_name=agent_name,
                clue_reliability=clue_reliability,
                clue_cost=clue_cost,
                outcomes=outcomes,
            )
        )

    return results


def run_sweep(
    *,
    reliabilities: list[float],
    costs: list[float],
    episodes: int,
    seed: int,
    information_weight: float,
) -> list[Result]:
    """Run all reliability/cost combinations."""
    results: list[Result] = []

    for reliability_index, reliability in enumerate(reliabilities):
        for cost_index, cost in enumerate(costs):
            condition_seed = seed + reliability_index * 1_000 + cost_index
            results.extend(
                run_condition(
                    clue_reliability=reliability,
                    clue_cost=cost,
                    episodes=episodes,
                    seed=condition_seed,
                    information_weight=information_weight,
                )
            )

    return results


def write_csv(results: list[Result], output_path: Path) -> None:
    """Write experiment results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "agent",
                "clue_reliability",
                "clue_cost",
                "episodes",
                "accuracy",
                "mean_net_reward",
                "clue_rate",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "agent": result.agent,
                    "clue_reliability": result.clue_reliability,
                    "clue_cost": result.clue_cost,
                    "episodes": result.episodes,
                    "accuracy": f"{result.accuracy:.6f}",
                    "mean_net_reward": f"{result.mean_net_reward:.6f}",
                    "clue_rate": f"{result.clue_rate:.6f}",
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
        description="Sweep clue reliability and cost in a minimal information-seeking task.",
    )
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--information-weight", type=float, default=1.0)
    parser.add_argument(
        "--reliabilities",
        type=_parse_float_list,
        default=_parse_float_list("0.5,0.6,0.7,0.8,0.9,1.0"),
    )
    parser.add_argument(
        "--costs",
        type=_parse_float_list,
        default=_parse_float_list("0.0,0.05,0.1,0.2,0.3,0.5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sweep.csv"),
    )
    return parser


def main() -> None:
    """Run the configured sweep and save results."""
    args = _build_parser().parse_args()
    results = run_sweep(
        reliabilities=args.reliabilities,
        costs=args.costs,
        episodes=args.episodes,
        seed=args.seed,
        information_weight=args.information_weight,
    )
    write_csv(results, args.output)
    print(f"wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
