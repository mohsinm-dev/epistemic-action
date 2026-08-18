"""Plot reward curves from an experiment CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def plot_results(csv_path: Path, output_dir: Path) -> None:
    """Create one reward-vs-reliability figure per clue cost."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        rows.extend(csv.DictReader(file_handle))

    costs = sorted({float(row["clue_cost"]) for row in rows})
    output_dir.mkdir(parents=True, exist_ok=True)

    for cost in costs:
        figure, axis = plt.subplots()
        for agent in ("greedy", "epistemic"):
            agent_rows = [
                row
                for row in rows
                if row["agent"] == agent and float(row["clue_cost"]) == cost
            ]
            agent_rows.sort(key=lambda row: float(row["clue_reliability"]))
            axis.plot(
                [float(row["clue_reliability"]) for row in agent_rows],
                [float(row["mean_net_reward"]) for row in agent_rows],
                marker="o",
                label=agent,
            )

        axis.set_xlabel("Clue reliability")
        axis.set_ylabel("Mean net reward")
        axis.set_title(f"Clue cost = {cost:g}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"reward_cost_{cost:g}.png", dpi=160)
        plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    """Create the plotting CLI parser."""
    parser = argparse.ArgumentParser(description="Plot epistemic-action sweep results.")
    parser.add_argument("csv_path", type=Path, nargs="?", default=Path("results/sweep.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser


def main() -> None:
    """Plot a saved experiment sweep."""
    args = _build_parser().parse_args()
    plot_results(args.csv_path, args.output_dir)
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
