"""Reliability diagrams and the frozen-cost risk-coverage curve."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def reliability(bins, title, path):
    kept = [item for item in bins if item["n"] and item["confidence"] is not None]
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    axis.plot([0, 1], [0, 1], color="#9aa0a6", linewidth=1)
    if kept:
        axis.plot(
            [item["confidence"] for item in kept],
            [item["accuracy"] for item in kept],
            marker="o",
            color="#1a73e8",
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def risk_coverage(curve, path):
    usable = [item for item in curve if item["coverage"] is not None and item["risk"] is not None]
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    if usable:
        axis.plot(
            [item["coverage"] for item in usable],
            [item["risk"] for item in usable],
            marker="o",
            color="#c5221f",
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Execute coverage")
    axis.set_ylabel("Risk among executed acts")
    axis.set_title("Risk-coverage on held-out episodes")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_plots(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for head in ("p_success", "p_unsafe"):
        for scale in ("raw", "scaled"):
            path = directory / f"reliability_{head}_{scale}.png"
            reliability(report["reliability"][head][scale], f"{head} {scale}", path)
            written.append(str(path))
    path = directory / "risk_coverage.png"
    risk_coverage(report["risk_coverage"], path)
    written.append(str(path))
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_plots(json.loads(args.report.read_text()), args.out)))
