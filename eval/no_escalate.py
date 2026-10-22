"""Calibration-only evaluation for a model with no escalate head.

Same protocol as eval/offline.py: per-head temperatures (action, p_success,
p_unsafe) by NLL on calibration, then policy thresholds on calibration mean
cost, both frozen before test is read; test is scored once. With no escalate
head the policy sees escalate = "act" for every event, so only the
p_success / p_unsafe thresholds can route an event away from execute. Gold
escalate (for the frozen cost table) is read from the full v4 label files.

Also usable on a four-head run's predictions with --ignore-escalate, which
drops its escalate rows: the same decision rule with the head switched off at
inference time instead of never trained.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.offline import (  # noqa: E402
    ESCALATE,
    LEGAL,
    event_key,
    fit_policy,
    fit_temperature,
    load_jsonl,
    policy_report,
    risk_coverage,
    softmax,
    summarize,
)

HEADS = ["action", "p_success", "p_unsafe"]


def join_events(prediction_rows, labeled_rows, temperatures):
    labels = {}
    for row in labeled_rows:
        labels.setdefault(event_key(row), {})[row["dataset"]] = row
    grouped = {}
    for row in prediction_rows:
        if row["dataset"] in HEADS:
            grouped.setdefault(event_key(row), {})[row["dataset"]] = row
    events = []
    for key, heads in grouped.items():
        if not all(name in heads for name in HEADS):
            continue
        gold_rows = labels[key]
        action_p = softmax(heads["action"]["logits"], temperatures["action"])
        success_p = softmax(heads["p_success"]["logits"], temperatures["p_success"])
        unsafe_p = softmax(heads["p_unsafe"]["logits"], temperatures["p_unsafe"])
        action = LEGAL[int(action_p.argmax())]
        legal = gold_rows["action"].get("state", {}).get("legal", LEGAL)
        events.append(
            {
                "key": key,
                "action": action,
                "action_confidence": float(action_p.max()),
                "p_success": float(success_p[1]),
                "p_unsafe": float(unsafe_p[1]),
                "escalate": "act",
                "legal": legal,
                "illegal": action not in legal,
                "gold": {
                    "verb": LEGAL[int(np.argmax(heads["action"]["target"]))],
                    "success": bool(np.argmax(heads["p_success"]["target"]) == 1),
                    "unsafe": bool(np.argmax(heads["p_unsafe"]["target"]) == 1),
                    "escalate": ESCALATE[int(np.argmax(gold_rows["escalate"]["target"]))],
                },
            }
        )
    return events


def evaluate_run(calibration_predictions, test_predictions, calibration_labels, test_labels):
    calibration_predictions = [row for row in calibration_predictions if row["dataset"] in HEADS]
    test_predictions = [row for row in test_predictions if row["dataset"] in HEADS]
    temperatures = {
        name: fit_temperature([row for row in calibration_predictions if row["dataset"] == name]) for name in HEADS
    }
    calibration_events = join_events(calibration_predictions, calibration_labels, temperatures)
    thresholds = fit_policy(calibration_events)
    test_events = join_events(test_predictions, test_labels, temperatures)
    heads = {}
    for kind, scale in (("raw", None), ("scaled", temperatures)):
        heads[kind] = {}
        for name in HEADS:
            metrics = summarize(
                [row for row in test_predictions if row["dataset"] == name], scale[name] if scale else 1.0
            )
            metrics.pop("reliability", None)
            heads[kind][name] = metrics
    return {
        "escalate_head": "none; policy sees escalate = act",
        "temperatures": temperatures,
        "thresholds": thresholds,
        "calibration_policy": policy_report(calibration_events, thresholds),
        "test_heads": heads,
        "test_policy": policy_report(test_events, thresholds),
        "risk_coverage": risk_coverage(test_events, thresholds),
        "reliability": {
            name: {
                "raw": summarize([row for row in test_predictions if row["dataset"] == name], 1.0)["reliability"],
                "scaled": summarize(
                    [row for row in test_predictions if row["dataset"] == name], temperatures[name]
                )["reliability"],
            }
            for name in ("action", "p_success")
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--calibration-labels", type=Path, required=True, help="full v4 file, with escalate rows")
    parser.add_argument("--test-labels", type=Path, required=True, help="full v4 file, with escalate rows")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_run(
        load_jsonl(args.calibration_predictions),
        load_jsonl(args.test_predictions),
        load_jsonl(args.calibration_labels),
        load_jsonl(args.test_labels),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    scaled = report["test_heads"]["scaled"]
    print(
        json.dumps(
            {
                "temperatures": report["temperatures"],
                "thresholds": report["thresholds"],
                "action_accuracy": scaled["action"]["accuracy"],
                "p_success_accuracy": scaled["p_success"]["accuracy"],
                "p_success_ece": scaled["p_success"]["ece_15"],
                "policy": report["test_policy"],
            }
        )
    )


if __name__ == "__main__":
    main()
