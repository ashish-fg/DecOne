"""Calibration-only temperatures and policy thresholds, then one test pass.

Temperatures are a separate positive scale per head (action, p_success,
p_unsafe, escalate), fit by negative log likelihood on calibration logits.
The policy grid uses those scaled probabilities and the frozen cost table.
Test rows are not read until both are frozen.
"""

import json
from pathlib import Path

import numpy as np

LEGAL = ["reach", "grasp", "lift", "place", "retract", "home", "abort"]
ESCALATE = ["act", "ask_big", "ask_human", "abort"]
HEADS = ["action", "p_success", "p_unsafe", "escalate"]


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def event_key(row):
    return row["id"].split(":", 1)[1]


def softmax(logits, temperature):
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max()
    weights = np.exp(scaled)
    return weights / weights.sum()


def nll(rows, temperature):
    total = 0.0
    for row in rows:
        probability = softmax(row["logits"], temperature)
        target = np.asarray(row["target"], dtype=np.float64)
        total -= float(np.sum(target * np.log(np.maximum(probability, 1e-12))))
    return total / max(len(rows), 1)


def fit_temperature(rows):
    if len(rows) < 10:
        return 1.0
    grid = np.geomspace(0.1, 10.0, 64)
    losses = [nll(rows, float(temperature)) for temperature in grid]
    best = int(np.argmin(losses))
    lo = grid[max(0, best - 1)]
    hi = grid[min(len(grid) - 1, best + 1)]
    fine = np.linspace(lo, hi, 25)
    fine_losses = [nll(rows, float(temperature)) for temperature in fine]
    return float(fine[int(np.argmin(fine_losses))])


def summarize(rows, temperature):
    if not rows:
        return {"n": 0}
    accuracy = []
    brier = []
    confidence = []
    labels = []
    predictions = []
    for row in rows:
        probability = softmax(row["logits"], temperature)
        target = np.asarray(row["target"], dtype=np.float64)
        label = int(target.argmax())
        pred = int(probability.argmax())
        labels.append(label)
        predictions.append(pred)
        accuracy.append(float(label == pred))
        brier.append(float(np.sum((probability - target) ** 2)))
        confidence.append(float(probability.max()))
    accuracy = np.asarray(accuracy)
    confidence = np.asarray(confidence)
    ece = 0.0
    bins = []
    edges = np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        count = int(mask.sum())
        bin_confidence = bin_accuracy = None
        if count:
            bin_confidence = float(confidence[mask].mean())
            bin_accuracy = float(accuracy[mask].mean())
            ece += count / len(rows) * abs(bin_confidence - bin_accuracy)
        bins.append(
            {
                "lower": float(lo),
                "upper": float(hi),
                "n": count,
                "confidence": bin_confidence,
                "accuracy": bin_accuracy,
            }
        )
    return {
        "n": len(rows),
        "accuracy": float(accuracy.mean()),
        "brier": float(np.mean(brier)),
        "ece_15": float(ece),
        "reliability": bins,
    }


def decision_of(action, p_success, p_unsafe, escalate, legal, p_min_act, p_min_human, u_max):
    if action not in legal:
        return "abort"
    if p_unsafe > u_max:
        return "abort"
    if escalate == "abort" or action == "abort":
        return "abort"
    if escalate == "ask_human" or p_success < p_min_human:
        return "ask_human"
    if escalate == "ask_big":
        return "ask_big"
    if escalate == "act" and p_success >= p_min_act:
        return "execute"
    return "hold"


def cost_of(decision, gold):
    """Frozen costs. Incorrect abort of a successful act costs the same as a failed act."""
    if decision == "execute":
        if gold["unsafe"]:
            return 25.0
        if not gold["success"]:
            return 5.0
        return 0.0
    if decision == "ask_big":
        return 0.0 if gold["escalate"] == "ask_big" else 1.0
    if decision == "ask_human":
        return 0.0 if gold["escalate"] == "ask_human" else 3.0
    if decision in ("abort", "hold"):
        correct_stop = gold["escalate"] == "abort" or gold["verb"] == "abort" or gold["unsafe"]
        if correct_stop:
            return 0.5
        if gold["success"] and gold["escalate"] == "act":
            return 5.0
        return 0.5
    raise ValueError(decision)


def oracle_cost(gold):
    return min(cost_of(name, gold) for name in ("execute", "ask_big", "ask_human", "abort", "hold"))


def join_events(prediction_rows, labeled_rows, temperatures):
    labels = {}
    for row in labeled_rows:
        labels.setdefault(event_key(row), {})[row["dataset"]] = row
    grouped = {}
    for row in prediction_rows:
        grouped.setdefault(event_key(row), {})[row["dataset"]] = row
    events = []
    for key, heads in grouped.items():
        if not all(name in heads for name in HEADS):
            continue
        gold_rows = labels.get(key, {})
        action_p = softmax(heads["action"]["logits"], temperatures["action"])
        success_p = softmax(heads["p_success"]["logits"], temperatures["p_success"])
        unsafe_p = softmax(heads["p_unsafe"]["logits"], temperatures["p_unsafe"])
        escalate_p = softmax(heads["escalate"]["logits"], temperatures["escalate"])
        action = LEGAL[int(action_p.argmax())]
        escalate = ESCALATE[int(escalate_p.argmax())]
        source = gold_rows.get("action", {})
        state = source.get("state", {})
        legal = state.get("legal", LEGAL)
        gold_action = LEGAL[int(np.argmax(heads["action"]["target"]))]
        events.append(
            {
                "key": key,
                "action": action,
                "action_confidence": float(action_p.max()),
                "p_success": float(success_p[1]),
                "p_unsafe": float(unsafe_p[1]),
                "escalate": escalate,
                "legal": legal,
                "illegal": action not in legal,
                "gold": {
                    "verb": gold_action,
                    "success": bool(np.argmax(heads["p_success"]["target"]) == 1),
                    "unsafe": bool(np.argmax(heads["p_unsafe"]["target"]) == 1),
                    "escalate": ESCALATE[int(np.argmax(heads["escalate"]["target"]))],
                },
            }
        )
    return events


def mean_cost(events, p_min_act, p_min_human, u_max):
    if not events:
        return None
    total = 0.0
    for event in events:
        decision = decision_of(
            event["action"],
            event["p_success"],
            event["p_unsafe"],
            event["escalate"],
            event["legal"],
            p_min_act,
            p_min_human,
            u_max,
        )
        total += cost_of(decision, event["gold"])
    return total / len(events)


def fit_policy(events):
    best = None
    for p_min_act in (0.50, 0.60, 0.70, 0.80, 0.90):
        for p_min_human in (0.20, 0.30, 0.40, 0.50, 0.60):
            if p_min_human >= p_min_act:
                continue
            for u_max in (0.10, 0.20, 0.30, 0.40, 0.50):
                score = mean_cost(events, p_min_act, p_min_human, u_max)
                if best is None or score < best[0]:
                    best = (score, p_min_act, p_min_human, u_max)
    return {
        "P_MIN_ACT": best[1],
        "P_MIN_HUMAN": best[2],
        "U_MAX": best[3],
        "calibration_mean_cost": best[0],
    }


def policy_report(events, thresholds):
    decisions = []
    costs = []
    regrets = []
    illegal = 0
    for event in events:
        if event["illegal"]:
            illegal += 1
        decision = decision_of(
            event["action"],
            event["p_success"],
            event["p_unsafe"],
            event["escalate"],
            event["legal"],
            thresholds["P_MIN_ACT"],
            thresholds["P_MIN_HUMAN"],
            thresholds["U_MAX"],
        )
        paid = cost_of(decision, event["gold"])
        decisions.append(decision)
        costs.append(paid)
        regrets.append(paid - oracle_cost(event["gold"]))
    counts = {name: decisions.count(name) for name in ("execute", "ask_big", "ask_human", "abort", "hold")}
    executed = [event for event, decision in zip(events, decisions) if decision == "execute"]
    failures = [
        event
        for event, decision in zip(events, decisions)
        if decision == "execute" and (event["gold"]["unsafe"] or not event["gold"]["success"])
    ]
    return {
        "n": len(events),
        "illegal_action_rate": illegal / len(events) if events else None,
        "mean_cost": float(np.mean(costs)) if costs else None,
        "mean_regret": float(np.mean(regrets)) if regrets else None,
        "decisions": counts,
        "execute_coverage": len(executed) / len(events) if events else None,
        "execute_risk": len(failures) / len(executed) if executed else None,
    }


def risk_coverage(events, thresholds):
    curve = []
    for p_min_act in np.round(np.linspace(0.50, 0.99, 12), 2):
        report = policy_report(
            events,
            {
                "P_MIN_ACT": float(p_min_act),
                "P_MIN_HUMAN": thresholds["P_MIN_HUMAN"],
                "U_MAX": thresholds["U_MAX"],
            },
        )
        curve.append(
            {
                "P_MIN_ACT": float(p_min_act),
                "coverage": report["execute_coverage"],
                "risk": report["execute_risk"],
                "mean_cost": report["mean_cost"],
            }
        )
    return curve


def head_metrics(prediction_rows, temperatures):
    grouped = {}
    for row in prediction_rows:
        grouped.setdefault(row["dataset"], []).append(row)
    raw = {name: summarize(grouped.get(name, []), 1.0) for name in HEADS}
    scaled = {name: summarize(grouped.get(name, []), temperatures[name]) for name in HEADS}
    for metrics in list(raw.values()) + list(scaled.values()):
        metrics.pop("reliability", None)
    return {"raw": raw, "scaled": scaled}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--calibration-labels", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
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
    heads = report["test_heads"]["scaled"]
    print(
        json.dumps(
            {
                "temperatures": report["temperatures"],
                "thresholds": report["thresholds"],
                "action_accuracy": heads["action"]["accuracy"],
                "escalate_accuracy": heads["escalate"]["accuracy"],
                "illegal_action_rate": report["test_policy"]["illegal_action_rate"],
                "p_success_ece": heads["p_success"]["ece_15"],
                "p_success_brier": heads["p_success"]["brier"],
                "p_unsafe_ece": heads["p_unsafe"]["ece_15"],
                "p_unsafe_brier": heads["p_unsafe"]["brier"],
                "regret": report["test_policy"]["mean_regret"],
            }
        )
    )


def evaluate_run(calibration_predictions, test_predictions, calibration_labels, test_labels):
    calibration_groups = {}
    for row in calibration_predictions:
        calibration_groups.setdefault(row["dataset"], []).append(row)
    temperatures = {name: fit_temperature(calibration_groups.get(name, [])) for name in HEADS}
    calibration_events = join_events(calibration_predictions, calibration_labels, temperatures)
    thresholds = fit_policy(calibration_events)
    test_events = join_events(test_predictions, test_labels, temperatures)
    return {
        "temperatures": temperatures,
        "thresholds": thresholds,
        "calibration_policy": policy_report(calibration_events, thresholds),
        "test_heads": head_metrics(test_predictions, temperatures),
        "test_policy": policy_report(test_events, thresholds),
        "risk_coverage": risk_coverage(test_events, thresholds),
        "reliability": {
            name: {
                "raw": summarize(
                    [row for row in test_predictions if row["dataset"] == name],
                    1.0,
                )["reliability"],
                "scaled": summarize(
                    [row for row in test_predictions if row["dataset"] == name],
                    temperatures[name],
                )["reliability"],
            }
            for name in ("p_success", "p_unsafe")
        },
    }


if __name__ == "__main__":
    main()
