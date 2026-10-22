"""Caller-filtered legal[] test on the held-out test split (kept checkpoint, no refitting).

Every LIBERO row carries the same seven-verb legal list, so the filtering path of
the contract was never exercised. Here each test event's action question is
rebuilt with a reduced legal list: the gold verb, abort, and two other verbs
drawn with random.Random(<event id>), kept in the pack's verb order. The prompt
(state.legal and the choice criteria) changes exactly as a caller's filter would
change it. The gold verb is always legal, because a caller filters out verbs that
cannot run, and the gold verb did run.

Compared with the recorded full-list predictions, both as-is and masked post hoc
to the same subset. This is not the inspect_v1 pack-transfer test: there is no
inspection data or label to score that honestly.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from actone.api import action_question  # noqa: E402
from dohnuts.train import evaluate  # noqa: E402
from dohnuts.training_data import DecisionCollator, load_records  # noqa: E402
from scripts.gpu_latency import KEPT, MODEL, load_model  # noqa: E402

LEGAL = ["reach", "grasp", "lift", "place", "retract", "home", "abort"]
DATA = ROOT / "data/processed/actone_v4"
OTHERS = 2


def reduced(row):
    gold = LEGAL[row["target"].index(1.0)]
    pool = [verb for verb in LEGAL if verb not in (gold, "abort")]
    keep = {gold, "abort", *random.Random(row["id"]).sample(pool, OTHERS)}
    legal = [verb for verb in LEGAL if verb in keep]
    out = json.loads(json.dumps(row))
    out["state"]["legal"] = legal
    out["question"] = action_question(legal)
    out["target"] = [1.0 if verb == gold else 0.0 for verb in legal]
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/legal_subset_gpu2")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_num_threads(8)
    rows = [json.loads(line) for line in (DATA / "test.jsonl").read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["dataset"] == "action"]
    assert all(row["state"]["legal"] == LEGAL for row in rows)
    built = [reduced(row) for row in rows]
    inputs = args.out / "test_action_reduced_legal.jsonl"
    inputs.write_text("".join(json.dumps(row) + "\n" for row in built))
    model = load_model(KEPT / "seed-42/best.pt")
    predictions = evaluate(model, load_records(inputs), DecisionCollator(MODEL), args.out / "predictions.jsonl")
    by_id = {row["id"]: row for row in built}
    recorded = {
        row["id"]: row["logits"]
        for row in map(json.loads, (KEPT / "seed-42/test-predictions.jsonl").read_text().splitlines())
        if row["dataset"] == "action"
    }
    correct = illegal = full_correct = masked_correct = changed = 0
    for prediction in predictions:
        row = by_id[prediction["id"]]
        legal = row["state"]["legal"]
        gold = legal[row["target"].index(1.0)]
        pick = legal[int(np.argmax(prediction["logits"]))]
        full = recorded[row["id"]]
        full_pick = LEGAL[int(np.argmax(full))]
        masked_pick = max(legal, key=lambda verb: full[LEGAL.index(verb)])
        correct += pick == gold
        illegal += pick not in legal
        full_correct += full_pick == gold
        masked_correct += masked_pick == gold
        changed += pick != masked_pick
    n = len(predictions)
    report = {
        "events": n,
        "legal_list": f"gold + abort + {OTHERS} others drawn with random.Random(event id); {OTHERS + 2} candidates",
        "checkpoint": str(KEPT / "seed-42/best.pt"),
        "action_accuracy_reduced_prompt": correct / n,
        "illegal_action_rate_reduced_prompt": illegal / n,
        "action_accuracy_full_list_recorded": full_correct / n,
        "action_accuracy_full_list_masked_to_subset": masked_correct / n,
        "picks_differing_reduced_prompt_vs_masked": changed,
        "note": "argmax only; no temperature or threshold is refit. Gold verb is always in the reduced list.",
    }
    (args.out / "legal_subset.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
