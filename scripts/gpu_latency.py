"""GPU latency on the held-out test split, plus the full two-pass decision pipeline.

No training, no refitting. Each system uses the temperatures and thresholds that
were fitted on the calibration split and frozen before test was read.

Part 1, batch scoring: the vendor evaluate() path used for every offline number
(eval batch 16, shared-prefix candidate scoring, merged BF16 LoRA). One untimed
warm-up over a dev subset, then REPEATS timed passes over all 5,456 test rows
with CUDA synchronised before and after. The first pass's logits are compared
with the recorded predictions of the original GPU, so the timed model is the one
that produced the reported numbers.

Part 2, per decision: actone.api.ActOne.decide(), batch size 1, pass 1 (action)
then pass 2 (p_success, p_unsafe, escalate conditioned on the pass-1 verb), for
every test event, after WARMUP untimed dev decisions. Image decode is timed
separately and excluded from the decision latency. Also scores this production
pipeline on test with the frozen policy.
"""

import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from actone.api import ActOne  # noqa: E402
from actone.policy import from_thresholds  # noqa: E402
from dohnuts.model import DecisionModel  # noqa: E402
from dohnuts.train import evaluate, load_checkpoint  # noqa: E402
from dohnuts.training_data import DecisionCollator, load_records  # noqa: E402
from eval.offline import cost_of, oracle_cost  # noqa: E402

MODEL = ROOT / ".cache/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
INIT = ROOT / ".cache/hf/hub/models--PsiACE--Dohnuts-0.1.0-0.8B/snapshots/959828a3d9cf3dacbd1491e11bb1cd36d12709ac"
KEPT = ROOT / "runs/actone_v4_s3000"
DATA = ROOT / "data/processed/actone_v4"
LEGAL = ["reach", "grasp", "lift", "place", "retract", "home", "abort"]
ESCALATE = ["act", "ask_big", "ask_human", "abort"]
HEADS = ["action", "p_success", "p_unsafe", "escalate"]
REPEATS = 3
WARMUP = 32
SMOKE = None
CPU_THREADS = 8

SYSTEMS = {
    "actone_image": {
        "checkpoint": KEPT / "seed-42/best.pt",
        "images": True,
        "policy": KEPT / "seed-42/heldout_policy.json",
        "recorded": KEPT / "seed-42/test-predictions.jsonl",
    },
    "actone_no_image": {
        "checkpoint": KEPT / "seed-42/best.pt",
        "images": False,
        "policy": KEPT / "ablation/no_image_policy.json",
        "recorded": KEPT / "ablation/no_image_test.jsonl",
    },
    "zero_shot_image": {
        "checkpoint": None,
        "images": True,
        "policy": KEPT / "ablation/zero_shot_policy.json",
        "recorded": KEPT / "ablation/zero_shot_test.jsonl",
    },
}


def load_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_model(checkpoint):
    model = DecisionModel(MODEL)
    model.enable_lora(checkpointing=False)
    if checkpoint is None:
        model.load_adapter(INIT)
    else:
        load_checkpoint(model, checkpoint)
    model.merge()
    model.device = "cuda"
    return model


def strip_images(groups):
    for rows in groups.values():
        for row in rows:
            row.pop("image", None)
    return groups


def stats_ms(values):
    values = sorted(values)
    ms = [1000 * v for v in values]
    return {
        "n": len(ms),
        "mean_ms": statistics.fmean(ms),
        "std_ms": statistics.pstdev(ms),
        "median_ms": statistics.median(ms),
        "p90_ms": ms[int(0.90 * (len(ms) - 1))],
        "p99_ms": ms[int(0.99 * (len(ms) - 1))],
        "min_ms": ms[0],
        "max_ms": ms[-1],
    }


def batch_timing(name, spec, model, collator, out_dir):
    warm = load_records(DATA / "dev.jsonl", 64)
    if not spec["images"]:
        strip_images(warm)
    evaluate(model, warm, collator, out_dir / f"{name}_warmup_dev.jsonl")
    passes = []
    for repeat in range(REPEATS):
        groups = load_records(DATA / "test.jsonl", SMOKE)
        if not spec["images"]:
            strip_images(groups)
        rows = sum(len(v) for v in groups.values())
        dest = out_dir / f"{name}_test_rep{repeat}.jsonl"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        evaluate(model, groups, collator, dest)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        passes.append(
            {
                "repeat": repeat,
                "rows": rows,
                "events": rows // 4,
                "seconds": seconds,
                "ms_per_row": 1000 * seconds / rows,
                "ms_per_event_4_questions": 4000 * seconds / rows,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        print(json.dumps({"system": name, "batch_pass": passes[-1]}), flush=True)
    recorded = {row["id"]: row["logits"] for row in load_jsonl(spec["recorded"])}
    ours = {row["id"]: row["logits"] for row in load_jsonl(out_dir / f"{name}_test_rep0.jsonl")}
    diffs, agree = [], 0
    for key, logits in ours.items():
        theirs = recorded[key]
        diffs.append(max(abs(a - b) for a, b in zip(logits, theirs, strict=True)))
        agree += int(np.argmax(logits) == np.argmax(theirs))
    for repeat in range(1, REPEATS):
        (out_dir / f"{name}_test_rep{repeat}.jsonl").unlink()
    seconds = [p["seconds"] for p in passes]
    return {
        "eval_batch_size": 16,
        "images": spec["images"],
        "passes": passes,
        "mean_seconds": statistics.fmean(seconds),
        "std_seconds": statistics.pstdev(seconds),
        "mean_ms_per_row": statistics.fmean(p["ms_per_row"] for p in passes),
        "mean_ms_per_event_4_questions": statistics.fmean(p["ms_per_event_4_questions"] for p in passes),
        "fidelity_vs_original_gpu": {
            "rows_compared": len(ours),
            "rows_missing_in_recorded": len(set(ours) - set(recorded)),
            "argmax_agreement": agree / len(ours),
            "max_abs_logit_diff": max(diffs),
            "median_abs_logit_diff": statistics.median(diffs),
        },
    }


def split_events(path):
    events = {}
    for row in load_jsonl(path):
        events.setdefault(row["id"].split(":", 1)[1], {})[row["dataset"]] = row
    out = []
    for key, heads in sorted(events.items()):
        if all(name in heads for name in HEADS):
            out.append(
                {
                    "key": key,
                    "rows": heads,
                    "gold": {
                        "verb": LEGAL[heads["action"]["target"].index(1.0)],
                        "success": heads["p_success"]["target"][1] == 1.0,
                        "unsafe": heads["p_unsafe"]["target"][1] == 1.0,
                        "escalate": ESCALATE[heads["escalate"]["target"].index(1.0)],
                    },
                }
            )
    return out


def decision_timing(name, spec, model, out_dir):
    policy = json.loads(spec["policy"].read_text())
    api = ActOne(model, policy["temperatures"], source="actone" if name.startswith("actone") else "default")
    warm = split_events(DATA / "dev.jsonl")
    random.Random(0).shuffle(warm)
    for event in warm[:WARMUP]:
        row = event["rows"]["action"]
        image = Image.open(row["image"]).convert("RGB") if spec["images"] else None
        api.decide(image, row["state"])
    records = []
    for event in split_events(DATA / "test.jsonl")[:SMOKE]:
        row = event["rows"]["action"]
        decode_started = time.perf_counter()
        image = Image.open(row["image"]).convert("RGB") if spec["images"] else None
        decode_s = time.perf_counter() - decode_started
        decision = api.decide(image, row["state"])
        outcome = from_thresholds(decision, row["state"]["legal"], policy["thresholds"])
        gold = event["gold"]
        paid = cost_of(outcome, gold)
        records.append(
            {
                "key": event["key"],
                "gold": gold,
                "action": decision.action,
                "p_success": decision.p_success,
                "p_unsafe": decision.p_unsafe,
                "escalate": decision.escalate,
                "policy": outcome,
                "cost": paid,
                "regret": paid - oracle_cost(gold),
                "latency_s": decision.raw["latency_s"],
                "image_decode_s": decode_s,
                "logits": decision.raw["logits"],
            }
        )
    with (out_dir / f"{name}_two_pass_test.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    n = len(records)
    right = [r for r in records if r["action"] == r["gold"]["verb"]]
    decisions = [r["policy"] for r in records]
    executed = [r for r in records if r["policy"] == "execute"]
    failed = [r for r in executed if r["gold"]["unsafe"] or not r["gold"]["success"]]
    return {
        "images": spec["images"],
        "warmup_decisions": WARMUP,
        "temperatures": policy["temperatures"],
        "thresholds": policy["thresholds"],
        "latency": {
            "total": stats_ms([r["latency_s"]["total"] for r in records]),
            "pass1_action": stats_ms([r["latency_s"]["pass1"] for r in records]),
            "pass2_scored": stats_ms([r["latency_s"]["pass2"] for r in records]),
            "image_decode_excluded": stats_ms([r["image_decode_s"] for r in records]),
        },
        "two_pass_test": {
            "events": n,
            "action_accuracy": len(right) / n,
            "p_success_accuracy_all": sum((r["p_success"] >= 0.5) == r["gold"]["success"] for r in records) / n,
            "p_success_accuracy_when_action_correct": (
                sum((r["p_success"] >= 0.5) == r["gold"]["success"] for r in right) / len(right) if right else None
            ),
            "p_unsafe_accuracy_all": sum((r["p_unsafe"] >= 0.5) == r["gold"]["unsafe"] for r in records) / n,
            "escalate_accuracy_all": sum(r["escalate"] == r["gold"]["escalate"] for r in records) / n,
            "events_action_correct": len(right),
            "policy_decisions": {k: decisions.count(k) for k in ("execute", "ask_big", "ask_human", "abort", "hold")},
            "execute_coverage": len(executed) / n,
            "execute_risk": len(failed) / len(executed) if executed else None,
            "executed_with_wrong_action": sum(r["action"] != r["gold"]["verb"] for r in executed),
            "mean_cost": statistics.fmean(r["cost"] for r in records),
            "mean_regret": statistics.fmean(r["regret"] for r in records),
            "illegal_action_rate": sum(r["action"] not in LEGAL for r in records) / n,
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/latency_gpu2")
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--smoke", type=int, default=None, help="code check only: N rows per head, N events")
    args = parser.parse_args()
    global SMOKE
    SMOKE = args.smoke
    # Same as the recipe's cpu_threads (vendor final_evaluation); torch's default is
    # os.cpu_count(), which inside a CPU-quota container oversubscribes and throttles.
    torch.set_num_threads(CPU_THREADS)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "per_decision_latency_includes": "processor image resize/normalise and both forward passes; JPEG decode excluded",
        "test_split": str(DATA / "test.jsonl"),
        "repeats": REPEATS,
        "systems": {},
    }
    for name in args.systems:
        spec = SYSTEMS[name]
        started = time.perf_counter()
        model = load_model(spec["checkpoint"])
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        collator = DecisionCollator(MODEL)
        result = {"model_load_s": load_s, "checkpoint": str(spec["checkpoint"] or INIT)}
        result["batch"] = batch_timing(name, spec, model, collator, args.out)
        result["per_decision"] = decision_timing(name, spec, model, args.out)
        report["systems"][name] = result
        (args.out / "latency.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"system": name, "done": True}), flush=True)
        del model, collator
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"out": str(args.out / "latency.json")}), flush=True)


if __name__ == "__main__":
    main()
