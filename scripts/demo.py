"""Defence demo: 10 labelled held-out test images in, ActOneDecision JSON out,
ActOne (kept checkpoint) vs zero-shot Dohnuts on the same inputs.

No training, no refitting. Each model uses its own temperatures and policy
thresholds that were fitted on the calibration split and frozen before test
(runs/actone_v4_s3000/seed-42/heldout_policy.json for ActOne,
runs/actone_v4_s3000/ablation/zero_shot_policy.json for zero-shot Dohnuts).

Event selection rule (fixed, label-only, never looks at predictions): from the
TEST split, for each verb in VERBS and each gold escalate class in ("act",
"ask_big"), draw one event with random.Random(SEED).choice over the events
sorted by (episode, frame), skipping episodes already drawn. If a (verb,
escalate) cell is empty, draw any unused event with that verb. 5 verbs x 2 = 10.
"""

import argparse
import gc
import json
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from actone.api import ActOne, action_question, scored_questions, scored_state, softmax  # noqa: E402
from actone.policy import from_thresholds  # noqa: E402
from actone.runtime import DOHNUTS, QWEN, default_device, load_decision_model, snapshot  # noqa: E402
from eval.offline import cost_of  # noqa: E402

SEED = 42
VERBS = ["reach", "grasp", "lift", "retract", "home"]
ESCALATE_CELLS = ["act", "ask_big"]
LEGAL = ["reach", "grasp", "lift", "place", "retract", "home", "abort"]
ESCALATE = ["act", "ask_big", "ask_human", "abort"]
HEADS = ["action", "p_success", "p_unsafe", "escalate"]


def load_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_events(path):
    events = {}
    for row in load_jsonl(path):
        key = row["id"].split(":", 1)[1]
        events.setdefault(key, {})[row["dataset"]] = row
    out = []
    for key, heads in events.items():
        if not all(name in heads for name in HEADS):
            continue
        episode, frame, _ = key.split(":")
        out.append(
            {
                "key": key,
                "episode": int(episode),
                "frame": int(frame),
                "rows": heads,
                "gold": {
                    "action": LEGAL[heads["action"]["target"].index(1.0)],
                    "p_success": heads["p_success"]["target"][1] == 1.0,
                    "p_unsafe": heads["p_unsafe"]["target"][1] == 1.0,
                    "escalate": ESCALATE[heads["escalate"]["target"].index(1.0)],
                },
            }
        )
    out.sort(key=lambda event: (event["episode"], event["frame"], event["key"]))
    return out


def select(events, seed=SEED):
    rng = random.Random(seed)
    used, picked = set(), []
    for verb in VERBS:
        for cell in ESCALATE_CELLS:
            pool = [
                e
                for e in events
                if e["gold"]["action"] == verb
                and e["gold"]["escalate"] == cell
                and e["episode"] not in used
            ]
            label = f"{verb}/{cell}"
            if not pool:
                pool = [e for e in events if e["gold"]["action"] == verb and e["episode"] not in used]
                label += " (empty in test; fallback to any escalate)"
            event = rng.choice(pool)
            used.add(event["episode"])
            picked.append({**event, "cell": label})
    return picked


def check_prompts(event):
    """The demo's rebuilt questions must equal the frozen test rows."""
    rows, gold = event["rows"], event["gold"]["action"]
    state = rows["action"]["state"]
    assert action_question(state["legal"]) == rows["action"]["question"], event["key"]
    rebuilt = scored_questions(gold)
    for name in ("p_success", "p_unsafe", "escalate"):
        assert rebuilt[name] == rows[name]["question"], (event["key"], name)
        assert scored_state(state, gold) == rows[name]["state"], (event["key"], name)


def image_path(artifacts, row):
    return artifacts / "data" / "frames" / "libero" / Path(row["image"]).name


def correct(decision, gold):
    return {
        "action": decision["action"] == gold["action"],
        "p_success": (decision["p_success"] >= 0.5) == gold["p_success"],
        "p_unsafe": (decision["p_unsafe"] >= 0.5) == gold["p_unsafe"],
        "escalate": decision["escalate"] == gold["escalate"],
    }


def run_model(name, model_kwargs, policy, recorded_path, events, artifacts, device):
    started = time.perf_counter()
    model = load_decision_model(snapshot(QWEN), device=device, **model_kwargs)
    load_s = time.perf_counter() - started
    api = ActOne(model, policy["temperatures"], source="actone" if name == "actone" else "default")
    recorded = {row["id"]: row["logits"] for row in load_jsonl(recorded_path)}
    first = events[0]
    with Image.open(image_path(artifacts, first["rows"]["action"])) as image:
        api.decide(image.convert("RGB"), first["rows"]["action"]["state"])  # warm-up, not recorded
    results = []
    for event in events:
        row = event["rows"]["action"]
        with Image.open(image_path(artifacts, row)) as image:
            decision = api.decide(image.convert("RGB"), row["state"])
        payload = decision.to_dict()
        outcome = from_thresholds(decision, row["state"]["legal"], policy["thresholds"])
        gold = event["gold"]
        payload["raw"]["policy"] = outcome
        payload["raw"]["policy_cost"] = cost_of(
            outcome,
            {
                "verb": gold["action"],
                "success": gold["p_success"],
                "unsafe": gold["p_unsafe"],
                "escalate": gold["escalate"],
            },
        )
        # Pass-2 prompts equal the recorded offline rows only when pass 1 picked the gold verb.
        compared = ["action"] + (["p_success", "p_unsafe", "escalate"] if decision.action == gold["action"] else [])
        fidelity = {}
        for head in compared:
            ours_logits = payload["raw"]["logits"][head]
            gpu_logits = recorded[event["rows"][head]["id"]]
            temperature = policy["temperatures"][head]
            ours_p, gpu_p = softmax(ours_logits, temperature), softmax(gpu_logits, temperature)
            fidelity[head] = {
                "argmax_match": ours_p.index(max(ours_p)) == gpu_p.index(max(gpu_p)),
                "max_abs_prob_diff": max(abs(a - b) for a, b in zip(ours_p, gpu_p, strict=True)),
                "max_abs_logit_diff": max(abs(a - b) for a, b in zip(ours_logits, gpu_logits, strict=True)),
            }
        payload["raw"]["fidelity_vs_gpu_eval"] = fidelity
        results.append(
            {
                "decision": payload,
                "policy": outcome,
                "correct": correct(payload, gold),
                "latency_s": payload["raw"]["latency_s"]["total"],
            }
        )
        print(json.dumps({"model": name, "key": event["key"], "action": decision.action, "policy": outcome}), flush=True)
    del api, model
    gc.collect()
    return results, load_s


def fmt_p(value):
    return f"{value:.2f}"


def summary_counts(rows, model):
    return {head: sum(row[model]["correct"][head] for row in rows) for head in HEADS}


def write_markdown(path, report):
    rows = report["events"]
    n = len(rows)
    lines = [
        "# ActOne defence demo: 10 held-out test images",
        "",
        "Ten LIBERO keyframes from the held-out **test** split (episodes never used for training, "
        "checkpoint selection, calibration, or threshold fitting) go through real two-pass inference. "
        "Pass 1 picks an action from the caller's `legal[]`; pass 2 names that action and asks "
        "`p_success`, `p_unsafe`, and `escalate`. Each model uses its own calibration-fitted, frozen "
        "temperatures and policy thresholds. Nothing was retrained or refitted for this demo.",
        "",
        f"- ActOne: `{report['meta']['actone']['checkpoint']}` (rank-8 LoRA + scorer, dev macro 0.9375).",
        "- Zero-shot: `PsiACE/Dohnuts-0.1.0-0.8B` with no ActOne LoRA, same prompts, same two passes.",
        f"- Selection rule: {report['meta']['selection_rule']}",
        f"- Hardware: CPU only ({report['meta']['device_info']}), BF16, batch of one image.",
        "",
        "## Per-image results",
        "",
        "Each cell shows action / p_success / p_unsafe / escalate, then the frozen policy outcome. "
        "A head that disagrees with the gold label is marked **[miss]** (p values are read at 0.5).",
        "",
        "| # | Image | Gold | ActOne | Zero-shot Dohnuts | ActOne latency (s) |",
        "|---|---|---|---|---|---:|",
    ]

    def cell(result):
        d, ok = result["decision"], result["correct"]

        def mark(head, text):
            return text if ok[head] else f"{text} **[miss]**"

        return "<br>".join(
            [
                mark("action", f"`{d['action']}` ({fmt_p(d['action_probs'][d['action']])})"),
                mark("p_success", f"p_s {fmt_p(d['p_success'])}"),
                mark("p_unsafe", f"p_u {fmt_p(d['p_unsafe'])}"),
                mark("escalate", f"`{d['escalate']}`"),
                f"policy: `{result['policy']}`",
            ]
        )

    for index, row in enumerate(rows, 1):
        gold = row["gold"]
        gold_cell = "<br>".join(
            [
                f"`{gold['action']}`",
                f"success {str(gold['p_success']).lower()}",
                f"unsafe {str(gold['p_unsafe']).lower()}",
                f"`{gold['escalate']}`",
                f"<sub>ep {row['episode']} frame {row['frame']}</sub>",
            ]
        )
        thumb = f'<img src="images/{row["image_file"]}" width="112" alt="{row["key"]}">'
        lines.append(
            f"| {index} | {thumb} | {gold_cell} | {cell(row['actone'])} | {cell(row['zero_shot'])} | "
            f"{row['actone']['latency_s']:.2f} |"
        )
    actone, zero = summary_counts(rows, "actone"), summary_counts(rows, "zero_shot")
    lat = report["latency"]
    lines += [
        "",
        "## Correct out of 10, per head",
        "",
        "| Head | ActOne | Zero-shot Dohnuts | Gold label mix in these 10 |",
        "|---|---:|---:|---|",
    ]
    mixes = {
        "action": ", ".join(f"{v} {sum(r['gold']['action'] == v for r in rows)}" for v in VERBS),
        "p_success": f"true {sum(r['gold']['p_success'] for r in rows)}, false {sum(not r['gold']['p_success'] for r in rows)}",
        "p_unsafe": f"true {sum(r['gold']['p_unsafe'] for r in rows)}, false {sum(not r['gold']['p_unsafe'] for r in rows)}",
        "escalate": ", ".join(f"{c} {sum(r['gold']['escalate'] == c for r in rows)}" for c in ESCALATE),
    }
    for head in HEADS:
        lines.append(f"| {head} | {actone[head]}/{n} | {zero[head]}/{n} | {mixes[head]} |")
    policy_counts = {
        model: {o: sum(r[model]["policy"] == o for r in rows) for o in ("execute", "ask_big", "ask_human", "abort", "hold")}
        for model in ("actone", "zero_shot")
    }
    cost = {model: sum(r[model]["decision"]["raw"]["policy_cost"] for r in rows) / n for model in ("actone", "zero_shot")}
    lines += [
        "",
        "Policy outcomes (frozen thresholds): ActOne "
        + ", ".join(f"{k} {v}" for k, v in policy_counts["actone"].items() if v)
        + "; zero-shot " + ", ".join(f"{k} {v}" for k, v in policy_counts["zero_shot"].items() if v)
        + f". Mean frozen cost on these 10: ActOne {cost['actone']:.2f}, zero-shot {cost['zero_shot']:.2f}.",
        "",
        "## CPU latency per decision (both passes, model already loaded)",
        "",
        "| Model | Mean (s) | Median (s) | Min (s) | Max (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, label in (("actone", "ActOne"), ("zero_shot", "Zero-shot Dohnuts")):
        s = lat[model]
        lines.append(f"| {label} | {s['mean']:.2f} | {s['median']:.2f} | {s['min']:.2f} | {s['max']:.2f} |")
    lines += [
        "",
        f"Model load (not included above): ActOne {report['meta']['actone']['load_s']:.1f} s, "
        f"zero-shot {report['meta']['zero_shot']['load_s']:.1f} s. A warm-up decision was run and discarded "
        "before timing. The thesis GPU numbers (L40S, batched, fused kernels) are separate and unchanged.",
        "",
        "## Honest reading",
        "",
    ]
    misses = []
    for index, row in enumerate(rows, 1):
        for head in HEADS:
            if not row["actone"]["correct"][head]:
                d = row["actone"]["decision"]
                got = {
                    "action": d["action"],
                    "p_success": fmt_p(d["p_success"]),
                    "p_unsafe": fmt_p(d["p_unsafe"]),
                    "escalate": d["escalate"],
                }[head]
                want = row["gold"][head]
                want = str(want).lower() if isinstance(want, bool) else want
                misses.append(f"#{index} {head} (gold `{want}`, ActOne `{got}`)")
    lines.append(
        f"- ActOne misses on these 10: {'; '.join(misses) if misses else 'none'}."
    )
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    Path(path).write_text("\n".join(lines))


def latency_stats(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "from-gpu" / "gpu-actone")
    parser.add_argument("--run", default="runs/actone_v4_s3000")
    parser.add_argument("--data", default="data/processed/actone_v4")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "demo")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = args.device or default_device()
    artifacts = args.artifacts.resolve()
    run = artifacts / args.run
    seed_dir = run / "seed-42"
    checkpoint = seed_dir / "best.pt"
    actone_policy = json.loads((seed_dir / "heldout_policy.json").read_text())
    zero_policy = json.loads((run / "ablation" / "zero_shot_policy.json").read_text())
    vendor_temperatures = json.loads((seed_dir / "temperatures.json").read_text())

    events = select(test_events(artifacts / args.data / "test.jsonl"), args.seed)
    for event in events:
        check_prompts(event)

    actone_results, actone_load = run_model(
        "actone",
        {"checkpoint": checkpoint},
        actone_policy,
        seed_dir / "test-predictions.jsonl",
        events,
        artifacts,
        device,
    )
    zero_results, zero_load = run_model(
        "zero_shot",
        {"adapter_dir": snapshot(DOHNUTS)},
        zero_policy,
        run / "ablation" / "zero_shot_test.jsonl",
        events,
        artifacts,
        device,
    )

    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (event, ours, zero) in enumerate(zip(events, actone_results, zero_results, strict=True), 1):
        action_row = event["rows"]["action"]
        source = image_path(artifacts, action_row)
        name = f"{index:02d}_{source.name}"
        shutil.copy2(source, images / name)
        rows.append(
            {
                "index": index,
                "key": event["key"],
                "episode": event["episode"],
                "frame": event["frame"],
                "selection_cell": event["cell"],
                "image_file": name,
                "source_image": str(source.relative_to(artifacts)),
                "task": action_row["state"]["task"],
                "state": action_row["state"],
                "legal": action_row["state"]["legal"],
                "gold": event["gold"],
                "actone": ours,
                "zero_shot": zero,
            }
        )

    checks = [
        check
        for row in rows
        for model in ("actone", "zero_shot")
        for check in row[model]["decision"]["raw"]["fidelity_vs_gpu_eval"].values()
    ]
    fidelity = {
        "compared": len(checks),
        "argmax_match": sum(check["argmax_match"] for check in checks),
        "max_abs_prob_diff": max(check["max_abs_prob_diff"] for check in checks),
    }
    cpu = "unknown CPU"
    try:
        cpu = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    report = {
        "meta": {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split": "test",
            "seed": args.seed,
            "selection_rule": (
                f"test split only; for each verb in {VERBS} and each gold escalate in {ESCALATE_CELLS}, "
                f"random.Random({args.seed}).choice over events sorted by (episode, frame), "
                "one event per episode; labels only, predictions never consulted."
            ),
            "device": device,
            "device_info": f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__}",
            "inference": "two-pass; pass 1 action over legal[], pass 2 p_success/p_unsafe/escalate naming the pass-1 action",
            "correct_rule": "action and escalate: argmax equals gold; p_success and p_unsafe: (p >= 0.5) equals gold",
            "actone": {
                "checkpoint": str(checkpoint.relative_to(artifacts)),
                "temperatures": actone_policy["temperatures"],
                "thresholds": actone_policy["thresholds"],
                "vendor_temperatures_not_used": vendor_temperatures,
                "load_s": actone_load,
            },
            "zero_shot": {
                "checkpoint": f"{DOHNUTS[0]}@{DOHNUTS[1]}",
                "temperatures": zero_policy["temperatures"],
                "thresholds": zero_policy["thresholds"],
                "load_s": zero_load,
            },
            "base_model": f"{QWEN[0]}@{QWEN[1]}",
        },
        "latency": {
            "actone": latency_stats([r["actone"]["latency_s"] for r in rows]),
            "zero_shot": latency_stats([r["zero_shot"]["latency_s"] for r in rows]),
        },
        "fidelity_vs_gpu_eval": fidelity,
        "summary_correct_of_10": {
            "actone": summary_counts(rows, "actone"),
            "zero_shot": summary_counts(rows, "zero_shot"),
        },
        "notes": [
            "Ten events are a demonstration, not a measurement; the thesis numbers are the 1,364-event test results "
            "(ActOne action 0.852, escalate 0.912, p_success 0.946, p_unsafe 1.000), which this demo does not change.",
            "Every p_unsafe gold label in this dataset is false (no LIBERO frame crosses the 5 cm / 0.1 s rule), so "
            "a correct p_unsafe here only shows the model predicts 'not unsafe'; it is not evidence of danger detection.",
            "Gold escalate is only ever act or ask_big in this dataset (no ask_human or abort labels exist), so the "
            "escalate head is effectively act vs ask_big.",
            "Gold p_success describes the gold verb's executed outcome. When pass 1 picks a different verb, pass 2 is "
            "asked about that other verb, so its p_success is scored against a label for a different action.",
            "The offline test evaluation scored pass-2 heads with the gold verb named; this demo names the predicted "
            "verb (true two-pass), so pass-2 heads can differ from the offline rows when pass 1 is wrong.",
            "CPU inference uses transformers' PyTorch Qwen3.5 kernels (flash-linear-attention needs Triton and does "
            "not run on CPU). Against the GPU evaluation logits for identical prompts, the CPU path agreed on "
            f"argmax {fidelity['argmax_match']}/{fidelity['compared']} times with a largest calibrated-probability "
            f"difference of {fidelity['max_abs_prob_diff']:.4f} (per-decision detail in decisions.json).",
        ]
        + [
            "CPU BF16 results wobble slightly between identical runs (thread-level reduction order). Two runs on "
            "the 4-thread build VM differed by up to 0.015 in calibrated probability for ActOne and 0.063 for "
            "zero-shot (whose p_unsafe temperature of 0.17 amplifies logit noise), so a value sitting on a "
            "threshold can flip between runs."
        ]
        + [
            f"Selection cell `{row['selection_cell']}` (image #{row['index']}): the test split has no event for that "
            "cell, so the fixed rule drew another event with the same verb."
            for row in rows
            if "fallback" in row["selection_cell"]
        ],
        "events": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "decisions.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(args.out / "demo.md", report)
    print(json.dumps({"out": str(args.out), "summary": report["summary_correct_of_10"], "latency_mean_s": {
        k: v["mean"] for k, v in report["latency"].items()}}))


if __name__ == "__main__":
    main()
