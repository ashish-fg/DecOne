"""Zero-shot Dohnuts and no-image inference. Does not train and does not edit vendor."""

import json
import time
from pathlib import Path

from dohnuts.model import DecisionModel
from dohnuts.train import evaluate, load_checkpoint
from dohnuts.training_data import DecisionCollator, load_records

ROOT = Path("/home/ubuntu/actone")
MODEL = Path(
    "/home/ubuntu/actone/.cache/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/"
    "2fc06364715b967f1860aea9cf38778875588b17"
)
INIT = Path(
    "/home/ubuntu/actone/.cache/hf/hub/models--PsiACE--Dohnuts-0.1.0-0.8B/snapshots/"
    "959828a3d9cf3dacbd1491e11bb1cd36d12709ac"
)


def score(model, collator, data, split, dest, drop_images):
    groups = load_records(data / f"{split}.jsonl")
    if drop_images:
        for rows in groups.values():
            for row in rows:
                row.pop("image", None)
    started = time.perf_counter()
    evaluate(model, groups, collator, dest)
    elapsed = time.perf_counter() - started
    count = sum(len(rows) for rows in groups.values())
    return {"split": split, "rows": count, "seconds": elapsed, "seconds_per_row": elapsed / max(count, 1), "images": not drop_images}


def zero_shot(data, dest_dir):
    model = DecisionModel(MODEL)
    model.enable_lora(checkpointing=False)
    model.load_adapter(INIT)
    model.merge()
    collator = DecisionCollator(MODEL)
    timing = []
    for split in ("calibration", "test"):
        timing.append(score(model, collator, data, split, dest_dir / f"zero_shot_{split}.jsonl", False))
    return timing


def no_image(checkpoint, data, dest_dir):
    model = DecisionModel(MODEL)
    model.enable_lora(checkpointing=False)
    load_checkpoint(model, checkpoint)
    model.merge()
    collator = DecisionCollator(MODEL)
    timing = []
    for split in ("calibration", "test"):
        timing.append(score(model, collator, data, split, dest_dir / f"no_image_{split}.jsonl", True))
    return timing


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", choices=["zero-shot", "no-image"], required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "actone")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "actone" / "seed-42" / "best.pt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.which == "zero-shot":
        timing = zero_shot(args.data, args.out)
    else:
        timing = no_image(args.checkpoint, args.data, args.out)
    (args.out / f"{args.which}_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing))


if __name__ == "__main__":
    main()
