"""Shuffle train targets within each head. Dev, calibration, and test stay real.

The copies are new files. The live thesis jsonl is only read.
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/ubuntu/actone")


def main():
    source = ROOT / "data" / "processed" / "actone"
    dest = ROOT / "data" / "processed" / "actone_shuffled"
    if (ROOT / "runs" / "actone" / "recipe.json").exists() and dest.resolve() == source.resolve():
        raise SystemExit("refusing to overwrite the live split")
    dest.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    rows = [json.loads(line) for line in (source / "train.jsonl").read_text().splitlines() if line.strip()]
    for row in rows:
        groups[row["dataset"]].append(row)
    rng = random.Random(0)
    for name, subset in groups.items():
        targets = [row["target"] for row in subset]
        rng.shuffle(targets)
        same = sum(row["target"] == target for row, target in zip(subset, targets))
        for row, target in zip(subset, targets):
            row["target"] = target
        print(json.dumps({"dataset": name, "n": len(subset), "unchanged_after_shuffle": same}))
    with (dest / "train.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    for split in ("dev", "calibration", "test"):
        shutil.copy2(source / f"{split}.jsonl", dest / f"{split}.jsonl")
    print(json.dumps({"out": str(dest), "train_rows": len(rows)}))


if __name__ == "__main__":
    main()
