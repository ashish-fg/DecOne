"""No-escalate-head ablation data: the v4 splits with every escalate row removed.

The copies are new files; the live v4 jsonl is only read. action, p_success and
p_unsafe rows are byte-identical to v4. Gold escalate stays in the v4 files and
is read back only by eval/no_escalate.py for the frozen-cost policy.
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path("/home/ubuntu/actone")
SPLITS = ("train", "dev", "calibration", "test")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "processed" / "actone_v4")
    parser.add_argument("--dest", type=Path, default=ROOT / "data" / "processed" / "actone_v4_no_escalate")
    args = parser.parse_args()
    if args.dest.resolve() == args.source.resolve():
        raise SystemExit("refusing to overwrite the live split")
    args.dest.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        kept, dropped = Counter(), 0
        with (args.source / f"{split}.jsonl").open() as src, (args.dest / f"{split}.jsonl").open("w") as dst:
            for line in src:
                if not line.strip():
                    continue
                if json.loads(line)["dataset"] == "escalate":
                    dropped += 1
                    continue
                kept[json.loads(line)["dataset"]] += 1
                dst.write(line if line.endswith("\n") else line + "\n")
        print(json.dumps({"split": split, "kept": dict(kept), "dropped_escalate": dropped}))


if __name__ == "__main__":
    main()
