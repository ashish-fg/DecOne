"""Turn one labeled keyframe into four Dohnuts JSONL rows.

Questions match ACTONE_CORE. Training noul and escalate instructions name the
gold verb. Splits stay the episode splits from collect_labels. The overfit
directory repeats the train rows as dev so the trainer can score those rows.
Its calibration and test files are a slice of the same rows and are not a
held-out result.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "jsonl" / "events.jsonl"
LEGAL = ["reach", "grasp", "lift", "place", "retract", "home", "abort"]
ESCALATE = ["act", "ask_big", "ask_human", "abort"]


def one_hot(index, width):
    target = [0.0] * width
    target[index] = 1.0
    return target


def state_for(event, phase):
    return {
        "body": "arm_franka",
        "pack": "tabletop_v1",
        "phase": phase,
        "skill": phase,
        "skill_age_s": event["skill_age_s"],
        "ee_z": event["ee_z"],
        "stuck_s": event["stuck_s"],
        "last_outcome": event["last_outcome"],
        "legal": LEGAL,
        "task": event["task"],
    }


def rows_for(event, previous_verb):
    verb = event["verb"]
    # The action question must not be handed its own answer. phase/skill here
    # are the previous keyframe, which a caller would already know.
    action_state = state_for(event, previous_verb)
    # noul and escalate name the gold verb in the instruction, by contract.
    scored_state = state_for(event, verb)
    group = f"libero:{event['episode_index']}"
    image = event["image"]
    base = f"{event['episode_index']}:{event['frame_index']}:{verb}"
    shared = {
        "group": group,
        "split": event["split"],
        "image": image,
        "aliases": [group],
    }
    action = {
        **shared,
        "state": action_state,
        "id": f"action:{base}",
        "dataset": "action",
        "question": {
            "type": "choice",
            "instructions": (
                "You are the tactical layer of a robot. "
                "Pick exactly one legal action. Do not invent actions. "
                "If none is appropriate, pick abort."
            ),
            "criteria": {name: name for name in LEGAL},
        },
        "target": one_hot(LEGAL.index(verb), len(LEGAL)),
    }
    success = {
        **shared,
        "state": scored_state,
        "id": f"p_success:{base}",
        "dataset": "p_success",
        "question": {
            "type": "noul",
            "instructions": (
                f"If {verb} is executed now, will it succeed within the skill "
                "timeout without collision or leaving the workspace?"
            ),
        },
        "target": one_hot(int(event["p_success"]), 2),
    }
    unsafe = {
        **shared,
        "state": scored_state,
        "id": f"p_unsafe:{base}",
        "dataset": "p_unsafe",
        "question": {
            "type": "noul",
            "instructions": (
                f"Would executing {verb} now risk collision, limit violation, "
                "or energy exhaustion?"
            ),
        },
        "target": one_hot(int(event["p_unsafe"]), 2),
    }
    escalate = {
        **shared,
        "state": scored_state,
        "id": f"escalate:{base}",
        "dataset": "escalate",
        "question": {
            "type": "choice",
            "instructions": (
                f"The chosen action is {verb}. "
                "act = execute it; ask_big = call a larger vision model; "
                "ask_human = stop and request a human; abort = safe stop."
            ),
            "criteria": {
                "act": "Execute the chosen legal action now.",
                "ask_big": "Ambiguous scene; ask a larger model.",
                "ask_human": "Risk or ambiguity too high; ask a person.",
                "abort": "Do not act.",
            },
        },
        "target": one_hot(ESCALATE.index(event["escalate"]), len(ESCALATE)),
    }
    return [action, success, unsafe, escalate]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "actone")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    live = ROOT / "runs" / "actone" / "recipe.json"
    default_out = ROOT / "data" / "processed" / "actone"
    if live.exists() and args.out.resolve() == default_out.resolve():
        raise SystemExit("refusing to overwrite data/processed/actone while runs/actone is live")
    events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    thesis = {split: [] for split in ["train", "dev", "calibration", "test"]}
    overfit_rows = []
    by_episode = {}
    for event in events:
        by_episode.setdefault(event["episode_index"], []).append(event)
    previous = {}
    for rows in by_episode.values():
        rows.sort(key=lambda item: (item["frame_index"], item["verb"]))
        earlier = "none"
        for event in rows:
            previous[id(event)] = earlier
            earlier = event["verb"]
    for event in events:
        built = rows_for(event, previous[id(event)])
        thesis[event["split"]].extend(built)
        if event.get("overfit"):
            overfit_rows.extend(built)
    for split, rows in thesis.items():
        if not rows:
            raise ValueError(f"no rows for split {split}")
        write_jsonl(args.out / f"{split}.jsonl", rows)
    manifest = {
        "thesis_rows": {split: len(rows) for split, rows in thesis.items()},
        "events": str(args.events),
        "out": str(args.out),
        "thesis_note": "train, dev, calibration, and test are disjoint episode sets",
    }
    if args.out.resolve() == default_out.resolve():
        # Memorization set. Dev repeats it. Calibration and test are a short slice
        # of the same rows so the trainer can start; they are not held-out.
        overfit_dir = ROOT / "data" / "processed" / "overfit"
        write_jsonl(overfit_dir / "train.jsonl", overfit_rows)
        write_jsonl(overfit_dir / "dev.jsonl", overfit_rows)
        slice_rows = overfit_rows[: 8 * 4]
        write_jsonl(overfit_dir / "calibration.jsonl", slice_rows)
        write_jsonl(overfit_dir / "test.jsonl", slice_rows)
        manifest["overfit_rows"] = len(overfit_rows)
        manifest["overfit_images"] = len(overfit_rows) // 4
        manifest["overfit_note"] = (
            "dev repeats train; calibration and test are a slice of those rows, not held-out"
        )
    manifest_path = args.manifest or (
        ROOT / "data" / "jsonl" / "split_manifest.json"
        if args.out.resolve() == default_out.resolve()
        else args.out / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
