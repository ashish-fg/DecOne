"""Train the thesis run and calibrate it, without Dohnuts acceptance tests.

run_experiment.py always calls tests/acceptance.py relative to the process
cwd. That file lives under vendor/dohnuts/tests and imports bub. This launcher
calls dohnuts.train directly: train, then evaluate. Evaluate fits temperatures
on the calibration split and scores the test split once, on the dev-selected
checkpoint. Vendor code is not edited.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from dohnuts.recipe import training_recipe

ROOT = Path("/home/ubuntu/actone")
MODEL = Path(
    "/home/ubuntu/actone/.cache/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/"
    "2fc06364715b967f1860aea9cf38778875588b17"
)
INIT = Path(
    "/home/ubuntu/actone/.cache/hf/hub/models--PsiACE--Dohnuts-0.1.0-0.8B/snapshots/"
    "959828a3d9cf3dacbd1491e11bb1cd36d12709ac"
)
DATA = ROOT / "data" / "processed" / "actone"
OUTPUT = ROOT / "runs" / "actone"
STEPS = 1500


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = args.data if args.data.is_absolute() else ROOT / args.data
    output = args.output if args.output.is_absolute() else ROOT / args.output
    steps = args.steps
    for path in (MODEL, INIT / "adapter.safetensors", data / "train.jsonl", data / "dev.jsonl"):
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "recipe.json").exists():
        raise FileExistsError(f"{output} already exists; refusing to reuse it")
    recipe = training_recipe(model=MODEL, data=data, steps=steps, seed=args.seed)
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    config = output / f"recipe-seed-{args.seed}.json"
    config.write_text(json.dumps(recipe, indent=2) + "\n")
    (output / "recipe.json").write_text(
        json.dumps({"recipe": recipe, "launcher": "scripts/launch_thesis.py", "acceptance": "skipped"}, indent=2)
        + "\n"
    )
    run = output / f"seed-{args.seed}"
    env = os.environ.copy()
    env["ROCR_VISIBLE_DEVICES"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    env["TRITON_CACHE_DIR"] = str(ROOT / ".cache" / "triton")
    env["HF_HOME"] = str(ROOT / ".cache" / "hf")
    env["TOKENIZERS_PARALLELISM"] = "false"
    train = [
        sys.executable,
        "-m",
        "dohnuts.train",
        "train",
        "--config",
        str(config),
        "--run",
        str(run),
        "--initialize-from",
        str(INIT),
    ]
    print(json.dumps({"stage": "train", "steps": steps, "output": str(output), "data": str(data)}), flush=True)
    subprocess.run(train, cwd=ROOT, env=env, check=True)
    evaluate = [
        sys.executable,
        "-m",
        "dohnuts.train",
        "evaluate",
        "--config",
        str(config),
        "--run",
        str(run),
    ]
    print(json.dumps({"stage": "evaluate", "calibration": "calibration.jsonl", "held_out": "test.jsonl"}), flush=True)
    subprocess.run(evaluate, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
