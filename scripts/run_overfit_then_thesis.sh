#!/bin/bash
# Resume after labels and frames exist. Rank stays 8.
set -u
cd "$HOME/actone"
# shellcheck disable=SC1091
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
CKPT="$HOME/actone/.cache/hf/hub/models--PsiACE--Dohnuts-0.1.0-0.8B/snapshots/959828a3d9cf3dacbd1491e11bb1cd36d12709ac"
BASE="$HOME/actone/.cache/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
echo "OVERFIT_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python vendor/dohnuts/scripts/run_experiment.py \
  --initialize-from "$CKPT" \
  --model "$BASE" \
  --data "$HOME/actone/data/processed/overfit" \
  --output "$HOME/actone/runs/actone_overfit" \
  --steps 200
echo "OVERFIT_EXIT $? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python - << 'PY'
import json
from pathlib import Path
path = Path("runs/actone_overfit/seed-42/metrics.jsonl")
if not path.exists():
    Path("logs/overfit_gate.txt").write_text(json.dumps({"pass": False, "reason": "missing metrics"}) + "\n")
    raise SystemExit(0)
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
trains = [row for row in rows if row.get("kind") == "train"]
devs = [row for row in rows if row.get("kind") == "dev"]
action = devs[-1]["metrics"].get("action", {}) if devs else {}
acc = action.get("accuracy")
loss_down = bool(trains) and trains[-1]["loss"] < trains[0]["loss"]
text = {
    "pass": bool(acc is not None and acc >= 0.85 and loss_down),
    "loss_first": None if not trains else trains[0]["loss"],
    "loss_last": None if not trains else trains[-1]["loss"],
    "loss_down": loss_down,
    "action_accuracy": acc,
    "dev_step": None if not devs else devs[-1]["step"],
    "macro_accuracy": None if not devs else devs[-1]["metrics"].get("macro_accuracy"),
}
Path("logs/overfit_gate.txt").write_text(json.dumps(text, indent=2) + "\n")
print(json.dumps(text))
PY
GATE=$(python -c 'import json; print(json.load(open("logs/overfit_gate.txt")).get("pass"))')
if [ "$GATE" != "True" ]; then
  echo "THESIS_SKIPPED overfit did not pass $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi
echo "THESIS_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python vendor/dohnuts/scripts/run_experiment.py \
  --initialize-from "$CKPT" \
  --model "$BASE" \
  --data "$HOME/actone/data/processed/actone" \
  --output "$HOME/actone/runs/actone" \
  --steps 1500
echo "THESIS_EXIT $? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PIPELINE_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
