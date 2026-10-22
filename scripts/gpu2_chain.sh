#!/bin/bash
# Second-L40S chain after seed 44: runs back to back so the GPU is never idle.
. ~/venv/bin/activate; cd ~/actone; export HF_HOME=/home/ubuntu/actone/.cache/hf
log(){ echo "$(date -u +%FT%TZ) $*" >> logs/chain.log; }
log "chain start; waiting for seed44 launcher"
while pgrep -f "output runs/actone_v4_s3000_seed4[4]" >/dev/null; do sleep 20; done
log "seed44 launcher exited"
D=data/processed/actone_v4; R=runs/actone_v4_s3000_seed44/seed-44
python eval/offline.py --calibration-predictions $R/calibration-predictions.jsonl --test-predictions $R/test-predictions.jsonl --calibration-labels $D/calibration.jsonl --test-labels $D/test.jsonl --out $R/heldout_policy.json > logs/eval_seed44.log 2>&1; log "seed44 offline eval rc=$?"
python scripts/gpu_latency.py --out runs/latency_gpu2 > logs/latency_gpu2.log 2>&1; log "latency rc=$?"
python scripts/legal_subset_eval.py --out runs/legal_subset_gpu2 > logs/legal_subset_gpu2.log 2>&1; log "legal subset rc=$?"
python - > logs/export_no_escalate.log 2>&1 <<'PY'
from pathlib import Path
import torch
from dohnuts.train import export_checkpoint
run = Path("runs/actone_v4_s3000_no_escalate/seed-42").resolve()
state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
export_checkpoint(state, run, run.parent / "checkpoint")
PY
log "no_escalate adapter export rc=$?"
log "chain done"
