#!/bin/bash
# Second-L40S chain after seed 44: runs back to back so the GPU is never idle.
. ~/venv/bin/activate; cd ~/actone; export HF_HOME=/home/ubuntu/actone/.cache/hf
log(){ echo "$(date -u +%FT%TZ) $*" >> logs/chain2.log; }
log "chain2 start (latency rerun with 8 torch threads)"
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
