#!/bin/bash
. ~/venv/bin/activate; cd ~/actone; export HF_HOME=/home/ubuntu/actone/.cache/hf
log(){ echo "$(date -u +%FT%TZ) $*" >> logs/queue.log; }
log "queue start; waiting for no_escalate launcher"
while pgrep -f "output runs/actone_v4_s3000_no_[e]scalate" >/dev/null; do sleep 30; done
log "no_escalate launcher exited"
R=runs/actone_v4_s3000_no_escalate/seed-42; D=data/processed/actone_v4
python eval/no_escalate.py --calibration-predictions $R/calibration-predictions.jsonl --test-predictions $R/test-predictions.jsonl --calibration-labels $D/calibration.jsonl --test-labels $D/test.jsonl --out $R/heldout_policy.json > logs/eval_no_escalate.log 2>&1; log "no_escalate eval rc=$?"
python eval/no_escalate.py --calibration-predictions runs/actone_v4_s3000/seed-42/calibration-predictions.jsonl --test-predictions runs/actone_v4_s3000/seed-42/test-predictions.jsonl --calibration-labels $D/calibration.jsonl --test-labels $D/test.jsonl --out runs/actone_v4_s3000_no_escalate/kept_escalate_ignored_policy.json > logs/eval_kept_escalate_ignored.log 2>&1; log "kept-ignore-escalate eval rc=$?"
python scripts/gpu_latency.py --out runs/latency_gpu2 > logs/latency_gpu2.log 2>&1; log "latency rc=$?"
python scripts/launch_thesis.py --data data/processed/actone_v4 --output runs/actone_v4_s3000_seed44 --steps 3000 --seed 44 > logs/thesis_v4_s3000_seed44.log 2>&1; log "seed44 train+eval rc=$?"
R=runs/actone_v4_s3000_seed44/seed-44
python eval/offline.py --calibration-predictions $R/calibration-predictions.jsonl --test-predictions $R/test-predictions.jsonl --calibration-labels $D/calibration.jsonl --test-labels $D/test.jsonl --out $R/heldout_policy.json > logs/eval_seed44.log 2>&1; log "seed44 offline rc=$?"
log "queue done"
