# ActOne

ActOne fine-tunes a small multimodal System One model to score a caller-supplied
legal action set for a robot arm and to refuse, with probabilities calibrated on
executed outcomes. BITS F421T thesis, Ashish Bhardwaj (2023A7PS0324U), BITS Pilani
Dubai, supervisor Dr. Mithun Mukherjee.

`f(image, state_json, legal[]) -> ActOneDecision` with `action`, `action_probs`,
`p_success`, `p_unsafe`, `escalate` (`act` / `ask_big` / `ask_human` / `abort`),
`escalate_probs`. It is a rank-8 LoRA plus candidate scorer on
[Dohnuts-0.1.0-0.8B](https://huggingface.co/PsiACE/Dohnuts-0.1.0-0.8B) (frozen
`Qwen/Qwen3.5-0.8B` backbone and vision encoder). No motion control: the model
stops at JSON. This repository holds the implementation only.

Weights licence: CC BY-NC-SA 4.0 (thesis use only). Dohnuts code: Apache-2.0.

## Layout

| Path | What |
|---|---|
| `actone/api.py` | Two-pass inference -> `ActOneDecision` |
| `actone/policy.py` | Caller-side policy (execute / ask_big / ask_human / abort / hold) |
| `actone/runtime.py` | Loads the merged model on CUDA (vendor path) or CPU |
| `scripts/hello_predict.py` | One-pass load test against bare Dohnuts |
| `scripts/collect_labels.py` | LIBERO keyframe labeler (proprioception only), episode splits |
| `train/to_dohnuts.py` | One event -> four Dohnuts JSONL rows |
| `scripts/launch_thesis.py` | Train + evaluate via `dohnuts.train` |
| `scripts/make_shuffled.py` | Shuffled-label ablation data |
| `scripts/ablation_infer.py` | Zero-shot Dohnuts and no-image inference |
| `eval/offline.py` | Calibration-only temperatures and thresholds, then one test pass |
| `eval/plots.py` | Reliability diagrams, risk-coverage curve |
| `scripts/demo.py` | Defence demo: 10 test images, ActOne vs zero-shot |
| `vendor/dohnuts` | Upstream Dohnuts, never edited (not committed; clone below) |

Not committed: `data/` contents, `runs/` contents, checkpoints, `from-gpu/`, `.env`.

## Kept result

`runs/actone_v4_s3000/seed-42/best.pt` (3,000 steps, v4 data, seed 42, dev macro
0.9375). Held-out test (1,364 events), calibration-fitted temperatures and
thresholds (`P_MIN_ACT` 0.80, `P_MIN_HUMAN` 0.20, `U_MAX` 0.10):

| Head | ActOne | Zero-shot Dohnuts | Constant baseline |
|---|---:|---:|---:|
| action | 0.852 | 0.015 | 0.308 |
| escalate | 0.912 | 0.818 | 0.831 |
| p_success | 0.946 | 0.801 | 0.880 |
| p_unsafe | 1.000 | 0.820 | 1.000 |

Mean regret 0.407 (zero-shot 1.461); illegal-action rate 0 by construction.
`p_unsafe` has no positive labels in LIBERO, and gold `escalate` is only `act` or
`ask_big`, so those two heads say less than their numbers suggest.

## Setup

Python 3.12. Clone Dohnuts at the pinned commit (do not edit it):

```bash
git clone https://github.com/PsiACE/dohnuts.git vendor/dohnuts
git -C vendor/dohnuts checkout e22b01cba0f83c0b4e8fe065f946d775c0336a4c
python3.12 -m venv .venv && source .venv/bin/activate && pip install -U pip
```

GPU (training, NVIDIA, CUDA 12.8). Do not `pdm install` the vendor's ROCm lock on NVIDIA:

```bash
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e vendor/dohnuts
```

CPU (demo and evaluation; flash-linear-attention needs Triton and does not run on
CPU, so transformers' PyTorch Qwen3.5 ops are used):

```bash
pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-deps -e vendor/dohnuts
```

Base models download from Hugging Face at pinned revisions on first use:
`PsiACE/Dohnuts-0.1.0-0.8B@959828a3d9cf3dacbd1491e11bb1cd36d12709ac` and
`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`.

## Reproduce training (GPU)

The training scripts are committed exactly as they ran on the GPU box, so
`launch_thesis.py`, `ablation_infer.py` and `make_shuffled.py` keep
`ROOT = /home/ubuntu/actone` and the model snapshot paths under
`/home/ubuntu/actone/.cache/hf`. Run from a checkout at that path with
`HF_HOME=/home/ubuntu/actone/.cache/hf`, or edit `ROOT` / `MODEL` / `INIT`.
Commands below are reconstructed from the run logs and manifests.

```bash
python scripts/hello_predict.py                         # gate: four typed answers
# lerobot/libero (1.9 GB) into data/raw/lerobot_libero, then label v4 (full episode pools)
python scripts/collect_labels.py --events data/jsonl/events_v4.jsonl \
  --report data/jsonl/label_report_v4.json \
  --train-episodes 981 --dev-episodes 200 --calibration-episodes 231 --test-episodes 281
python scripts/collect_labels.py --extract --events data/jsonl/events_v4.jsonl
python train/to_dohnuts.py --events data/jsonl/events_v4.jsonl --out data/processed/actone_v4
python scripts/launch_thesis.py --data data/processed/actone_v4 \
  --output runs/actone_v4_s3000 --steps 3000 --seed 42
```

Checkpoint selection is dev macro accuracy only; test is never used to pick a step,
seed or dataset.

## Reproduce evaluation

Temperatures and thresholds are fit on the calibration split only, then test is
scored once. This runs on CPU from the recorded predictions and reproduces
`heldout_policy.json` exactly:

```bash
R=runs/actone_v4_s3000; D=data/processed/actone_v4
python eval/offline.py \
  --calibration-predictions $R/seed-42/calibration-predictions.jsonl \
  --test-predictions $R/seed-42/test-predictions.jsonl \
  --calibration-labels $D/calibration.jsonl --test-labels $D/test.jsonl \
  --out $R/seed-42/heldout_policy.json
python eval/plots.py --report $R/seed-42/heldout_policy.json --out $R/plots
# GPU: zero-shot and no-image ablations, then offline.py on their predictions
python scripts/ablation_infer.py --which zero-shot --data $D --out $R/ablation
python scripts/ablation_infer.py --which no-image --data $D \
  --checkpoint $R/seed-42/best.pt --out $R/ablation
```

## Run the defence demo (CPU or GPU)

Ten held-out test images in, `ActOneDecision` JSON out, ActOne vs zero-shot
Dohnuts. Real two-pass inference with the frozen temperatures and thresholds; no
retraining. Events are chosen by a fixed label-only rule (two per verb, one `act`
and one `ask_big` where the test split has one, `random.Random(42)`).

```bash
# expects the GPU artifacts (runs/, data/processed/actone_v4, data/frames/libero)
# under from-gpu/gpu-actone/, or pass --artifacts <dir>
python scripts/demo.py --threads 4 --out runs/demo
```

Writes `decisions.json` (gold labels, both decisions, policy outcome, latency,
CPU-vs-GPU logit check), `demo.md` (table with thumbnails, summary, misses) and
`images/`. About 1.7 s per two-pass decision on a 4-thread Xeon CPU in BF16.
