# ActOne — End to end: what you do

Operator checklist. No theory. Do the steps in order. Stop at the first fail.

POMDP / belief tracking is not a step.

---

## 0. Before you open Cursor

Confirm on paper:

- GPU: NVIDIA 16–24 GB, AMD 24 GB, or “I will rent a 4090”
- OS: Ubuntu 22.04/24.04 (WSL2 works, more pain)
- Python **3.12** can be installed
- ~50 GB free disk (15 GB is enough for step 3)

If none of those exist, rent a RunPod / Vast RTX 4090 (~$0.14–0.34/hr) only when you train. Do not leave a pod on overnight empty.

Drop these docs into the repo:

```
docs/README.md
docs/01_PROJECT_CONTEXT.md
docs/02_END_TO_END.md
docs/03_DIAGRAMS_AND_WORKFLOWS.md
```

Cursor first prompt:

> Read `docs/README.md` then `docs/01_PROJECT_CONTEXT.md`, `docs/02_END_TO_END.md`, `docs/03_DIAGRAMS_AND_WORKFLOWS.md`. Follow frozen decisions. Create the repo skeleton. Write only `scripts/hello_predict.py`. Do not edit `vendor/dohnuts`. Do not download a dataset.

---

## 1. Create the repo skeleton

```
actone/
  docs/                 ← these four files
  actone/               empty package
  scripts/
  train/
  eval/
  data/frames data/jsonl data/raw data/processed/actone
  runs/
  vendor/
```

```bash
mkdir -p ~/actone/{docs,actone,scripts,train,eval,data/{frames,jsonl,raw,processed/actone},runs,vendor}
cd ~/actone
git clone https://github.com/PsiACE/dohnuts.git vendor/dohnuts
```

Do not edit anything under `vendor/dohnuts`.

---

## 2. Install (pick one)

### Path B — NVIDIA (most people)

Do **not** `pdm install` the official lock. It pins ROCm wheels.

```bash
cd ~/actone
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=5.17,<5.18" "peft>=0.21,<0.22"
pip install Pillow huggingface_hub
pip install "flash-linear-attention==0.5.2"    # NOT the [rocm] extra
pip install -e vendor/dohnuts
pip install pyarrow scikit-learn pandas datasets
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If FLA CUDA kernels fail to compile, stop and fix that. Do not rewrite Dohnuts kernels.

### Path A — AMD + official ROCm

Only if you already have a ROCm 6.4 card (7900 XTX class).

```bash
cd ~/actone/vendor/dohnuts
pdm use 3.12
pdm install --check --prod
# later:
pdm install --check --prod -G train -G agent
```

---

## 3. Gate — `hello_predict` (do not skip)

Put any RGB photo at `~/actone/desk.jpg`.

Write `scripts/hello_predict.py` exactly as specified in `01_PROJECT_CONTEXT.md` §7 questions + the skeleton in the original Cursor pack: one `Predictor.from_checkpoint("PsiACE/Dohnuts-0.1.0-0.8B")` call, four questions, print `result["answers"]`.

```bash
cd ~/actone
source .venv/bin/activate
python scripts/hello_predict.py
```

**Pass:** four typed answers print (`action`, `p_success`, `p_unsafe`, `escalate`). First run 20–40 min (download ~2–8 GB). Later runs seconds. Quality may be mediocre. That is fine.

**Fail:** stop. Typical causes: Python ≠ 3.12, ROCm lock on NVIDIA, FLA build, disk full in `~/.cache/huggingface`.

This script is **one-pass on purpose**. It only proves Dohnuts loads. Production inference is two-pass in `actone/api.py`.

**Do not download LIBERO until this prints.**

---

## 4. Wrappers (still no dataset)

One file per change:

| File | Job |
|---|---|
| `actone/packs.py` | `TABLETOP_V1`, `INSPECT_V1`, `ACTONE_CORE`, fill criteria from `legal[]` |
| `actone/api.py` | Two-pass wrapper → `ActOneDecision` |
| `actone/policy.py` | `P_MIN_ACT=0.70`, `P_MIN_HUMAN=0.40`, `U_MAX=0.30` and the if-chain |
| `scripts/hello_actone.py` | Same desk photo, now through `api.py` + `policy.py` |

Two-pass: pass 1 asks `action` only. Pass 2 rewrites noul / escalate instructions to name the chosen verb, then asks those three.

---

## 5. Data factory

Only now download the small mix:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("lerobot/libero", repo_type="dataset",
                  local_dir="data/raw/lerobot_libero")
PY
```

1.9 GB. Franka, 2 cameras 256², 1,693 episodes.

Then write:

| File | Job |
|---|---|
| `scripts/collect_labels.py` | Keyframe + gripper/EE heuristics → discrete skill + executed outcome + escalate gold |
| `train/to_dohnuts.py` | One event → four JSONL rows |

Mapping:

| Signal | Label |
|---|---|
| Gripper open, EE toward object | reach |
| Gripper closing near object | grasp |
| Object height rising after close | lift |
| EE to receptacle with object held | place |
| EE away after release | retract |
| Near start / reset | home |
| Collision / workspace / timeout | abort |

`p_success` = skill finished without collision / workspace exit / timeout.  
Refuse rows ≥15%. Split by **episode id**.

This week’s targets only:

- 20 frames → zero-shot Dohnuts table (`eval/offline.py` stub)
- 200 rows (~50 images × 4) → overfit
- not 8,000 yet

---

## 6. Prove the trainer

```bash
python vendor/dohnuts/scripts/run_experiment.py \
  --initialize-from PsiACE/Dohnuts-0.1.0-0.8B \
  --output runs/actone_overfit \
  --data data/processed/actone \
  --steps 200
```

**Pass:** loss drops; action accuracy on those 200 rows approaches 1.0.  
**Fail:** fix JSONL schema or the CUDA stack. Do not scale up.

---

## 7. Grow data, then the thesis run

Grow to ≥2,000 rows (month-1 exit), then 8,000–12,000 train + 1,000–2,000 calib + 1,500–2,000 test.

```bash
python vendor/dohnuts/scripts/run_experiment.py \
  --initialize-from PsiACE/Dohnuts-0.1.0-0.8B \
  --output runs/actone \
  --data data/processed/actone \
  --steps 1500
```

Always a **new** `--output` directory.

Then, on the **calibration** split only:

1. `actone/calibrate.py` — four temperatures (action, p_success, p_unsafe, escalate)
2. Freeze policy thresholds
3. Never look at test while doing 1 or 2

Optional: 20–30% generic typed-decision rows so instruction-following does not die. Rank-16 only if rank-8 underfits.

---

## 8. Evaluate and write

`eval/offline.py` + `eval/plots.py` must produce:

- zero-shot Dohnuts vs ActOne LoRA vs no-image vs shuffled-label
- action acc, escalate acc, illegal-action rate
- ECE + Brier, raw and temperature-scaled
- reliability diagrams
- risk-coverage curve
- frozen-cost regret
- latency

Defence demo: 10 labelled images in, JSON `ActOneDecision` out. Not a moving arm.

Write-up months 3–4. Cite OmniJev / Dohnuts / Jev / Kaelbling-as-related-work in week 1, not month 4.

---

## 9. Hours (one 24 GB card)

| Job | Time |
|---|---|
| hello_predict first download | 20–40 min |
| Overfit 200 | 15–40 min |
| ~2k rows | 1–3 h |
| Thesis 8–12k / 1500–3000 updates | 6–18 h |
| Temps + eval + plots | 1–3 h |
| Whole thesis GPU budget | 3–5 GPU-days · ~$15–80 if you rent only when training |

---

## 10. Calendar

| When | Done when |
|---|---|
| Weekend 1 | hello_predict prints. Wrappers exist. |
| Week 2 | lerobot/libero on disk. 200-row overfit works. 20-frame table written. |
| Weeks 3–4 | ≥2k rows, episode splits, ≥15% refuse. |
| Month 2 | Full LoRA. First reliability diagram. |
| Month 3 | 8–12k + ablations. Optional inspect pack zero-shot. |
| Month 4 | Freeze numbers. Write. |

Hard cuts if late: drop `mobile_v1`, extra optional questions, Path A if ROCm eats a week.  
Never drop: ECE, illegal-action rate, zero-shot Dohnuts baseline.

---

## 11. Do not do

- Download data before hello_predict prints
- Edit `vendor/dohnuts`
- Wrap Qwen with `PeftModel`
- Build IK / a 100 Hz loop / a world model / a belief updater
- Label every LIBERO frame
- Fit temperatures on test
- Treat Dohnuts `confidence` as P(correct)
- Claim one-pass `p_success` is about the chosen action
- Productise CC BY-NC-SA weights
