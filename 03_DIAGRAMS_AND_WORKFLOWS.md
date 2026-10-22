# ActOne — System diagrams and workflows

All diagrams are mermaid. Render in GitHub / Cursor preview.  
POMDP belief tracking is not in any diagram on purpose.

---

## 1. System context

The thesis product stops at JSON. The caller is not the thesis.

```mermaid
flowchart LR
  subgraph Caller["Caller — not the thesis"]
    cam[RGB frame]
    filt[Filter illegal moves<br/>limits / workspace / energy]
    exec[Execute or ask or stop]
  end

  subgraph ActOne["ActOne — the thesis"]
    api[Two-pass wrapper]
    model[Dohnuts + LoRA]
    pol[Tiny policy]
  end

  cam --> api
  filt -->|legal list + JSON state| api
  api --> model
  model --> api
  api -->|ActOneDecision| pol
  pol -->|execute / ask_big / ask_human / abort / hold| exec
```

---

## 2. Two-pass inference (production)

This is the frozen inference contract. `hello_predict` is one-pass only as a load test.

```mermaid
flowchart TD
  I[RGB frame I] --> P1
  S[JSON state s + legal A] --> P1
  P1["Pass 1 — choice over A"]
  P1 --> ACT["â ∈ A ∪ {abort}<br/>p(a) for each a"]
  ACT --> RW["Rewrite noul / escalate<br/>instructions to NAME â"]
  I --> P2
  S --> P2
  RW --> P2
  P2["Pass 2 — p_success, p_unsafe, escalate"]
  P2 --> D[ActOneDecision]
  D --> POL["Policy<br/>P_MIN_ACT=0.70<br/>P_MIN_HUMAN=0.40<br/>U_MAX=0.30"]
  POL --> OUT[execute / ask_big / ask_human / abort / hold]
  OUT -.-> X[No torques. No IK. End.]
```

Why two passes: Dohnuts questions in one `predict()` call are independent. A noul head cannot read the action head. So “if the chosen action succeeds” is a lie unless the verb is written into pass 2.

---

## 3. What is inside the net (inherited)

Do not redraw this as a new architecture. You train two blocks only.

```mermaid
flowchart TB
  img[RGB frame] --> vis["Frozen Qwen3.5-0.8B vision encoder"]
  txt["state + instructions + candidate text"] --> tok[Tokenizer]
  vis --> prefix[Shared causal prefix]
  tok --> prefix
  prefix --> suf["Per-candidate suffixes<br/>+ reserved marker token"]
  suf --> scorer["Trainable candidate scorer"]
  prefix --> lora["Trainable rank-8 language LoRA α=16"]
  lora --> suf
  scorer --> sm["Softmax over supplied candidates only"]
  sm --> ans[choice / noul / score + probs]
```

| Block | Status |
|---|---|
| Qwen language backbone | Frozen |
| Vision encoder | Frozen |
| Rank-8 LoRA + scorer | Trainable |
| Packs, two-pass wrapper, temps, eval | Written by this thesis |

---

## 4. Data factory

LIBERO is continuous 7-DoF. ActOne needs discrete skills. This pipeline *is* month-1 work.

```mermaid
flowchart TD
  raw["lerobot/libero<br/>1.9 GB · 1693 episodes"] --> kf[Subsample keyframes]
  kf --> heur["collect_labels.py<br/>gripper / EE heuristics"]
  heur --> ev["One event:<br/>image + JSON + gold verb<br/>+ executed success/unsafe<br/>+ escalate gold"]
  ev --> conv["to_dohnuts.py"]
  conv --> r1[Row: action choice]
  conv --> r2[Row: p_success noul]
  conv --> r3[Row: p_unsafe noul]
  conv --> r4[Row: escalate choice]
  r1 --> split["Split by episode id"]
  r2 --> split
  r3 --> split
  r4 --> split
  split --> tr[train.jsonl 8–12k]
  split --> ca[calibration.jsonl 1–2k]
  split --> te[test.jsonl 1.5–2k]
```

Refuse classes (`ask_big` / `ask_human` / `abort`) ≥ 15% or the escalate head collapses to `act`.

---

## 5. Training workflow

```mermaid
flowchart LR
  ckpt["PsiACE/Dohnuts-0.1.0-0.8B"] --> init["--initialize-from<br/>new optimizer"]
  data["data/processed/actone"] --> train["run_experiment.py"]
  init --> train
  train --> over["Gate: 200 rows · 200 steps<br/>loss ↓ · action acc → 1"]
  over --> full["Thesis run · 1500–3000 steps<br/>rank-8 LoRA + scorer"]
  full --> out["runs/actone/checkpoint"]
```

Always a **new** `--output` directory. Do not online-update LoRA during eval.

---

## 6. Calibration then freeze

Test is not used here.

```mermaid
flowchart TD
  cal[calibration.jsonl] --> fit["Fit 4 temperatures<br/>T_action T_success T_unsafe T_escalate"]
  fit --> thr["Fit / confirm<br/>P_MIN_ACT P_MIN_HUMAN U_MAX"]
  thr --> lock[Freeze both]
  lock --> test[Only now touch test.jsonl]
```

---

## 7. Evaluation pipeline

```mermaid
flowchart TB
  testdata[Held-out episodes] --> zs[Zero-shot Dohnuts]
  testdata --> ao[ActOne LoRA]
  testdata --> ni[No-image ablation]
  testdata --> sh[Shuffled-label sanity]
  zs --> tab[Comparison table]
  ao --> tab
  ni --> tab
  sh --> tab
  ao --> ece[ECE + Brier raw and scaled]
  ao --> rel[Reliability diagrams]
  ao --> rc[Risk-coverage curve]
  ao --> cost[Frozen-cost regret]
  ao --> ill[Illegal-action rate]
  tab --> thesis[Thesis numbers]
  ece --> thesis
  rel --> thesis
  rc --> thesis
  cost --> thesis
  ill --> thesis
```

Defence demo = 10 images in, JSON out.

---

## 8. Policy if-chain

```mermaid
flowchart TD
  d[ActOneDecision] --> a{action in legal?}
  a -->|no| abort[abort]
  a -->|yes| u{p_unsafe > U_MAX?}
  u -->|yes| abort
  u -->|no| e1{escalate is abort<br/>or action is abort?}
  e1 -->|yes| abort
  e1 -->|no| e2{escalate is ask_human<br/>or p_success < P_MIN_HUMAN?}
  e2 -->|yes| human[ask_human]
  e2 -->|no| e3{escalate is ask_big?}
  e3 -->|yes| big[ask_big]
  e3 -->|no| e4{escalate is act<br/>and p_success ≥ P_MIN_ACT?}
  e4 -->|yes| go[execute]
  e4 -->|no| hold[hold]
```

---

## 9. Weekend-to-defence swimlane

```mermaid
flowchart LR
  w1[Weekend 1<br/>install + hello_predict<br/>+ wrappers]
  w2[Week 2<br/>LIBERO + 200-row overfit<br/>+ 20-frame table]
  m1[Weeks 3–4<br/>≥2k rows]
  m2[Month 2<br/>LoRA + first reliability]
  m3[Month 3<br/>8–12k + ablations]
  m4[Month 4<br/>freeze numbers + write]
  w1 --> w2 --> m1 --> m2 --> m3 --> m4
```

---

## 10. What Cursor may touch

```mermaid
flowchart TB
  ok["Cursor MAY write"]
  no["Cursor must NOT write"]
  ok --> a[actone/*.py]
  ok --> s[scripts/*.py]
  ok --> t[train/*.py]
  ok --> e[eval/*.py]
  ok --> d[docs/*]
  no --> v[vendor/dohnuts/**]
  no --> ctrl[IK / PD / 100 Hz loop]
  no --> wm[World model / belief SE]
  no --> peft[PeftModel wrap of Qwen]
  no --> datafirst[Dataset download before hello_predict]
```

---

## 11. End-to-end sequence (one picture)

```mermaid
sequenceDiagram
  participant You
  participant Cursor
  participant Dohnuts as vendor/dohnuts
  participant HF as Hugging Face
  participant Train as run_experiment.py

  You->>Cursor: drop docs/, ask for skeleton
  Cursor->>Dohnuts: git clone (do not edit)
  You->>Cursor: write hello_predict.py
  Cursor->>HF: Predictor.from_checkpoint
  HF-->>You: four typed answers
  Note over You: GATE passed. Now data.
  You->>HF: snapshot_download lerobot/libero
  You->>Cursor: collect_labels + to_dohnuts
  Cursor->>Train: 200-step overfit
  Train-->>You: loss drops
  Cursor->>Train: 1500-step thesis run
  You->>Cursor: calibrate.py on calib only
  You->>Cursor: eval table + plots
  You->>You: write thesis
```
