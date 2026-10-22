# ActOne — Project context

**Read this before writing code.** If a later chat conflicts with the frozen list, this file wins unless Ashish explicitly reopens a numbered item.

| | |
|---|---|
| Name | ActOne |
| Student | Ashish Bhardwaj · BITS Pilani Dubai · 2023A7PS0324U |
| Course | BITS F421T (16 credit) · AY 2026-27 Semester I · On campus |
| Supervisor | Dr. Mithun Mukherjee |
| Title | ActOne: Fine-Tuning a Multimodal System-One Model for Action-Conditioned Robot Decisions and Calibrated Escalation |
| Status | Design frozen. Implementation not started. |
| Date frozen | 25 September 2026 |

---

## 1. One paragraph

ActOne is a small multimodal **System One** decision model. You give it one camera frame, a short JSON state, and a closed list of **legal actions**. In two forward passes it returns typed answers, not prose: which action, P(that action succeeds), P(it is unsafe), and whether to act, ask a larger model, ask a human, or abort. It is **LoRA on Dohnuts-0.1.0-0.8B**, not a new architecture. Code around it may apply thresholds. The model never sends actuator commands.

Slogan: *Jev decided from text. Dohnuts decided from a picture. ActOne decides whether the robot should move.*

Thesis sentence:

> Prior System One models decide over text or generic images. ActOne fine-tunes a small System One model to score a caller-supplied legal action set for a robot and to refuse, with probabilities checked against executed outcomes.

---

## 2. Is / is not

**Is:** a function `f(image, state_json, legal_actions, pack) → ActOneDecision`. Fine-tuned Dohnuts. Advisory. Reused across bodies by swapping the pack and `legal[]`, not by new weights. Calibrated on **executed outcomes**.

**Is not:** a motion controller, IK solver, detector, planner that invents skills, world model, VLA that emits continuous controls, chatbot, product, POMDP / belief-state tracker, or the first visual System One (that is Dohnuts / OmniJev / Laya Vision).

POMDP belief (Kaelbling, Littman & Cassandra 1998) is **related work only**. Cite it as the classical “act vs sense vs abort” statement. Do not implement `SE(b,a,o)` or a belief MDP.

---

## 3. Frozen decisions

1. Name: ActOne.
2. Method: fine-tune Dohnuts-0.1.0-0.8B. No from-scratch model.
3. Outputs only: `action`, `p_success`, `p_unsafe`, `escalate`.
4. Interface: Dohnuts `choice` / `noul` / `score`.
5. `legal[]` is filtered **before** the model. Never invent a verb.
6. **Two-pass inference.** Pass 1 = action. Pass 2 = noul + escalate with the chosen (or gold) verb named in the instructions. One-pass conditional `p_success` is false: Dohnuts questions in one `predict()` call cannot read each other.
7. Calibration on executed outcomes is a first-class result.
8. Reuse across robots = packs + `legal[]`, not new weights per body.
9. No motion-control stack in the deliverable.
10. No world model. No POMDP belief updater.
11. One consumer 16–24 GB GPU.
12. Demo: labelled images in, JSON decisions out, tables vs zero-shot Dohnuts.
13. `vendor/dohnuts` stays untouched. Wrap it. Do not PEFT-wrap Qwen.
14. Dohnuts weights: **CC BY-NC-SA 4.0**. Thesis OK. Product not OK from this checkpoint.
15. Temperatures and policy thresholds fitted on the **calibration** split only.
16. Splits by **episode / group**, never by frames of the same episode.
17. No large dataset until `scripts/hello_predict.py` prints four typed answers.

---

## 4. Related work (cite, do not clone)

| System | Note |
|---|---|
| Jev (TypeSafe, 15 Sep 2026) | First System One. Text only. Hosted. No public weights. |
| Dohnuts (PsiACE, ~21 Sep 2026) | 0.8B visual System One. **This thesis fine-tunes it.** |
| OmniJev | LIBERO-10 frame decisions 77.1% (0.8B) / 80.7% (4B), ECE 0.031. Closest neighbour. No 4-way escalate + outcome-calibrated p_success over a caller legal set + pack transfer + frozen-cost regret. |
| robojev, PlayJev, jev-libero, Laya Vision | Concurrent. Cite. |
| RT-2, OpenVLA, π0 | VLAs. Different contract. Out of scope. |
| Kaelbling et al. 1998 POMDP | Classical “act vs sense.” Related work only. |
| Selective classification (Chow; Geifman & El-Yaniv 2017) | The right *family* for escalate. |
| Temperature scaling (Guo et al. 2017) | How `p_success` becomes a number you can defend. |

---

## 5. Base model

| | |
|---|---|
| Repo | https://github.com/PsiACE/dohnuts |
| Checkpoint | https://huggingface.co/PsiACE/Dohnuts-0.1.0-0.8B |
| Base | `Qwen/Qwen3.5-0.8B` rev `2fc06364715b967f1860aea9cf38778875588b17` |
| Load | `Predictor.from_checkpoint("PsiACE/Dohnuts-0.1.0-0.8B")` only |
| Trainable | rank-8 language LoRA (α=16) + candidate scorer |
| Frozen | Qwen language backbone + vision encoder |
| Code licence | Apache-2.0 |
| Weight licence | CC BY-NC-SA 4.0 |

Checkpoint is **not** a standard PEFT adapter. Do not wrap Qwen with `PeftModel`.

---

## 6. Input contract

One RGB image (PIL, `.convert("RGB")`). Compact JSON in `state["message"]`:

```json
{
  "body": "arm_franka",
  "pack": "tabletop_v1",
  "phase": "grasp",
  "skill": "reach",
  "skill_age_s": 2.1,
  "battery": 0.72,
  "ee_z": 0.18,
  "target_in_view": true,
  "target_frac": 0.15,
  "stuck_s": 0.0,
  "last_outcome": "success",
  "legal": ["reach", "grasp", "lift", "retract", "home", "abort"]
}
```

`legal` is already filtered by caller code. No raw actuator packets. LIBERO frames use `"body": "arm_franka"`.

---

## 7. Core questions

```python
ACTONE_CORE = {
    "action": {
        "type": "choice",
        "instructions": (
            "You are the tactical layer of a robot. "
            "Pick exactly one legal action. Do not invent actions. "
            "If none is appropriate, pick abort."
        ),
        "criteria": {},  # filled from legal[]
    },
    "p_success": {
        "type": "noul",
        "instructions": (
            "If the chosen action is executed now, will it succeed "
            "within the skill timeout without collision or leaving the workspace?"
        ),
    },
    "p_unsafe": {
        "type": "noul",
        "instructions": (
            "Would executing the chosen action now risk collision, "
            "limit violation, or energy exhaustion?"
        ),
    },
    "escalate": {
        "type": "choice",
        "instructions": (
            "act = execute the chosen legal action; "
            "ask_big = call a larger vision model on this frame; "
            "ask_human = stop and request a human; "
            "abort = safe stop."
        ),
        "criteria": {
            "act": "Execute the chosen legal action now.",
            "ask_big": "Ambiguous scene; ask a larger model.",
            "ask_human": "Risk or ambiguity too high; ask a person.",
            "abort": "Do not act.",
        },
    },
}
```

At **inference**, after pass 1 returns `â`, rewrite noul instructions to name `â`.  
At **training**, noul / escalate rows name the **gold** verb.

Packs (same weights):

| Pack | Verbs |
|---|---|
| `tabletop_v1` (primary) | reach, grasp, lift, place, retract, home, abort |
| `inspect_v1` (transfer test) | search, approach, cover, photo, next, hold, abort |
| `mobile_v1` | drop first if late |

---

## 8. Output and policy

```python
@dataclass
class ActOneDecision:
    action: str
    action_probs: dict
    p_success: float
    p_unsafe: float
    escalate: Literal["act", "ask_big", "ask_human", "abort"]
    escalate_probs: dict
    raw: dict
    source: str   # "actone" | "default"
```

Thresholds, fitted on **calibration**, then frozen:

```
P_MIN_ACT = 0.70
P_MIN_HUMAN = 0.40
U_MAX = 0.30
```

```
if action not in legal: abort
elif p_unsafe > U_MAX: abort
elif escalate == "abort" or action == "abort": abort
elif escalate == "ask_human" or p_success < P_MIN_HUMAN: ask_human
elif escalate == "ask_big": ask_big
elif escalate == "act" and p_success >= P_MIN_ACT: execute
else: hold
```

This policy is ~20 lines. It is not the thesis product. It must not send torques.

---

## 9. Data

One labelled moment → **four JSONL rows** (Dohnuts trains one question per row).

`p_success` / `p_unsafe` targets = **what happened after the skill ran**, not “the heuristic said grasp.”

Escalate gold is a published rule: near-tie / occlusion → `ask_big`; warning energy + unfamiliar → `ask_human`; collision imminent → `abort`; else `act`. Oversample refuse classes to **≥15%**.

LIBERO is continuous 7-DoF. `scripts/collect_labels.py` maps gripper / EE motion onto discrete verbs. Keyframes only.

| Split | Rows | Role |
|---|---|---|
| Train | 8k–12k | LoRA |
| Calibration | 1k–2k | temperatures + thresholds |
| Test | 1.5k–2k | report. Never used to pick a checkpoint. |

Month-1 source: https://huggingface.co/datasets/lerobot/libero (1.9 GB). Not Open X-Embodiment. Not full 40 GB HDF5 on day 1.

---

## 10. Training

Frozen vision + frozen base. Rank-8 LoRA + scorer.  
LoRA LR `1e-4`, head LR `5e-4`, batch 8 × accum 4.  
Overfit 200 rows first. Then `--steps 1500` (up to 3000).  
`--initialize-from PsiACE/Dohnuts-0.1.0-0.8B` into a **new** `--output` dir.  
Four temperatures on calib only.

Path B (NVIDIA): venv + CUDA torch. Install `flash-linear-attention==0.5.2` **without** `[rocm]`. Do not install the official ROCm `pdm.lock`.

---

## 11. Evaluation (this is the defence)

Always: **zero-shot Dohnuts vs ActOne LoRA** on held-out **episodes**.

Required: action acc, escalate acc, illegal-action rate, ECE + Brier on p_success and p_unsafe (raw and scaled), reliability diagrams, risk-coverage curve, frozen-cost regret, latency.

Frozen costs: correct act 0, extra ask_big 1, extra ask_human 3, failed act 5, unsafe act 25, correct abort 0.5.

Ablations: no image, no escalate head, shuffled labels. Optional pack-transfer.

Never drop: ECE, illegal-action rate, zero-shot Dohnuts baseline.

---

## 12. Repo tree

```
actone/
  docs/          ← these files
  actone/        packs.py api.py policy.py calibrate.py
  scripts/       hello_predict.py collect_labels.py
  train/         to_dohnuts.py
  eval/          offline.py plots.py
  data/          frames/ jsonl/ raw/ processed/actone/
  runs/          actone_overfit/ actone/
  vendor/dohnuts/   clone; do not edit
```

---

## 13. Cursor rules

- One file per change.
- Never edit `vendor/dohnuts`.
- Never add IK, PD, 100 Hz loop, world model, belief updater, VLA controls.
- Never wrap Qwen with vanilla PEFT.
- Never fit temps or thresholds on test.
- Never split by isolated frames of one episode.
- After each change, say which frozen item it implements.
