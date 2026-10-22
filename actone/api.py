"""Two-pass ActOne inference: f(image, state, legal) -> ActOneDecision.

Pass 1 asks `action` only, over the caller-filtered legal list. Pass 2 names the
pass-1 action in the p_success / p_unsafe / escalate instructions (and in the
state's phase/skill, as the training rows name the gold verb there), then asks
those three questions. Dohnuts questions inside one predict() cannot read each
other, so a one-pass conditional p_success is not possible.

Question text must stay byte-identical to train/to_dohnuts.py.
"""

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import torch

from dohnuts.predictor import Predictor

ESCALATE = ["act", "ask_big", "ask_human", "abort"]
HEADS = ["action", "p_success", "p_unsafe", "escalate"]


@dataclass
class ActOneDecision:
    action: str
    action_probs: dict
    p_success: float
    p_unsafe: float
    escalate: Literal["act", "ask_big", "ask_human", "abort"]
    escalate_probs: dict
    raw: dict = field(default_factory=dict)
    source: str = "actone"

    def to_dict(self):
        return asdict(self)


def action_question(legal):
    return {
        "type": "choice",
        "instructions": (
            "You are the tactical layer of a robot. "
            "Pick exactly one legal action. Do not invent actions. "
            "If none is appropriate, pick abort."
        ),
        "criteria": {name: name for name in legal},
    }


def scored_questions(verb):
    return {
        "p_success": {
            "type": "noul",
            "instructions": (
                f"If {verb} is executed now, will it succeed within the skill "
                "timeout without collision or leaving the workspace?"
            ),
        },
        "p_unsafe": {
            "type": "noul",
            "instructions": (
                f"Would executing {verb} now risk collision, limit violation, "
                "or energy exhaustion?"
            ),
        },
        "escalate": {
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
    }


def scored_state(state, verb):
    return {**state, "phase": verb, "skill": verb}


def softmax(logits, temperature):
    scaled = [value / temperature for value in logits]
    top = max(scaled)
    weights = [math.exp(value - top) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


class ActOne:
    """temperatures: per-head {action, p_success, p_unsafe, escalate}, fitted on calibration."""

    def __init__(self, model, temperatures, *, source="actone"):
        self.model = model
        self.predictor = Predictor(model)
        self.temperatures = temperatures
        self.source = source
        self.device = getattr(model, "device", "cuda")

    @torch.inference_mode()
    def logits(self, state, image, questions):
        """Raw scorer logits per question, via the vendor's serving prompt path."""
        self.model.eval()
        payload = {**state, "image": image} if image is not None else dict(state)
        inputs, positions, mask, metadata = self.predictor.prepare(payload, questions)
        inputs = {
            key: (
                value.to(self.device, dtype=torch.bfloat16)
                if key == "pixel_values"
                else value.to(self.device)
            )
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }
        output = self.model(inputs, positions.to(self.device)).float().cpu()
        result = {}
        for row, (qid, _, labels) in enumerate(metadata):
            result[qid] = {"labels": labels, "logits": output[row, : len(labels)].tolist()}
        return result

    def decide(self, image, state, legal=None):
        legal = list(legal or state["legal"])
        state = {**state, "legal": legal}
        started = time.perf_counter()
        first = self.logits(state, image, {"action": action_question(legal)})["action"]
        pass1_s = time.perf_counter() - started
        action_p = softmax(first["logits"], self.temperatures["action"])
        action = first["labels"][max(range(len(action_p)), key=action_p.__getitem__)]
        second_started = time.perf_counter()
        second = self.logits(scored_state(state, action), image, scored_questions(action))
        pass2_s = time.perf_counter() - second_started
        success_p = softmax(second["p_success"]["logits"], self.temperatures["p_success"])
        unsafe_p = softmax(second["p_unsafe"]["logits"], self.temperatures["p_unsafe"])
        escalate_p = softmax(second["escalate"]["logits"], self.temperatures["escalate"])
        escalate = ESCALATE[max(range(len(escalate_p)), key=escalate_p.__getitem__)]
        return ActOneDecision(
            action=action,
            action_probs=dict(zip(first["labels"], action_p, strict=True)),
            p_success=success_p[1],
            p_unsafe=unsafe_p[1],
            escalate=escalate,
            escalate_probs=dict(zip(ESCALATE, escalate_p, strict=True)),
            raw={
                "logits": {
                    "action": first["logits"],
                    **{name: second[name]["logits"] for name in ("p_success", "p_unsafe", "escalate")},
                },
                "temperatures": dict(self.temperatures),
                "latency_s": {"pass1": pass1_s, "pass2": pass2_s, "total": pass1_s + pass2_s},
            },
            source=self.source,
        )
