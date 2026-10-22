"""One-pass load test. Proves Dohnuts loads. Production inference is two-pass."""

from pathlib import Path

from PIL import Image
from dohnuts.predictor import Predictor

# Sample legal[] from the input contract. Action criteria are filled from it.
LEGAL = ["reach", "grasp", "lift", "retract", "home", "abort"]

ACTONE_CORE = {
    "action": {
        "type": "choice",
        "instructions": (
            "You are the tactical layer of a robot. "
            "Pick exactly one legal action. Do not invent actions. "
            "If none is appropriate, pick abort."
        ),
        "criteria": {name: name for name in LEGAL},  # filled from legal[]
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


def main():
    root = Path(__file__).resolve().parents[1]
    predictor = Predictor.from_checkpoint("PsiACE/Dohnuts-0.1.0-0.8B")
    with Image.open(root / "desk.jpg") as image:
        state = {
            "image": image.convert("RGB"),
            "message": {
                "body": "arm_franka",
                "pack": "tabletop_v1",
                "phase": "grasp",
                "skill": "reach",
                "skill_age_s": 2.1,
                "battery": 0.72,
                "ee_z": 0.18,
                "target_in_view": True,
                "target_frac": 0.15,
                "stuck_s": 0.0,
                "last_outcome": "success",
                "legal": LEGAL,
            },
        }
        result = predictor.predict(state, ACTONE_CORE)
    print(result["answers"])


if __name__ == "__main__":
    main()
