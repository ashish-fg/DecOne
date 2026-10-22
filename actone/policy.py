"""Caller-side policy over an ActOneDecision. Same rule as eval/offline.py decision_of.

Thresholds are fitted on the calibration split only and then frozen
(heldout_policy.json). Outcomes: execute, ask_big, ask_human, abort, hold.
"""

P_MIN_ACT = 0.70
P_MIN_HUMAN = 0.40
U_MAX = 0.30


def decide(action, p_success, p_unsafe, escalate, legal, *, p_min_act=P_MIN_ACT, p_min_human=P_MIN_HUMAN, u_max=U_MAX):
    if action not in legal:
        return "abort"
    if p_unsafe > u_max:
        return "abort"
    if escalate == "abort" or action == "abort":
        return "abort"
    if escalate == "ask_human" or p_success < p_min_human:
        return "ask_human"
    if escalate == "ask_big":
        return "ask_big"
    if escalate == "act" and p_success >= p_min_act:
        return "execute"
    return "hold"


def from_thresholds(decision, legal, thresholds):
    return decide(
        decision.action,
        decision.p_success,
        decision.p_unsafe,
        decision.escalate,
        legal,
        p_min_act=thresholds["P_MIN_ACT"],
        p_min_human=thresholds["P_MIN_HUMAN"],
        u_max=thresholds["U_MAX"],
    )
