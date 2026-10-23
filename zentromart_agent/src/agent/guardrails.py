import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import TODAY, TOOLS_DIR, DATASET_DIR, HIGH_VALUE_THRESHOLD
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    from tools import create_ticket  # type: ignore[import-untyped, import-not-found]
except (ImportError, ModuleNotFoundError):
    try:
        from Capstone_Project_Dataset_FTTT_2026.tools.tools import create_ticket  # type: ignore[import-untyped, import-not-found]
    except (ImportError, ModuleNotFoundError):
        from src.agent.tools_wrapper import create_ticket

from src.data_layer.business_days import add_business_days, refund_max_deadline

# Deterministic escalation triggers
LEGAL_KEYWORDS = re.compile(
    r"\b(legal|lawyer|court|consumer\s+court|consumer\s+forum|police|fir|police\s+complaint)\b",
    re.IGNORECASE
)

SAFETY_KEYWORDS = re.compile(
    r"\b(swelling|swollen|burning|smoke|spark|sparks|sparking|fire|electric\s+shock|shock|blast|explosion|injury|injured|burns)\b",
    re.IGNORECASE
)

HUMAN_AGENT_KEYWORDS = re.compile(
    r"\b(talk\s+to\s+(a\s+)?human|speak\s+to\s+(a\s+)?(human|person|agent|representative|manager)|connect\s+(me\s+)?to\s+(a\s+)?(human|person|agent|representative|executive)|customer\s+care\s+executive|real\s+person)\b",
    re.IGNORECASE
)

NEW_ORDER_KEYWORDS = re.compile(
    r"\b(another\s+order|different\s+order|new\s+order|dusra\s+order|check\s+another|other\s+order)\b",
    re.IGNORECASE
)


def check_escalation_triggers(session, message: str) -> Optional[Tuple[str, str, str]]:
    """
    Checks deterministic escalation triggers.
    Returns (trigger_name, reason, description) if triggered, else None.
    """
    msg_clean = message.lower()
    order = session.order_data if hasattr(session, "order_data") else None

    # Trigger 2: Legal threat
    if LEGAL_KEYWORDS.search(msg_clean):
        return ("legal_threat", "legal_threat", "Customer mentioned potential legal action or consumer forum")

    # Trigger 3: Safety risk
    if SAFETY_KEYWORDS.search(msg_clean):
        return ("safety_risk", "safety_risk", "Physical safety hazard reported (burn, smoke, spark, injury)")

    # Trigger 6: Explicit human request
    if HUMAN_AGENT_KEYWORDS.search(msg_clean):
        return ("human_requested", "human_requested", "Customer explicitly requested a human specialist")

    if order:
        # Trigger 4: Repeat contact (contacts_last_7_days >= 2)
        try:
            contacts = int(order.get("contacts_last_7_days", 0))
            if contacts >= 2:
                return ("repeat_contact", "repeat_contact", "Customer contacted support >= 2 times in the last 7 days")
        except (ValueError, TypeError):
            pass

        # Trigger 1: High-value damage (> HIGH_VALUE_THRESHOLD INR, default ₹5,000)
        try:
            tot = float(order.get("total_inr", 0))
            has_damage = bool(
                order.get("damage_issue")
                or re.search(r"\b(damage|damaged|defective|broken|cracked|shattered|faulty)\b", msg_clean)
            )
            if has_damage and tot > HIGH_VALUE_THRESHOLD:
                return (
                    "high_value_damage",
                    "high_value_damage",
                    f"High-value item exceeding {HIGH_VALUE_THRESHOLD:,} INR reported damaged or defective"
                )
        except (ValueError, TypeError):
            pass

        # Trigger 5: Refund overdue (initiated_date + max_timeline_bd + 3 bd < TODAY)
        for prefix in ["return_refund", "cancellation_refund", "rto_refund"]:
            init_date_str = order.get(f"{prefix}_initiated_on")
            method = order.get(f"{prefix}_method") or order.get("payment_method") or "upi"
            comp_date_str = order.get(f"{prefix}_completed_on")
            if init_date_str and not comp_date_str:
                try:
                    init_d = date.fromisoformat(init_date_str)
                    deadline = refund_max_deadline(init_d, method)
                    overdue_threshold = add_business_days(deadline, 3)
                    if overdue_threshold < TODAY:
                        return ("refund_overdue", "refund_overdue", "Refund delayed beyond standard bank timelines + 3 business days")
                except Exception:
                    pass

    return None


def execute_escalation_if_needed(session, message: str) -> None:
    """
    Evaluates guardrails deterministically.
    If triggered, creates a ticket and updates session state so the LLM communicates
    the resolution and ticket reference to the customer in their native language.
    """
    # Context switch: if user asks about another order, do not block with previous escalation
    norm_digits = "".join(ch for ch in message if ch.isdigit())
    oid_match = re.search(r"\b(\d{8})\b", message) or re.search(r"(\d{8})", norm_digits)
    extracted_oid = oid_match.group(1) if oid_match else None
    if NEW_ORDER_KEYWORDS.search(message) or (extracted_oid and session.order_id and extracted_oid != session.order_id):
        return None

    # If already escalated in this session, let the LLM handle subsequent turns with the active ticket context
    if getattr(session, "escalated", False) and getattr(session, "ticket_id", None):
        return None

    # First-time escalation evaluation
    trigger_info = check_escalation_triggers(session, message)
    if not trigger_info:
        return None

    trigger_name, reason, description = trigger_info
    oid = session.order_id or ""

    summary = (
        f"Customer issue raised: {message[:120]}. "
        f"Order ID: {oid or 'None'}. "
        f"Escalation trigger fired: {trigger_name}. "
        f"Session language: {session.detected_lang}."
    )

    ticket_res = create_ticket(
        summary=summary,
        order_id=oid,
        reason=reason,
        intent="escalation"
    )

    if ticket_res.get("created"):
        session.state = "ESCALATED"
        session.escalated = True
        session.ticket_id = ticket_res.get("ticket_id")
        session.escalation_reason = description
        session.tool_calls_made.append(f"create_ticket(reason={reason})")
    else:
        session.state = "ESCALATED"
        session.escalated = True
        session.escalation_reason = description

    return None
