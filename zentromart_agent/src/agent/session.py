from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class ConversationSession:
    """
    In-memory conversation session tracking verification and dialog state.
    Never persisted to disk to protect customer data.
    """
    session_id: str
    channel: str = "text"                      # "text" or "voice"
    detected_lang: str = "en"
    state: str = "GREETING"                   # GREETING, AWAITING_ORDER_ID, AWAITING_PROOF, AWAITING_RETRY, VERIFIED, ESCALATED
    order_id: Optional[str] = None            # 8-digit string, no spaces
    order_data: Optional[Dict[str, Any]] = None  # populated ONLY after verified=True
    verification_attempts: int = 0            # increments on each proof submission
    history: List[Dict[str, Any]] = field(default_factory=list)  # {"role": "user"|"model"|"assistant", "content": str}
    section_ids_used: List[str] = field(default_factory=list)
    tool_calls_made: List[str] = field(default_factory=list)
    escalated: bool = False
    ticket_id: Optional[str] = None
    escalation_reason: Optional[str] = None
    just_verified: bool = False
