import functools
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Rule-based fallback translations for document vocabulary
FALLBACK_REPLACEMENTS = [
    (r"\b(how\s+many\s+days|how\s+long|kitne\s+din|कितने\s+दिन|lautane|लौटाने|wapas|वापस|return\s+window|किती\s+दिवस|परत|परतावा|मागे\s+द्यायचे)\b", "return window category RET-2"),
    (r"\b(opened|khol\s+liya|khol|खोल|try\s+kiya|tried|khol\s+diya|उघडले|वापरले)\b", "condition of returned items seal intact unopened RET-4"),
    (r"\b(doesn'?t\s+fit|size\s+issue|too\s+tight|too\s+loose|briefs|swimsuit|underwear|lingerie)\b", "items that cannot be returned innerwear RET-5"),
    (r"\b(festive|sale\s+mein|festival)\b", "festive sale returns RET-8"),
    (r"\b(collection\s+free|pickup\s+free|free\s+collection|free\s+pickup|pickup\s+fee|collection\s+fee)\b", "reverse pickup fee RET-6"),
    (r"\b(torn|phata|crack|broken|damaged|defective|came\s+torn|खराब|तुटलेले|फुटलेले)\b", "damaged defective wrong item DWM-1 DWM-2 DWM-3"),
    (r"\bsend\s+back\b", "reverse pickup"),
    (r"\b(cancel\s+order|रद्द\s+करायची|रद्द\s+करा)\b", "cancellation CAN-1"),
    (r"\b(money\s+back|रिफंड\s+कधी|रिफंड\s+मिळालेला\s+नाही|पैसे\s+परत)\b", "refund timeline REF-2 REF-1"),
    (r"\b(nobody\s+came|delivery\s+person\s+didn'?t\s+come|not\s+home)\b", "delivery attempts courier DEL-3"),
    (r"\b(late|late\s+delivery|arrived\s+a\s+day\s+late|arrived\s+late|उशीर)\b", "late deliveries express fee compensation DEL-4"),
    (r"\bcash\s+on\s+delivery\b", "COD"),
    (r"\bpremium\s+member\b", "ZentroMart Plus"),
    (r"\barea\s+code\b", "pincode"),
    (r"\binspection\b", "quality check"),
    (r"\bsent\s+back\s+to\s+warehouse\b", "return to origin"),
    (r"\b(never\s+placed|unauthorized|fraud|not\s+my\s+order)\b", "unauthorized fraud account security ESC-1"),
]

def rule_based_rewrite(text: str) -> str:
    res = text
    for pat, rep in FALLBACK_REPLACEMENTS:
        res = re.sub(pat, rep, res, flags=re.IGNORECASE)
    return res.strip()

@functools.lru_cache(maxsize=512)
def rewrite_query(customer_message: str) -> str:
    """
    Translates customer message into document vocabulary using fast deterministic rules.
    Preserves exact SKUs, dates, and order numbers while conserving API quota.
    """
    return rule_based_rewrite(customer_message)
