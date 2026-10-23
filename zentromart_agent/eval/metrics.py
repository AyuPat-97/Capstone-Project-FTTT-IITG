import re
from typing import List, Tuple

# Month equivalents for date normalization
MONTH_MAP = {
    "01": ["january", "जनवरी"], "02": ["february", "फ़रवरी"], "03": ["march", "मार्च"],
    "04": ["april", "अप्रैल"], "05": ["may", "मई"], "06": ["june", "जून"],
    "07": ["july", "जुलाई"], "08": ["august", "अगस्त"], "09": ["september", "सितंबर"],
    "10": ["october", "अक्टूबर"], "11": ["november", "नवंबर"], "12": ["december", "दिसंबर"]
}

def normalize_text_for_matching(text: str) -> str:
    """Normalizes text for flexible fact matching (currency, dates, casing)."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"₹\s*", "", t)
    t = re.sub(r"\brs\.?\s*", "", t)
    t = re.sub(r"\brupees\b", "", t)
    t = re.sub(r"[,._\-:;!?*`]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def check_must_include(reply: str, must_include_str: str) -> Tuple[int, int, float]:
    """
    Checks must_include facts.
    Facts separated by ' ; '.
    Alternatives within a fact separated by ' / ' (any one is sufficient).
    Returns: (facts_found, total_facts, coverage_pct)
    """
    if not must_include_str or str(must_include_str).strip() == "" or str(must_include_str).lower() == "nan":
        return 0, 0, 100.0

    facts = [f.strip() for f in str(must_include_str).split(";") if f.strip()]
    if not facts:
        return 0, 0, 100.0

    norm_reply = normalize_text_for_matching(reply)
    found_count = 0

    for fact in facts:
        alternatives = [a.strip() for a in fact.split("/") if a.strip()]
        fact_satisfied = False

        for alt in alternatives:
            norm_alt = normalize_text_for_matching(alt)
            if not norm_alt:
                continue

            # Check direct substring
            if norm_alt in norm_reply:
                fact_satisfied = True
                break

            # Date pattern matching e.g. 2026-10-27 <-> 27 october 2026
            date_m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", alt.strip())
            if date_m:
                y, m, d = date_m.group(1), date_m.group(2), str(int(date_m.group(3)))
                m_names = MONTH_MAP.get(m, [])
                for mn in m_names:
                    if f"{d} {mn} {y}" in norm_reply or f"{d} {mn}" in norm_reply:
                        fact_satisfied = True
                        break
            if fact_satisfied:
                break

        if fact_satisfied:
            found_count += 1

    pct = (found_count / len(facts)) * 100.0
    return found_count, len(facts), pct

def check_must_not_include(reply: str, must_not_include_str: str) -> bool:
    """Returns True if none of the forbidden phrases appear in the reply."""
    if not must_not_include_str or str(must_not_include_str).strip() == "" or str(must_not_include_str).lower() == "nan":
        return True

    phrases = [p.strip() for p in str(must_not_include_str).split(";") if p.strip()]
    norm_reply = normalize_text_for_matching(reply)

    for p in phrases:
        norm_p = normalize_text_for_matching(p)
        if norm_p and norm_p in norm_reply:
            return False
    return True

def classify_decision(reply: str) -> str:
    """Heuristic classifier for answer, clarify, escalate, refuse."""
    r_lower = reply.lower()

    if any(k in r_lower for k in ["ticket", "escalat", "specialist team", "handing over", "transferred", "senior colleague"]):
        return "escalate"

    if any(k in r_lower for k in [
        "cannot be cancelled", "not eligible", "cannot return", "not returnable",
        "outside the return window", "we do not offer compensation", "policy does not allow",
        "cannot share", "can't share", "cannot disclose", "privacy policy", "personal details"
    ]):
        # If providing policy alternative (refuse at door / return after delivery), it's informative answer
        if any(alt in r_lower for alt in ["refuse the parcel", "refuse it at", "refuse delivery", "return it after delivery", "मना कर सकते हैं"]):
            return "answer"
        return "refuse"

    # Clarification checks
    courtesy_closings = [
        "anything else", "further assistance", "more questions", "help you with anything",
        "madad chahiye", "aur sahayata", "aur jankari", "aur jaankari", "aur madad"
    ]
    cleaned_reply = reply.strip()
    for cc in courtesy_closings:
        cleaned_reply = re.sub(rf"[\s,.\-]*[^\.\n?!]*{cc}[^\.\n?!]*\?*$", "", cleaned_reply, flags=re.IGNORECASE).strip()

    clarify_patterns = [
        "could you please share", "please provide", "share your order id",
        "which time slot", "phone number or delivery pincode", "which product",
        "which item", "which category", "kaun sa saman", "kaun sa product",
        "kis product", "kis category", "order id share", "order id batayein",
        "kripya apna order id", "kripya batayein"
    ]
    if any(k in r_lower for k in clarify_patterns) or (cleaned_reply.endswith("?") and not any(cc in r_lower for cc in courtesy_closings)):
        return "clarify"

    return "answer"

def section_recall(section_ids_used: List[str], gold_evidence_str: str) -> bool:
    """Checks if any gold evidence section was retrieved."""
    if not gold_evidence_str or str(gold_evidence_str).strip() == "" or str(gold_evidence_str).lower() == "nan":
        return True

    golds = [g.strip() for g in str(gold_evidence_str).split(";") if g.strip()]
    used_set = set(section_ids_used)
    for g in golds:
        if g in used_set:
            return True
    return False
