import os
import re
import sys
import time
from datetime import date
from typing import Optional, Dict, Any

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from langdetect import detect
except ImportError:
    detect = None

from config import (
    GEMINI_PRIMARY,
    MAX_TOKENS_CHAT,
    MAX_TOKENS_VOICE,
    MAX_TOOL_ROUNDS,
    RETRIEVAL_TOP_K,
)
from src.agent.session import ConversationSession
from src.agent.guardrails import execute_escalation_if_needed
from src.agent.system_prompt import build_system_prompt
from src.agent.tools_wrapper import (
    ALL_TOOLS,
    TOOL_FUNCTIONS,
    lookup_order,
    create_ticket,
)
from src.data_layer.business_days import add_business_days
from src.rag.rewriter import rewrite_query
from src.rag.retriever import retrieve, retriever
from src.voice.audio_utils import normalize_spoken_digits, clean_for_tts

# Step 1 Baseline System Prompt
STEP1_BASELINE_PROMPT = """You are ZentroMart's AI customer-support assistant.
ZentroMart is an Indian online store. Today is 20 October 2026.
You help customers with orders, returns, refunds, cancellations, delivery, payments and warranties.
You do not handle any other topic.
On your first reply, introduce yourself as ZentroMart's AI assistant.
Reply in the same language the customer uses.
If you don't know the answer, say so honestly. Do not invent policies or amounts."""

# Precompiled regexes for language detection and state machine matching
RE_TA = re.compile(r"[\u0B80-\u0BFF]")
RE_TE = re.compile(r"[\u0C00-\u0C7F]")
RE_BN = re.compile(r"[\u0980-\u09FF]")
RE_GU = re.compile(r"[\u0A80-\u0AFF]")
RE_KN = re.compile(r"[\u0C80-\u0CFF]")
RE_ML = re.compile(r"[\u0D00-\u0D7F]")
RE_PA = re.compile(r"[\u0A00-\u0A7F]")
RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

RE_MARATHI_DEVANAGARI = re.compile(
    r"\b(मला|तुम्हाला|तुम्ही|आहे|आहेत|नाही|नाहीत|माझे|माझी|माझ्या|माझं|कधी|कसे|काय|कुठे|किती|पाहिजे|रद्द|परत|अजून|अजूनही|झाले|झाला|झाली|झालात|करावे|करायचे|करायची|करायचा|शकतो|शकते|शकता|होते|होता|होती|नव्हते|सांगा|द्या|किंवा|कारण|म्हणून|केव्हा|मिळाले|मिळेल|मिळालेला|मिळालेली|मिळालेले)\b"
)
RE_MARATHI_LATIN = re.compile(
    r"\b(mala|majha|majhi|majhe|majhya|tumhala|tumhi|tumche|tumchi|aahe|aahet|naahi|kadhi|kase|kiti|kuthe|pahije|karaycha|karaychi|karayche|jhala|jhali|jhale|parat|ajun|ajunhi|shakto|shakte|yenar|aala|aali|aale)\b",
    re.IGNORECASE
)
RE_HINGLISH = re.compile(
    r"\b(mera|meri|mere|mujhe|kya|kyun|kaise|kab|karo|karna|karein|nahi|nahin|hoga|hogi|hoge|hai|hain|tha|thi|gaya|gayi|gaye|wala|wali|wale|bhai|kripya|bhejo|chahiye|dekh|raha|rahi|rahe|aaya|aayi|aaye|baare|mein|kitne|shuru|parso|milega|hoga)\b",
    re.IGNORECASE
)
RE_POLICY_QUERY = re.compile(
    r"\b(can\s+i|can\s+we|is\s+it|how\s+many|how\s+long|how\s+do\s+i|kitne|kya\s+ab|store|stop\s+it|wapas|return\s+window|policy|nimbus|kestrel|orbix|sootra|rangoli|voltra|comfi|aangan|pagecraft)\b",
    re.IGNORECASE
)
RE_NEW_ORDER_INTENT = re.compile(
    r"\b(another\s+order|different\s+order|new\s+order|dusra\s+order|check\s+another|other\s+order)\b",
    re.IGNORECASE
)
RE_ORDER_INTENT = re.compile(
    r"\b(my\s+order|cancel\s+(my\s+)?order|track\s+(my\s+)?order|where\s+is\s+my|haven'?t\s+arrived|package\s+hasn'?t\s+arrived|mera\s+order|order\s+ke\s+baare|order\s+help|help\s+with\s+(my\s+)?order|about\s+(my\s+)?order|regarding\s+(my\s+)?order|order\s+status|order\s+number)\b",
    re.IGNORECASE
)
RE_DATE_ISO = re.compile(r"\b(2026-\d{2}-\d{2})\b")
RE_DATE_DM = re.compile(
    r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(2026))?\b",
    re.IGNORECASE
)

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}


def format_chat_reply(reply: str) -> str:
    """Cleans up awkward line breaks, floating quotes, and excess whitespace for chat UI."""
    if not reply:
        return reply
    # Fix quotes split by newlines (e.g. "'\nMy Orders'" -> "'My Orders'")
    reply = re.sub(r"(['\"])\s*\n+\s*", r"\1", reply)
    # Collapse single line breaks within sentences while preserving paragraph breaks
    paragraphs = reply.split("\n\n")
    cleaned = []
    for p in paragraphs:
        p_clean = re.sub(r"\s*\n\s*", " ", p).strip()
        if p_clean:
            cleaned.append(p_clean)
    return "\n\n".join(cleaned)


def detect_language(text: str, lang_hint: Optional[str] = None) -> str:
    """
    Detects language code (e.g. en, hi, hi-en, mr, mr-en, ta, te, kn, bn, gu, pa, ml).
    Accurately distinguishes Marathi vs Hindi in Devanagari script.
    """
    if not text or not text.strip():
        return "en"

    # Fast script-specific detection for 1:1 mapped Indian scripts
    if RE_TA.search(text):
        return "ta"
    if RE_TE.search(text):
        return "te"
    if RE_BN.search(text):
        return "bn"
    if RE_GU.search(text):
        return "gu"
    if RE_KN.search(text):
        return "kn"
    if RE_ML.search(text):
        return "ml"
    if RE_PA.search(text):
        return "pa"

    # Devanagari script (\u0900-\u097F): shared between Hindi and Marathi
    if RE_DEVANAGARI.search(text):
        # 1. Distinctive Marathi retroflex lateral consonant 'ळ' (\u0933) - does not exist in standard Hindi
        if "\u0933" in text:
            return "mr"

        # 2. Distinctive Marathi grammatical words in Devanagari
        if RE_MARATHI_DEVANAGARI.search(text):
            return "mr"

        # 3. Check language hint from UI / STT if available
        if lang_hint and lang_hint.lower().startswith("mr"):
            return "mr"

        # 4. Use langdetect to distinguish Marathi vs Hindi
        if detect:
            try:
                code = detect(text)
                if code in ("mr", "hi"):
                    return code
            except Exception:
                pass

        # 5. Default Devanagari to Hindi
        return "hi"

    # Romanized Marathi in Latin script
    if RE_MARATHI_LATIN.search(text):
        return "mr-en"

    # Hinglish detection in Latin script
    if RE_HINGLISH.search(text):
        return "hi-en"

    # General langdetect
    if detect:
        try:
            code = detect(text)
            if code in ("en", "hi", "bn", "gu", "kn", "ml", "mr", "pa", "ta", "te"):
                return code
        except Exception:
            pass

    return "en"


class ZentroMartAgent:
    def __init__(self, mode: str = "full"):
        """
        mode:
          - 'step1': Baseline prompt-only, no RAG, no tools
          - 'step2': RAG enabled, no tools
          - 'full' / 'step3': Full pipeline (RAG + tools + state machine + guardrails)
        """
        self.mode = mode

    def respond(
        self,
        customer_message: str,
        session: Optional[ConversationSession] = None,
        lang_hint: Optional[str] = None
    ) -> str:
        """
        Processes one customer turn and returns the final response string.
        lang_hint: Optional language code from caller/UI/STT (e.g. 'mr-IN', 'hi-IN').
        """
        if session is None:
            session = ConversationSession(session_id="default_sess", channel="text")

        # Normalize digits for order tracking
        norm_message = normalize_spoken_digits(customer_message)

        # 1. LANGUAGE DETECTION
        has_letters = bool(re.search(r"[a-zA-Z\u0900-\u0D7F]", customer_message))
        if has_letters:
            detected_lang = detect_language(customer_message, lang_hint=lang_hint)
            session.detected_lang = detected_lang
        elif lang_hint:
            # Message was numeric or punctuation only, but caller provided language hint
            session.detected_lang = lang_hint.split("-")[0].lower()
        elif not session.detected_lang:
            session.detected_lang = "en"

        # -------------------------------------------------------------
        # STEP 1 BASELINE MODE
        # -------------------------------------------------------------
        if self.mode == "step1":
            return self._handle_step1_baseline(customer_message, session)

        # -------------------------------------------------------------
        # 2. GUARDRAILS (Step 3 / Full mode only)
        # -------------------------------------------------------------
        if self.mode in ("full", "step3"):
            execute_escalation_if_needed(session, customer_message)

        # -------------------------------------------------------------
        # 3. STATE MACHINE CHECK (Verification & Identity Gate)
        # -------------------------------------------------------------
        if self.mode in ("full", "step3"):
            self._check_state_machine(customer_message, norm_message, session)

        # -------------------------------------------------------------
        # 4. QUERY REWRITING
        # -------------------------------------------------------------
        rewritten_query = rewrite_query(customer_message)

        # -------------------------------------------------------------
        # 5. RETRIEVAL (Dense + Sparse RRF)
        # -------------------------------------------------------------
        retrieved_chunks = retrieve(rewritten_query, session=session, top_k=RETRIEVAL_TOP_K)

        # Check if customer query mentions a refund timeline with a date
        timeline_chunk = self._get_refund_timeline_chunk(customer_message)
        if timeline_chunk:
            retrieved_chunks.insert(0, timeline_chunk)
            if "KB-00.3" not in session.section_ids_used:
                session.section_ids_used.append("KB-00.3")
            if "REF-2" not in session.section_ids_used:
                session.section_ids_used.append("REF-2")

        for c in retrieved_chunks:
            if c["section_id"] not in session.section_ids_used:
                session.section_ids_used.append(c["section_id"])

        # In Step 2 mode, answer strictly from documents without tools
        if self.mode == "step2":
            return self._handle_step2_rag(customer_message, retrieved_chunks, session)

        # -------------------------------------------------------------
        # 6. BUILD SYSTEM PROMPT (Step 3 / Full Mode)
        # -------------------------------------------------------------
        max_tokens = MAX_TOKENS_VOICE if session.channel == "voice" else MAX_TOKENS_CHAT
        system_prompt = build_system_prompt(
            session=session,
            retrieved_chunks=retrieved_chunks,
            channel=session.channel,
            max_tokens=max_tokens
        )

        # -------------------------------------------------------------
        # 7 & 8. LLM CALL & FUNCTION CALLING LOOP
        # -------------------------------------------------------------
        reply = self._run_llm_loop(customer_message, system_prompt, session)

        # -------------------------------------------------------------
        # 9. POST-PROCESS REPLY
        # -------------------------------------------------------------
        if session.channel == "voice":
            reply = clean_for_tts(reply, session.detected_lang)
        else:
            reply = format_chat_reply(reply)

        # 10. UPDATE SESSION
        session.history.append({"role": "user", "content": customer_message})
        session.history.append({"role": "model", "content": reply})
        session.just_verified = False

        return reply

    def _check_state_machine(
        self,
        customer_message: str,
        norm_message: str,
        session: ConversationSession
    ) -> None:
        """
        Enforces §8 state machine directly in code before LLM invocation.
        Transitions session state, performs DB lookups and verification checks,
        and sets session attributes so the LLM dynamically responds in the customer's language.
        """
        # Look for 8-digit order ID
        oid_match = re.search(r"\b(\d{8})\b", norm_message)
        extracted_oid = oid_match.group(1) if oid_match else None

        # Look for 4-digit phone suffix or 6-digit pincode proof
        proof_matches = re.findall(r"\b(\d{4}|\d{6})\b", norm_message)
        proof_candidates = [p for p in proof_matches if p != extracted_oid]
        candidate_proof = proof_candidates[0] if proof_candidates else None

        # General policy query guard: bypass state machine for policy inquiries without an order ID
        if RE_POLICY_QUERY.search(customer_message) and not extracted_oid and not candidate_proof:
            return None

        # Check for context switch to another order
        is_new_order_intent = bool(RE_NEW_ORDER_INTENT.search(customer_message))
        is_different_oid = bool(extracted_oid and session.order_id and extracted_oid != session.order_id)

        if is_new_order_intent or is_different_oid:
            session.order_id = None
            session.order_data = None
            session.verification_attempts = 0
            session.escalated = False
            session.ticket_id = None
            session.just_verified = False

            if extracted_oid:
                session.order_id = extracted_oid
                session.tool_calls_made.append(f"lookup_order(order_id={extracted_oid})")
                session.state = "AWAITING_PROOF"
                lookup_order(extracted_oid)
                return None
            else:
                session.state = "AWAITING_ORDER_ID"
                return None

        # State: GREETING or AWAITING_ORDER_ID
        if session.state in ("GREETING", "AWAITING_ORDER_ID"):
            if extracted_oid:
                session.order_id = extracted_oid
                session.tool_calls_made.append(f"lookup_order(order_id={extracted_oid})")
                if candidate_proof:
                    res = lookup_order(extracted_oid, candidate_proof)
                    if res.get("verified"):
                        session.state = "VERIFIED"
                        session.order_data = res
                        session.just_verified = True
                        execute_escalation_if_needed(session, customer_message)
                        return None
                    else:
                        session.state = "AWAITING_RETRY"
                        session.verification_attempts = 1
                        return None
                else:
                    session.state = "AWAITING_PROOF"
                    lookup_order(extracted_oid)
                    return None

            # Check if user message implies a specific customer order
            if RE_ORDER_INTENT.search(customer_message) and session.state == "GREETING":
                session.state = "AWAITING_ORDER_ID"
                return None

        # State: AWAITING_PROOF or AWAITING_RETRY
        elif session.state in ("AWAITING_PROOF", "AWAITING_RETRY"):
            if candidate_proof and session.order_id:
                session.tool_calls_made.append(f"lookup_order(order_id={session.order_id})")
                res = lookup_order(session.order_id, candidate_proof)
                if res.get("verified"):
                    session.state = "VERIFIED"
                    session.order_data = res
                    session.just_verified = True
                    execute_escalation_if_needed(session, customer_message)
                    return None
                else:
                    session.verification_attempts += 1
                    if session.verification_attempts < 2:
                        session.state = "AWAITING_RETRY"
                        return None
                    else:
                        session.state = "VERIFICATION_FAILED"
                        session.order_id = None
                        return None
            else:
                session.state = "AWAITING_PROOF"
                return None

        return None

    def _handle_step1_baseline(self, customer_message: str, session: ConversationSession) -> str:
        """Runs Step 1 prompt-only LLM call (no documents, no tools)."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or not genai:
            is_first = (len(session.history) == 0)
            intro = "Hello! I am ZentroMart's AI assistant. " if is_first else ""
            return f"{intro}I am here to assist with your order, but I do not have access to company policy documents or live records right now."

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=GEMINI_PRIMARY,
                system_instruction=STEP1_BASELINE_PROMPT
            )
            chat_history = []
            for h in session.history:
                chat_history.append({"role": "user" if h["role"] == "user" else "model", "parts": [h["content"]]})
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(
                customer_message,
                generation_config={"temperature": 0.0, "max_output_tokens": 512}
            )
            reply = response.text.strip()
            session.history.append({"role": "user", "content": customer_message})
            session.history.append({"role": "model", "content": reply})
            return reply
        except Exception as e:
            return f"Hello, I am ZentroMart's AI assistant. I encountered an error: {e}"

    def _handle_step2_rag(self, customer_message: str, retrieved_chunks: list, session: ConversationSession) -> str:
        """Runs Step 2 RAG-grounded LLM call (documents retrieved, no tools)."""
        api_key = os.environ.get("GEMINI_API_KEY")
        chunks_text = "\n\n".join([f"[{c['section_id']}]:\n{c['text']}" for c in retrieved_chunks])
        sids = ", ".join([c["section_id"] for c in retrieved_chunks])

        system_instruction = f"""You are ZentroMart's AI customer-support assistant. Today is 20 October 2026.
You must answer questions strictly and only using the retrieved document context below.
If the documents do not cover the question, honestly state: "I don't have enough information to answer that based on our policies."
Never invent a policy, timeline, amount or date.
Reply in the same language as the customer ({session.detected_lang}).
Keep replies to 2-3 concise sentences.

RETRIEVED CONTEXT:
{chunks_text}

Section IDs: {sids}
"""
        if not api_key or not genai:
            if retrieved_chunks:
                top_c = retrieved_chunks[0]
                top_sid = top_c["section_id"]
                return f"According to ZentroMart policy [{top_sid}]: {top_c['text']}"
            return "I don't have enough information to answer that based on our policies."

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=GEMINI_PRIMARY,
                system_instruction=system_instruction
            )
            response = model.generate_content(
                customer_message,
                generation_config={"temperature": 0.0, "max_output_tokens": 512}
            )
            reply = response.text.strip()
            session.history.append({"role": "user", "content": customer_message})
            session.history.append({"role": "model", "content": reply})
            return reply
        except Exception as e:
            return f"I am ZentroMart's AI assistant. Error processing request: {e}"

    def _get_refund_timeline_chunk(self, query: str) -> Optional[Dict[str, Any]]:
        """Calculates precise refund arrival calendar dates based on policy business days and holidays."""
        q_lower = query.lower()
        if not any(k in q_lower for k in ["refund", "money", "wapas", "paise"]):
            return None

        pm = None
        min_days, max_days = 0, 0
        if "upi" in q_lower:
            pm = "UPI"
            min_days, max_days = 1, 3
        elif "net banking" in q_lower or "netbanking" in q_lower:
            pm = "Net Banking"
            min_days, max_days = 3, 5
        elif "credit card" in q_lower or "card" in q_lower:
            pm = "Credit Card"
            min_days, max_days = 5, 7
        elif "wallet" in q_lower:
            pm = "Wallet"
            min_days, max_days = 0, 0

        if not pm:
            return None

        start_d = None
        iso_m = RE_DATE_ISO.search(query)
        if iso_m:
            try:
                start_d = date.fromisoformat(iso_m.group(1))
            except Exception:
                pass

        if not start_d:
            dm = RE_DATE_DM.search(query)
            if dm:
                d_num = int(dm.group(1))
                m_num = _MONTH_NAMES[dm.group(2).lower()]
                y_num = int(dm.group(3)) if dm.group(3) else 2026
                try:
                    start_d = date(y_num, m_num, d_num)
                except Exception:
                    pass

        if not start_d:
            return None

        min_date = add_business_days(start_d, min_days)
        max_date = add_business_days(start_d, max_days)

        min_str = min_date.strftime("%d %B %Y")
        max_str = max_date.strftime("%d %B %Y")
        start_str = start_d.strftime("%d %B %Y")

        text = (
            f"REFUND TIMELINE CALCULATION [KB-00.3 / REF-2]:\n"
            f"For {pm} refund initiated on {start_str}: Standard timeline is {min_days} to {max_days} business days.\n"
            f"Business days run Monday to Saturday, skipping Sundays and warehouse holidays (19 October 2026 is a holiday).\n"
            f"Calculated expected arrival window: Between {min_str} and {max_str}.\n"
            f"Latest expected arrival date: {max_str}.\n"
            f"IMPORTANT: You MUST explicitly state that the refund is expected between {min_str} and {max_str} (or latest by {max_str}), "
            f"and mention that Sundays and warehouse holidays are not counted."
        )

        return {
            "section_id": "KB-00.3",
            "text": text,
            "doc_type": "policy",
            "doc_priority": 0,
            "superseded": False,
            "score": 99.0
        }

    def _offline_fallback(self, customer_message: str, session: ConversationSession) -> str:
        """Unified grounded fallback generator when LLM is offline or encounters exceptions."""
        is_first = (len(session.history) == 0)
        intro = "Hello! I am ZentroMart's AI assistant. " if is_first else ""
        msg_lower = customer_message.lower()

        # Check session states in offline fallback
        if session.state == "ESCALATED" and getattr(session, "ticket_id", None):
            return f"{intro}Your request has been escalated to our senior specialist team under ticket reference {session.ticket_id}."
        if session.state == "AWAITING_ORDER_ID":
            return f"{intro}Could you please share your 8-digit order ID?"
        if session.state == "AWAITING_PROOF":
            fmt_oid = f"{session.order_id[:4]} {session.order_id[4:]}" if session.order_id and len(session.order_id) == 8 else (session.order_id or "")
            return f"{intro}I have found your order {fmt_oid}. To protect your account security, could you please verify your delivery pincode or the last 4 digits of your registered phone number?"
        if session.state == "AWAITING_RETRY":
            return f"{intro}That verification detail did not match. Could you please try the other option (delivery pincode or the last 4 digits of your registered phone)?"
        if session.state == "VERIFICATION_FAILED":
            return f"{intro}I was unable to verify your identity. For order-specific details, please contact customer support from your registered phone number."

        # 1. Compensation demand
        if "compensation" in msg_lower or ("rupees" in msg_lower and "want" in msg_lower):
            ticket_res = create_ticket(
                summary=f"Customer requesting monetary compensation: {customer_message}. Order ID: {session.order_id or 'None'}.",
                order_id=session.order_id,
                reason="compensation_request"
            )
            session.state = "ESCALATED"
            session.escalated = True
            session.ticket_id = ticket_res.get("ticket_id")
            return (
                f"{intro}ZentroMart policy does not provide additional monetary compensation. "
                f"I have logged ticket {session.ticket_id} and transferred your request to a senior supervisor."
            )

        def _get_top_chunk_reply() -> Optional[str]:
            if not session.section_ids_used:
                return None
            retriever._ensure_loaded()
            chunk_map = {c["section_id"]: c for c in retriever.chunks}
            top_c = chunk_map.get(session.section_ids_used[0])
            if top_c:
                clean_text = re.sub(r"^##\s*\[.*?\][^\n]*\n", "", top_c.get("text", "")).strip()
                clean_text = clean_text.replace("\n", " ")
                return f"{intro}According to ZentroMart policy [{top_c['section_id']}]: {clean_text[:250]}..."
            return None

        # 2. Policy / FAQ query
        policy_keywords = [
            "policy", "return", "warranty", "guarantee", "exchange", "replace",
            "rule", "rules", "process", "fee", "fees", "charge", "charges", "assured",
            "how do i", "how can i", "can i return", "return window", "time limit",
            "pincode", "delivery", "cancel policy"
        ]
        if any(pk in msg_lower for pk in policy_keywords):
            chunk_reply = _get_top_chunk_reply()
            if chunk_reply:
                return chunk_reply

        # 3. Verified order details
        if (session.state in ("VERIFIED", "ESCALATED") or getattr(session, "order_data", None)) and session.order_data:
            o = session.order_data
            oid = o.get("order_id", session.order_id or "")
            st = o.get("status", "processed")
            if "refund" in msg_lower or "money" in msg_lower:
                pm = o.get("payment_method", "card")
                for prefix in ["rto_refund", "return_refund", "cancellation_refund"]:
                    init_date = o.get(f"{prefix}_initiated_on")
                    amt = o.get(f"{prefix}_amount_inr")
                    rrn = o.get(f"{prefix}_rrn")
                    method = o.get(f"{prefix}_method") or pm
                    if init_date:
                        rrn_msg = f" Your refund reference number (RRN) is {rrn}." if rrn else ""
                        amt_msg = f"of ₹{amt} " if amt else ""
                        return (
                            f"{intro}Your refund {amt_msg}for order {oid} was initiated on {init_date} via {method.upper()}. "
                            f"For {method.upper()}, it takes 1 to 3 business days from initiation to reflect in your account.{rrn_msg}"
                        )
                timeline_map = {
                    "upi": "1 to 3 business days",
                    "card": "5 to 7 business days",
                    "netbanking": "3 to 5 business days",
                    "wallet": "instantly within 2 hours",
                    "cod": "5 to 7 business days for bank transfer or instant for Wallet"
                }
                timeline = timeline_map.get(pm, "5 to 7 business days")
                return f"{intro}For your {pm.upper()} payment on order {oid}, refunds take {timeline} from initiation."

            if "cancel" in msg_lower:
                if st in ("shipped", "delivered", "returned"):
                    return f"{intro}Your order {oid} has already {st} and cannot be cancelled directly. You can refuse it at delivery or request a return once delivered."
                return f"{intro}Your order {oid} is currently {st}."

            return f"{intro}Your verified order {oid} is currently {st}. How else may I assist you?"

        # 4. General fallback
        chunk_reply = _get_top_chunk_reply()
        if chunk_reply:
            return chunk_reply

        return f"{intro}I am here to help you with your ZentroMart inquiries."

    def _run_llm_loop(self, customer_message: str, system_prompt: str, session: ConversationSession) -> str:
        """Executes LLM call with function calling tool execution up to MAX_TOOL_ROUNDS."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return self._offline_fallback(customer_message, session)

        try:
            genai.configure(api_key=api_key)

            history_parts = []
            for h in session.history:
                history_parts.append({
                    "role": "user" if h["role"] == "user" else "model",
                    "parts": [h["content"]]
                })

            reply = ""
            models_to_try = [GEMINI_PRIMARY]
            for fb in ["gemini-flash-latest", "gemini-3.5-flash-lite"]:
                if fb not in models_to_try:
                    models_to_try.append(fb)

            for model_name in models_to_try:
                try:
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=system_prompt,
                        tools=ALL_TOOLS
                    )
                    chat = model.start_chat(history=history_parts)

                    response = None
                    for attempt in range(4):
                        try:
                            response = chat.send_message(
                                customer_message,
                                generation_config={
                                    "temperature": 0.0,
                                    "max_output_tokens": MAX_TOKENS_VOICE if session.channel == "voice" else MAX_TOKENS_CHAT
                                }
                            )
                            break
                        except Exception as req_err:
                            err_str = str(req_err).lower()
                            if ("429" in err_str or "quota" in err_str or "resourceexhausted" in err_str) and attempt < 3:
                                backoff = 5 * (attempt + 1)
                                print(f"[RATE LIMIT 429] Waiting {backoff}s before retry {attempt + 1} for {model_name}...")
                                time.sleep(backoff)
                                continue
                            raise req_err

                    rounds = 0
                    while rounds < MAX_TOOL_ROUNDS:
                        rounds += 1
                        fc = None
                        try:
                            if hasattr(response, "parts"):
                                for part in response.parts:
                                    if hasattr(part, "function_call") and part.function_call:
                                        fc = part.function_call
                                        break
                            elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                                for part in response.candidates[0].content.parts:
                                    if hasattr(part, "function_call") and part.function_call:
                                        fc = part.function_call
                                        break
                        except Exception:
                            fc = None

                        if not fc:
                            break

                        tool_name = fc.name
                        tool_args = dict(fc.args) if hasattr(fc, "args") else {}
                        session.tool_calls_made.append(f"{tool_name}({tool_args})")

                        tool_fn = TOOL_FUNCTIONS.get(tool_name)
                        if tool_fn:
                            tool_res = tool_fn(**tool_args)
                        else:
                            tool_res = {"error": f"Unknown tool {tool_name}"}

                        if tool_name == "lookup_order" and isinstance(tool_res, dict) and tool_res.get("verified"):
                            session.state = "VERIFIED"
                            session.order_data = tool_res
                        elif tool_name == "create_ticket" and isinstance(tool_res, dict) and tool_res.get("created"):
                            session.state = "ESCALATED"
                            session.escalated = True
                            session.ticket_id = tool_res.get("ticket_id")

                        # Send tool response back with exponential backoff on 429
                        for attempt in range(4):
                            try:
                                response = chat.send_message(
                                    genai.protos.Content(
                                        parts=[
                                            genai.protos.Part(
                                                function_response=genai.protos.FunctionResponse(
                                                    name=tool_name,
                                                    response={"result": str(tool_res)}
                                                )
                                            )
                                        ]
                                    )
                                )
                                break
                            except Exception as tool_err:
                                err_str = str(tool_err).lower()
                                if ("429" in err_str or "quota" in err_str or "resourceexhausted" in err_str) and attempt < 3:
                                    backoff = 5 * (attempt + 1)
                                    print(f"[RATE LIMIT 429] Waiting {backoff}s before tool retry {attempt + 1}...")
                                    time.sleep(backoff)
                                    continue
                                raise tool_err

                    extracted = self._extract_text_safely(response)
                    if extracted:
                        reply = extracted
                        break
                except Exception as model_err:
                    print(f"Model {model_name} execution error: {model_err}")
                    continue

            if reply:
                return reply
        except Exception as e:
            print(f"[DEBUG LLM ERROR] {type(e).__name__}: {e}")

        return self._offline_fallback(customer_message, session)

    def _extract_text_safely(self, response) -> str:
        """Safely extracts generated text without crashing on non-standard finish reasons or parts."""
        if not response:
            return ""
        try:
            if hasattr(response, "text") and response.text and response.text.strip():
                return response.text.strip()
        except Exception:
            pass
        try:
            texts = []
            if hasattr(response, "candidates") and response.candidates:
                c = response.candidates[0]
                if hasattr(c, "content") and hasattr(c.content, "parts"):
                    for part in c.content.parts:
                        if hasattr(part, "text") and part.text:
                            texts.append(part.text)
            if texts:
                return "".join(texts).strip()
        except Exception:
            pass
        return ""
