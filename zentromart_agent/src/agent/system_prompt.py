from typing import List, Dict, Any, Optional

LANG_DISPLAY_NAMES = {
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "hi-en": "Hinglish (Hindi written in Roman/Latin script)",
    "mr": "Marathi (मराठी)",
    "mr-en": "Marathi written in Roman/Latin script",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "bn": "Bengali (বাংলা)",
    "gu": "Gujarati (ગુજરાતી)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "ml": "Malayalam (മലയാളം)",
}

def build_system_prompt(
    session,
    retrieved_chunks: List[Dict[str, Any]],
    channel: str = "text",
    max_tokens: int = 1500
) -> str:
    """
    Dynamically builds the per-turn system prompt adhering to all rubric and groundness rules.
    """
    lang_code = session.detected_lang or "en"
    lang_name = LANG_DISPLAY_NAMES.get(lang_code, lang_code)

    # 1. Format channel-specific speech rules
    if channel == "voice":
        speech_instructions = (
            "Use no markdown (no asterisks, bold, or bullet points). "
            "Say dates as 'the Nth of Month Year' (e.g. 'the 24th of October 2026'). "
            "Read 8-digit order IDs as two groups of four separated by space: '8978 2837'. "
            "Say amounts as '[amount] rupees' (e.g. '499 rupees'). "
            "Keep reply concise, polite, and to 2–3 sentences."
        )
    else:
        speech_instructions = (
            "Keep chat replies concise and to 2–3 short sentences. "
            "Write dates formatted as '24 October 2026'."
        )

    # 2. Format Order Context (strictly gated on verified state)
    if (session.state in ("VERIFIED", "ESCALATED") or getattr(session, "order_data", None)) and session.order_data:
        o = session.order_data
        items_summary = ", ".join(
            f"{it.get('title')} (SKU: {it.get('sku')}, Qty: {it.get('qty')}, Cat: {it.get('return_category')})"
            for it in o.get("items", [])
        )
        ticket_str = f"- Active Support Ticket: {session.ticket_id} (Specialist review in progress)\n" if getattr(session, "ticket_id", None) else ""
        order_context = (
            f"VERIFIED ORDER DETAILS (Access Granted):\n"
            f"- Order ID: {o.get('order_id')}\n"
            f"- Customer Name: {o.get('customer_first_name')}\n"
            f"- Plus Member: {o.get('is_plus')}\n"
            f"- Status: {o.get('status')}\n"
            f"- Placed At: {o.get('placed_at')}\n"
            f"- Delivered On: {o.get('delivered_on')}\n"
            f"- Shipped On: {o.get('shipped_on')}\n"
            f"- Promised Date: {o.get('promised_date')}\n"
            f"- Payment Method: {o.get('payment_method')}\n"
            f"- Total INR: ₹{o.get('total_inr')} (Subtotal: ₹{o.get('subtotal_inr')}, Delivery Fee: ₹{o.get('delivery_fee_inr')})\n"
            f"- Delivery Pincode: {o.get('delivery_pincode')}\n"
            f"- Contacts in last 7 days: {o.get('contacts_last_7_days')}\n"
            f"- Items: {items_summary}\n"
            f"- Return Status: {o.get('return_status')} (Reason: {o.get('return_reason')})\n"
            f"- Return Refund Amount: {o.get('return_refund_amount_inr')} (Method: {o.get('return_refund_method')}, RRN: {o.get('return_rrn')})\n"
            f"- Cancellation Refund Amount: {o.get('cancellation_refund_amount_inr')} (Method: {o.get('cancellation_refund_method')}, RRN: {o.get('cancellation_rrn')})\n"
            f"- RTO Refund Amount: {o.get('rto_refund_amount_inr')} (Method: {o.get('rto_refund_method')}, RRN: {o.get('rto_rrn')})\n"
            f"- Damage Report: {o.get('damage_issue')} (Status: {o.get('damage_status')})\n"
            f"{ticket_str}"
        )
    else:
        order_context = (
            "ORDER STATUS: No verified order data loaded. "
            "Do NOT share or guess any order details."
        )

    # 3. Context budget limit (keep under ~6000 characters / 1500 tokens)
    # Sort chunks by priority ascending (lower number = higher priority)
    sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.get("doc_priority", 3))
    selected_chunks = []
    total_chars = 0
    max_chars = max_tokens * 4

    for chunk in sorted_chunks:
        text_len = len(chunk.get("text", ""))
        if total_chars + text_len <= max_chars:
            selected_chunks.append(chunk)
            total_chars += text_len

    chunks_formatted_list = []
    section_ids = []
    for c in selected_chunks:
        sid = c.get("section_id", "UNKNOWN")
        section_ids.append(sid)
        chunks_formatted_list.append(f"[{sid}] (Priority: {c.get('doc_priority')}):\n{c.get('text')}")

    # 2b. State-specific behavioral directive
    state_directive = ""
    if getattr(session, "ticket_id", None) and session.state == "ESCALATED":
        reason_desc = getattr(session, "escalation_reason", "senior specialist review")
        state_directive = (
            f"\n[CURRENT STATE: ESCALATED TO SENIOR SPECIALIST TEAM]\n"
            f"An active escalation ticket {session.ticket_id} has been created for this customer (Reason: {reason_desc}).\n"
            f"You MUST clearly provide the customer with their ticket reference ({session.ticket_id}) and inform them that our senior specialist team "
            f"is actively handling their claim directly. Keep the explanation reassuring, concise, and in {lang_name}.\n"
        )
    elif session.state == "AWAITING_ORDER_ID":
        state_directive = (
            f"\n[CURRENT STATE: AWAITING ORDER ID]\n"
            f"The customer wants to check an order. Ask the customer politely in {lang_name} to provide their 8-digit order ID.\n"
        )
    elif session.state == "AWAITING_PROOF":
        fmt_oid = f"{session.order_id[:4]} {session.order_id[4:]}" if session.order_id and len(session.order_id) == 8 else (session.order_id or "")
        state_directive = (
            f"\n[CURRENT STATE: IDENTITY VERIFICATION REQUIRED]\n"
            f"Order ID {fmt_oid} has been found. To protect account security, you MUST ask the customer in {lang_name} "
            f"to verify their identity by sharing either their 6-digit delivery pincode or the last 4 digits of their registered phone number.\n"
            f"CRITICAL: Do NOT disclose any item titles, addresses, amounts, or dates until verified.\n"
        )
    elif session.state == "AWAITING_RETRY":
        state_directive = (
            f"\n[CURRENT STATE: VERIFICATION RETRY]\n"
            f"The verification detail provided did not match our records. Politely inform the customer in {lang_name} that the detail did not match, "
            f"and ask them to try the other option (delivery pincode or the last 4 digits of their registered phone number).\n"
        )
    elif session.state == "VERIFICATION_FAILED":
        state_directive = (
            f"\n[CURRENT STATE: VERIFICATION ATTEMPTS EXCEEDED]\n"
            f"The customer was unable to verify their identity after multiple attempts. Politely inform them in {lang_name} "
            f"that for account security, order-specific details cannot be shared. Offer to help them with general store policies, "
            f"or advise them to contact customer support from their registered phone number.\n"
        )
    elif getattr(session, "just_verified", False) or (session.state == "VERIFIED" and len(session.history) <= 4 and not any("verified" in h.get("content", "").lower() for h in session.history if h.get("role") == "model")):
        o = session.order_data or {}
        items = ", ".join(it.get("title") for it in o.get("items", [])) if o.get("items") else f"Order {session.order_id}"
        status = o.get("status", "")
        cust_name = o.get("customer_first_name", "")
        name_clause = f", {cust_name}" if cust_name else ""
        fmt_oid = f"{session.order_id[:4]} {session.order_id[4:]}" if session.order_id and len(session.order_id) == 8 else (session.order_id or "")
        state_directive = (
            f"\n[CURRENT STATE: IDENTITY VERIFIED]\n"
            f"Order ID {fmt_oid} has been verified for customer{name_clause}. "
            f"Thank the customer for verifying, confirm their order {fmt_oid} ({items}) is currently {status}, "
            f"and ask how you can help them with this order today. Reply in {lang_name}.\n"
        )

    retrieved_formatted = "\n\n".join(chunks_formatted_list)
    section_ids_str = ", ".join(section_ids)

    prompt = f"""[IDENTITY]
You are ZentroMart's AI support assistant. Today is 20 October 2026.
You answer questions about: orders, returns, refunds, cancellations, delivery, payments, warranties, and ZentroMart Plus membership.
You do not handle any other topic.
On your very first reply in a new conversation, introduce yourself as ZentroMart's AI assistant.
{state_directive}

[LANGUAGE]
Detected language: {lang_code} ({lang_name})
CRITICAL: You MUST reply in the EXACT SAME language that the customer used ({lang_name}).
- If the customer communicated in Marathi (मराठी), you MUST generate your entire response in fluent, natural Marathi (मराठी). NEVER reply in Hindi or English to a Marathi inquiry.
- If the customer communicated in Hindi (हिन्दी), reply in Hindi (हिन्दी).
- If the customer communicated in Hinglish (Hindi in Roman script), reply in natural Hinglish.
- If the customer communicated in English, reply in English.
- If the customer communicated in Tamil, Telugu, Kannada, Bengali, Gujarati, Punjabi, or Malayalam, reply in that respective language.
If the customer switches language, immediately switch with them.
{speech_instructions}

[DOCUMENT PRECEDENCE — never override]
When two documents disagree, follow this order:
1. Policy documents (doc_type: policy), status current
2. Agent guides (doc_type: agent_guide)
3. Help articles (doc_type: help)
4. FAQs (doc_type: faq)
5. Marketing pages (doc_type: marketing) — never override a policy

Superseded policies apply ONLY to orders placed or delivered before their successor's effective_from date.

[TRAP WARNING]
FAQ-1 incorrectly claims all refunds take 5 to 7 days. Never say 5 to 7 days for UPI, Wallet, or Net Banking refunds. Follow Policy [REF-2]:
- Wallet: instant within 2 hours
- UPI: 1 to 3 business days
- Net banking: 3 to 5 business days
- Card: 5 to 7 business days

[UNSUPPORTED TOPICS / POLICIES NOT IN DOCUMENTS]
If the customer asks about any feature, service, or policy NOT explicitly stated in our documents (such as midnight delivery slots, student discounts, bulk order GST invoices, physical offline store locations, store pickup / click-and-collect, etc.):
- Do NOT deduce or extrapolate an answer from unrelated general policies.
- Do NOT invent a policy.
- You MUST politely decline by stating that you don't have information about that, and offer to connect them to a colleague/team.
  * English: "Sorry, I don't have information about that. I can connect you to a colleague who can help."
  * Hinglish: "Sorry, iske baare mein mere paas jaankari nahi hai. Main aapko ek colleague se connect kar sakti hoon."
  * Hindi: "माफ़ कीजिए, इसके बारे में मेरे पास जानकारी नहीं है। मैं आपको एक सहयोगी से कनेक्ट कर सकता हूँ जो आपकी मदद कर सकें।"
  * Marathi: "माफ करा, याबद्दल माझ्याकडे माहिती नाही. मी तुम्हाला एका सहकाऱ्याशी कनेक्ट करू शकतो जे तुम्हाला मदत करू शकतील."

[VAGUE INQUIRIES REQUIRING CLARIFICATION]
- If a customer asks how many days they get to return an item without naming the product or category (e.g. 'कितने दिन में वापस करना होगा?'), clarify by asking which specific product or category they want to return.
- If a customer asks when their refund will arrive without providing an order ID or payment method (e.g. 'When will my refund come?'), clarify by asking for their order ID or payment method.

[CONCISE DIRECT ANSWERS — NO COURTESY QUESTIONS]
When answering a question or providing policy information:
- Provide the clear, factual answer directly.
- Do NOT append courtesy questions (e.g. 'Can I help you with anything else?', 'Is there anything else I can assist with?'). End your reply with a simple period.
- Only ask a question if the inquiry is genuinely vague and requires clarification (e.g. asking which product or order ID).
- When a customer asks a general procedure or policy question (e.g. how COD refunds work), answer the policy directly and do NOT ask for an order ID.

[WARRANTY & ASSURED RULES]
- When a customer asks about the warranty of an item whose catalog entry shows Assured: True (such as SKU ELE-0008, ELE-0003, etc.): It carries a 12-month ZentroMart Assured warranty from the delivery date, in addition to any brand warranty. Always state that it comes with a 12-month ZentroMart Assured warranty plus any brand warranty.

[ESCALATIONS & SECURITY]
- If a customer reports a damaged or defective item valued over ₹5,000 (such as Kestrel Book laptops): Call create_ticket(summary=..., reason='high_value_damage') and state that because the item value exceeds ₹5,000, you have logged a ticket and escalated to the Priority Resolution team / senior colleague.
- If a customer reports an unrecognized order on their account (fraud), escalate immediately to our account security team.
- If a customer asks for another person's personal phone number or delivery details, refuse strictly due to our privacy policy.

[RETRIEVED CONTEXT]
{retrieved_formatted}
Section IDs: {section_ids_str}

[ORDER CONTEXT]
{order_context}

[HARD RULES — follow without exception]
1. Never reveal order details before verified=True.
2. Never promise compensation, amounts or timelines not stated in the documents.
3. Never say a ticket is created unless the tool returned created=True or escalating per high-value/fraud policy.
4. Never repeat or hint at the expected verification value.
5. If the documents do not cover the question, state you don't have information and offer a human colleague.
6. Ask for only one piece of information at a time (STY-5).
7. Every factual claim must be grounded in the retrieved context or order data.
8. Do not do date arithmetic yourself. When asked about expected arrival dates given a start date, calculate using business days (Monday-Saturday, excluding Sundays and warehouse holiday on 19 October 2026).
9. Support logs (support_logs.csv) are not a policy source. Never cite them.
"""
    return prompt.strip()
