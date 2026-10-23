import sys
from datetime import date
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import TODAY, TOOLS_DIR, DATASET_DIR
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    from tools import (  # type: ignore[import-untyped, import-not-found]
        lookup_order as _raw_lookup_order,
        create_ticket as _raw_create_ticket,
    )
except (ImportError, ModuleNotFoundError):
    try:
        from Capstone_Project_Dataset_FTTT_2026.tools.tools import (  # type: ignore[import-untyped, import-not-found]
            lookup_order as _raw_lookup_order,
            create_ticket as _raw_create_ticket,
        )
    except (ImportError, ModuleNotFoundError):
        import importlib.util
        tools_file = TOOLS_DIR / "tools.py"
        spec = importlib.util.spec_from_file_location("tools_module", str(tools_file))
        tools_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tools_mod)
        _raw_lookup_order = tools_mod.lookup_order
        _raw_create_ticket = tools_mod.create_ticket
from src.data_layer.business_days import add_business_days
from src.data_layer.pincode_lookup import lookup_pincode as _lookup_pincode
from src.data_layer.product_lookup import lookup_product as _lookup_product

def lookup_order(order_id: str, verification_value: Optional[str] = None) -> dict:
    """
    Lookup a synthetic order, disclosing details ONLY after identity verification.
    order_id: 8-digit order ID (digits only).
    verification_value: Caller-provided 4-digit phone suffix OR 6-digit delivery pincode.
    """
    clean_oid = "".join(ch for ch in str(order_id) if ch.isdigit())
    clean_proof = "".join(ch for ch in str(verification_value) if ch.isdigit()) if verification_value else None
    return _raw_lookup_order(clean_oid, clean_proof)

def create_ticket(
    summary: str,
    order_id: Optional[str] = None,
    reason: Optional[str] = None,
    intent: Optional[str] = None
) -> dict:
    """
    Log a synthetic human-handover ticket for senior agent review.
    summary: Minimum 20 characters detailing issue, checks, and reason.
    order_id: Optional 8-digit order ID.
    reason: Escalation reason (e.g. high_value_damage, repeat_contact, legal_threat).
    intent: Customer inquiry intent.
    """
    clean_oid = "".join(ch for ch in str(order_id) if ch.isdigit()) if order_id else None
    return _raw_create_ticket(summary=summary, order_id=clean_oid, reason=reason, intent=intent)

def lookup_pincode(pincode: str) -> dict:
    """
    Check if a pincode is serviceable, supports cash on delivery (COD), or express delivery.
    pincode: 6-digit Indian postal code.
    """
    return _lookup_pincode(pincode)

def lookup_product(sku: str) -> dict:
    """
    Look up product specifications, return category, price, seller, assured status, and warranty.
    sku: Product SKU identifier (e.g. ELE-0012, FSH-0026).
    """
    return _lookup_product(sku)

def business_days_from_today(start_date: str, n_days: int) -> str:
    """
    Calculate the calendar date that is n business days after start_date.
    start_date: YYYY-MM-DD format (or 'today' for 2026-10-20).
    n_days: Number of business days to add (Mon-Sat, excluding holidays).
    """
    if str(start_date).lower() == "today" or not start_date:
        d = TODAY
    else:
        try:
            d = date.fromisoformat(str(start_date)[:10])
        except Exception:
            d = TODAY
    res = add_business_days(d, int(n_days))
    return res.isoformat()

# Mapping of tool names to functions for direct execution
TOOL_FUNCTIONS = {
    "lookup_order": lookup_order,
    "create_ticket": create_ticket,
    "lookup_pincode": lookup_pincode,
    "lookup_product": lookup_product,
    "business_days_from_today": business_days_from_today,
}

ALL_TOOLS = [
    lookup_order,
    create_ticket,
    lookup_pincode,
    lookup_product,
    business_days_from_today,
]
