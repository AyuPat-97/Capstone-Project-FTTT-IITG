import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agent.tools_wrapper import lookup_order

def test_lookup_order_gate():
    oid = "61698151"
    real_pincode = "201301"
    real_phone_suffix = "5493"

    # 1. Calling without verification_value
    res1 = lookup_order(oid)
    assert res1.get("verification_required") is True, "Must require verification"
    assert "items" not in res1, "Must NOT leak order details"
    assert res1.get("found") is None or res1.get("found") is False
    print("[PASS] lookup_order without verification correctly gated.")

    # 2. Calling with wrong proof
    res_wrong = lookup_order(oid, "0000")
    assert res_wrong.get("verified") is False, "Wrong proof must return verified=False"
    assert "items" not in res_wrong, "Must not leak details on wrong proof"
    print("[PASS] lookup_order with wrong proof correctly rejected.")

    # 3. Calling with correct pincode
    res_pin = lookup_order(oid, real_pincode)
    assert res_pin.get("verified") is True, f"Valid pincode must verify: {res_pin}"
    assert "items" in res_pin, "Verified response must include items"
    assert "customer_first_name" in res_pin
    print("[PASS] lookup_order with valid pincode verified successfully!")

    # 4. Calling with correct phone suffix
    res_phone = lookup_order(oid, real_phone_suffix)
    assert res_phone.get("verified") is True, f"Valid phone suffix must verify: {res_phone}"
    assert "items" in res_phone
    print("[PASS] lookup_order with valid phone suffix verified successfully!")

    print("ALL VERIFICATION GATE TESTS PASSED!")

if __name__ == "__main__":
    test_lookup_order_gate()
