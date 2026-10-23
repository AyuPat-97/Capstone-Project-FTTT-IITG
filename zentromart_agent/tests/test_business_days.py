import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_layer.business_days import (
    is_business_day,
    add_business_days,
    business_days_between,
    refund_max_deadline,
    HOLIDAYS,
)

def test_holidays_loaded():
    assert date(2026, 8, 15) in HOLIDAYS
    assert date(2026, 10, 2) in HOLIDAYS
    assert date(2026, 10, 19) in HOLIDAYS
    assert date(2026, 11, 9) in HOLIDAYS
    print("[PASS] Holidays loaded successfully:", HOLIDAYS)

def test_add_business_days():
    # 2026-10-17 is Saturday
    # 18 is Sunday (skip)
    # 19 is Monday holiday (skip)
    # 20 is Tuesday (BD 1)
    # 21 is Wednesday (BD 2)
    # 22 is Thursday (BD 3)
    res = add_business_days(date(2026, 10, 17), 3)
    assert res == date(2026, 10, 22), f"Expected 2026-10-22, got {res}"
    print("[PASS] add_business_days(2026-10-17, 3) = 2026-10-22")

    # From 19 Oct (holiday), +1 business day is 20 Oct (Tuesday)
    res2 = add_business_days(date(2026, 10, 19), 1)
    assert res2 == date(2026, 10, 20), f"Expected 2026-10-20, got {res2}"
    print("[PASS] add_business_days(2026-10-19, 1) = 2026-10-20")

def test_refund_deadlines():
    # UPI is 3 business days
    # From 2026-10-16 (Friday):
    # Sat 17 (BD 1), Sun 18 (skip), Mon 19 (holiday skip), Tue 20 (BD 2), Wed 21 (BD 3)
    deadline_upi = refund_max_deadline(date(2026, 10, 16), "upi")
    assert deadline_upi == date(2026, 10, 21), f"Expected 2026-10-21, got {deadline_upi}"
    print("[PASS] refund_max_deadline(2026-10-16, 'upi') = 2026-10-21")

    # Wallet is 0 days (same day)
    deadline_wallet = refund_max_deadline(date(2026, 10, 20), "wallet")
    assert deadline_wallet == date(2026, 10, 20)
    print("[PASS] refund_max_deadline(2026-10-20, 'wallet') = 2026-10-20")

def test_business_day_checks():
    assert not is_business_day(date(2026, 10, 18))  # Sunday
    assert not is_business_day(date(2026, 10, 19))  # Holiday
    assert is_business_day(date(2026, 10, 20))      # Business day
    # Sat 17, Sun 18 (skip), Mon 19 (skip), Tue 20, Wed 21 -> 3 business days before 22
    assert business_days_between(date(2026, 10, 17), date(2026, 10, 22)) == 3
    print("[PASS] is_business_day and business_days_between verified.")

if __name__ == "__main__":
    test_holidays_loaded()
    test_add_business_days()
    test_refund_deadlines()
    test_business_day_checks()
    print("ALL BUSINESS DAYS TESTS PASSED!")
