"""
Unified Test Runner for ZentroMart VoiceOps Agent.
Executes all regression and unit tests:
1. Business days & holiday calculations
2. Verification gate & PII withholding
3. Hybrid RAG retrieval & temporal version filtering
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_business_days import (
    test_holidays_loaded,
    test_add_business_days,
    test_refund_deadlines,
    test_business_day_checks
)
from tests.test_verification_gate import test_lookup_order_gate
from tests.test_retriever import test_retrieval

def run_all():
    print("========================================")
    print("RUNNING ALL ZENTROMART UNIT & LOGIC TESTS")
    print("========================================")

    print("\n--- 1. Testing Business Days & Holidays ---")
    test_holidays_loaded()
    test_add_business_days()
    test_refund_deadlines()
    test_business_day_checks()
    print("[PASS] All business days and holiday tests passed.")

    print("\n--- 2. Testing Identity Verification Gate ---")
    test_lookup_order_gate()
    print("[PASS] Identity gate successfully prevents premature data disclosure.")

    print("\n--- 3. Testing Hybrid RAG Retriever ---")
    test_retrieval()
    print("[PASS] Dense + sparse retrieval & temporal version filtering passed.")

    print("\n========================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY! (100% PASS)")
    print("========================================")

if __name__ == "__main__":
    run_all()
