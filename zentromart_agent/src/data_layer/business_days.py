import csv
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure config can be imported
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DATA_DIR

def _load_holidays() -> set[date]:
    holidays_file = DATA_DIR / "holidays.csv"
    holidays = set()
    if holidays_file.exists():
        with open(holidays_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d_str = row.get("date", "").strip()
                if d_str:
                    try:
                        holidays.add(date.fromisoformat(d_str))
                    except ValueError:
                        pass
    return holidays

HOLIDAYS: set[date] = _load_holidays()

def is_business_day(d: date) -> bool:
    """Check if date is Monday-Saturday and not a public holiday."""
    return d.weekday() != 6 and d not in HOLIDAYS

def add_business_days(start: date, n: int) -> date:
    """Return the date that is n business days after start."""
    d = start
    while n > 0:
        d += timedelta(days=1)
        if is_business_day(d):
            n -= 1
    return d

def business_days_between(start: date, end: date) -> int:
    """Count business days from start (inclusive) to end (exclusive)."""
    count = 0
    d = start
    while d < end:
        if is_business_day(d):
            count += 1
        d += timedelta(days=1)
    return count

def refund_max_deadline(initiated_on: date, payment_method: str) -> date:
    """
    Latest date by which refund should arrive, per REF-2.
    payment_method values: upi, card, netbanking, wallet, cod_wallet, cod_bank
    wallet and cod_wallet = 0 business days (instant, within 2 hours)
    """
    MAX_BD = {
        "upi": 3,
        "card": 7,
        "netbanking": 5,
        "wallet": 0,
        "cod_wallet": 0,
        "cod_bank": 7
    }
    n = MAX_BD.get(payment_method.lower(), 7)
    return add_business_days(initiated_on, n)
