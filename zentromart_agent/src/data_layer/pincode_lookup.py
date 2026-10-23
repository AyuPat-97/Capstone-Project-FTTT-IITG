import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_layer.loader import load_csv

_PINCODES_CACHE = None

def _get_pincodes_df():
    global _PINCODES_CACHE
    if _PINCODES_CACHE is None:
        df = load_csv("pincodes.csv")
        _PINCODES_CACHE = df.set_index("pincode").to_dict(orient="index")
    return _PINCODES_CACHE

def lookup_pincode(pincode: str) -> dict:
    """
    Look up pincode details in pincodes.csv.
    Returns:
      {
        "serviceable": bool,
        "cod_available": bool,
        "express_available": bool,
        "city": str,
        "state": str
      }
      or {"error": "pincode not found"}
    """
    clean_pin = "".join(ch for ch in str(pincode) if ch.isdigit())
    table = _get_pincodes_df()
    if clean_pin in table:
        row = table[clean_pin]
        return {
            "pincode": clean_pin,
            "serviceable": row.get("serviceable") == "True",
            "cod_available": row.get("cod_available") == "True",
            "express_available": row.get("express_available") == "True",
            "city": row.get("city", ""),
            "state": row.get("state", ""),
        }
    return {"error": "pincode not found"}
