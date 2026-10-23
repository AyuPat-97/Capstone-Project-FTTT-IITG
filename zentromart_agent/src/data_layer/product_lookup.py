import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_layer.loader import load_csv

_PRODUCTS_CACHE = None

def _get_products():
    global _PRODUCTS_CACHE
    if _PRODUCTS_CACHE is None:
        df = load_csv("products.csv")
        _PRODUCTS_CACHE = df.set_index("sku").to_dict(orient="index")
    return _PRODUCTS_CACHE

def lookup_product(sku: str) -> dict:
    """
    Look up product specifications by SKU in products.csv.
    Returns:
      {
        "sku": str,
        "title": str,
        "brand": str,
        "return_category": str,
        "price_inr": int,
        "fulfilled_by": str,
        "assured": bool,
        "warranty_months": int,
        "description": str
      }
      or {"error": "SKU not found"}
    """
    clean_sku = str(sku).strip().upper()
    products = _get_products()
    if clean_sku in products:
        row = products[clean_sku]
        price_val = 0
        if row.get("price_inr") and row.get("price_inr").isdigit():
            price_val = int(row.get("price_inr"))
        warranty_val = 0
        if row.get("warranty_months") and row.get("warranty_months").isdigit():
            warranty_val = int(row.get("warranty_months"))

        return {
            "sku": clean_sku,
            "title": row.get("title", ""),
            "brand": row.get("brand", ""),
            "return_category": row.get("return_category", ""),
            "price_inr": price_val,
            "fulfilled_by": row.get("fulfilled_by", ""),
            "assured": row.get("assured") == "True",
            "warranty_months": warranty_val,
            "description": row.get("description", "")
        }
    return {"error": "SKU not found"}
