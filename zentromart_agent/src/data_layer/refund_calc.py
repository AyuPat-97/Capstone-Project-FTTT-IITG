from decimal import Decimal, ROUND_HALF_UP

def round_half_up(val: float) -> int:
    """Rounds half a rupee up: 2479.5 -> 2480."""
    d = Decimal(str(val))
    return int(d.quantize(Decimal('1'), rounding=ROUND_HALF_UP))

def calculate_partial_refund(
    subtotal: float,
    returned_items_total: float,
    coupon_discount: float = 0.0,
    coupon_min_order: float = 0.0,
    reverse_pickup_fee: float = 0.0
) -> int:
    """
    Implements REF-4 partial refund + coupon math:
    - If remaining order falls below coupon_min_order, the coupon discount on remaining
      items is deducted from the refund (i.e. full coupon discount is clawed back from returned value).
    - Otherwise, the refund is the price paid for that item after its proportional share
      of the coupon discount.
    - Reverse pickup fee is deducted (if applicable).
    - Half a rupee rounds up.
    """
    if subtotal <= 0 or returned_items_total <= 0:
        return 0

    remaining_order = subtotal - returned_items_total

    if coupon_discount > 0 and coupon_min_order > 0 and remaining_order < coupon_min_order:
        # Full coupon discount is revoked from refund
        raw_refund = returned_items_total - coupon_discount - reverse_pickup_fee
    elif coupon_discount > 0:
        # Proportional coupon discount
        item_discount_share = coupon_discount * (returned_items_total / subtotal)
        raw_refund = returned_items_total - item_discount_share - reverse_pickup_fee
    else:
        raw_refund = returned_items_total - reverse_pickup_fee

    return max(0, round_half_up(raw_refund))
