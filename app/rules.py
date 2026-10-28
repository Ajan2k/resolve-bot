"""Business rules (section 2.3 of the task). Pure Python; the LLM never decides.

All numbers come from the policy_parameters table. Nothing is hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

from dateutil.relativedelta import relativedelta

VALID_REASONS = ("changed_mind", "defective")
VALID_TICKET_TYPES = ("warranty", "cancellation")


class RuleError(Exception):
    """A business-rule failure with a stable error code."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"error_code": self.error_code, "message": self.message}


ORDER_NOT_FOUND = RuleError("ORDER_NOT_FOUND", "I couldn't find that order on your account.")


@dataclass(frozen=True)
class Params:
    return_window_days: int
    standard_warranty_months: int
    extended_warranty_months: int

    @classmethod
    def from_table(cls, raw: dict[str, str]) -> "Params":
        def need(key: str) -> int:
            if key not in raw:
                raise KeyError(f"policy_parameters is missing '{key}'")
            return int(float(raw[key]))

        return cls(
            return_window_days=need("return_window_days"),
            standard_warranty_months=need("standard_warranty_months"),
            extended_warranty_months=need("extended_warranty_months"),
        )


# ---- date helpers -----------------------------------------------------------

def parse_date(value: Any) -> Optional[date]:
    """Accepts 'YYYY-MM-DD', ISO datetimes, or None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return date.fromisoformat(text[:10])


def days_since(delivered: date, today: date) -> int:
    return (today - delivered).days


def warranty_months(warranty_class: str, params: Params) -> Optional[int]:
    wc = (warranty_class or "").strip().lower()
    if wc == "standard":
        return params.standard_warranty_months
    if wc == "extended":
        return params.extended_warranty_months
    return None  # "none" or unknown -> no warranty


def warranty_end(delivered: date, months: int) -> date:
    """Same calendar day N months later; falls back to month end (relativedelta does this)."""
    return delivered + relativedelta(months=months)


def warranty_info(item: dict[str, Any], delivered: Optional[date], today: date, params: Params) -> dict[str, Any]:
    months = warranty_months(item.get("warranty_class", ""), params)
    if months is None:
        return {"warranty_months": 0, "warranty_end_date": None, "warranty_active": False, "has_warranty": False}
    if delivered is None:
        return {"warranty_months": months, "warranty_end_date": None, "warranty_active": False, "has_warranty": True}
    end = warranty_end(delivered, months)
    return {
        "warranty_months": months,
        "warranty_end_date": end.isoformat(),
        "warranty_active": today <= end,  # the end date itself is still covered
        "has_warranty": True,
    }


# ---- return eligibility (2.3) ---------------------------------------------------

def check_return_eligibility(
    order: dict[str, Any],
    items: list[dict[str, Any]],
    delivered: Optional[date],
    requested_item_ids: list[str],
    reason: str,
    today: date,
    params: Params,
) -> dict[str, Any]:
    """Runs the four checks in order and stops at the first failure.

    `order` must already be verified to belong to the customer (the caller does that;
    an unknown/foreign order never reaches this function).
    Raises RuleError on failure; returns an "eligible" dict on success.
    """
    if reason not in VALID_REASONS:
        raise RuleError("INVALID_REASON", f"Reason must be one of {', '.join(VALID_REASONS)}.")
    if not requested_item_ids:
        raise RuleError("INVALID_ITEM", "No items were given.")

    # Check 1: items belong to the order, and the order has been delivered
    by_id = {it["item_id"]: it for it in items}
    for item_id in requested_item_ids:
        if item_id not in by_id:
            raise RuleError("INVALID_ITEM", f"Item {item_id} is not in this order")
    if str(order.get("status", "")).lower() != "delivered" or delivered is None:
        raise RuleError("NOT_DELIVERED", f"Order {order['order_id']} has not been delivered yet.")

    # Check 2: never-returnable products
    for item_id in requested_item_ids:
        it = by_id[item_id]
        if str(it.get("returnable", "")).lower() == "no":
            raise RuleError("NON_RETURNABLE", f"{it['product_name']} ({item_id}) cannot be returned.")

    # Check 3: return window
    elapsed = days_since(delivered, today)
    if elapsed > params.return_window_days:
        raise RuleError(
            "RETURN_WINDOW_EXPIRED",
            f"The {params.return_window_days}-day return window ended; it has been {elapsed} days since delivery.",
        )

    # Check 4: opened + changed_mind + unopened_only
    if reason == "changed_mind":
        for item_id in requested_item_ids:
            it = by_id[item_id]
            opened = int(it.get("opened") or 0) == 1
            if opened and str(it.get("returnable", "")).lower() == "unopened_only":
                raise RuleError(
                    "OPENED_NOT_RETURNABLE",
                    f"{it['product_name']} ({item_id}) has been opened and can only be returned unopened.",
                )

    return {
        "status": "eligible",
        "order_id": order["order_id"],
        "item_ids": list(requested_item_ids),
        "reason": reason,
        "days_since_delivery": elapsed,
        "return_window_days": params.return_window_days,
        "items": [
            {"item_id": i, "product_name": by_id[i]["product_name"], "opened": int(by_id[i].get("opened") or 0) == 1}
            for i in requested_item_ids
        ],
    }


# ---- ticket rules -----------------------------------------------------------------

def check_warranty_ticket(order: dict[str, Any], items: list[dict[str, Any]], delivered: Optional[date],
                          today: date, params: Params) -> dict[str, Any]:
    if str(order.get("status", "")).lower() != "delivered" or delivered is None:
        raise RuleError("NOT_DELIVERED", f"Order {order['order_id']} has not been delivered yet.")
    infos = [(it, warranty_info(it, delivered, today, params)) for it in items]
    if not any(w["has_warranty"] for _, w in infos):
        raise RuleError("NO_WARRANTY", "None of the items in this order has a warranty.")
    active = [(it, w) for it, w in infos if w["warranty_active"]]
    if not active:
        raise RuleError("WARRANTY_EXPIRED", "The warranty on every item in this order has expired.")
    return {
        "status": "ok",
        "covered_items": [
            {"item_id": it["item_id"], "product_name": it["product_name"], "warranty_end_date": w["warranty_end_date"]}
            for it, w in active
        ],
    }


def check_cancellation(order: dict[str, Any]) -> dict[str, Any]:
    status = str(order.get("status", "")).lower()
    if status == "processing":
        return {"status": "ok", "order_id": order["order_id"]}
    if status == "cancelled":
        raise RuleError("ORDER_ALREADY_SHIPPED", f"Order {order['order_id']} is already cancelled.")
    raise RuleError("ORDER_ALREADY_SHIPPED", f"Order {order['order_id']} has already shipped ({status}) and cannot be cancelled.")
