"""The five tools the agent can call (section 2.2).

Every tool receives the customer_id from the HTTP header (never from the LLM) and
checks in Python that any order it touches belongs to that customer.
Tools never raise: they return {"error_code": ..., "message": ...} on failure.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from . import rules
from .config import settings
from .db import Repository
from .rag import PolicyIndex
from .rules import Params, RuleError

# JSON-schema style definitions handed to the LLM.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_orders",
        "description": (
            "Look up the logged-in customer's orders with their items, shipment status, delivery date, "
            "days since delivery, and per-item warranty end date / whether warranty is active. "
            "Call with order_id to fetch one order, or without it to list all orders (optionally filtered by status). "
            "Use this first whenever the customer mentions an order, an item, tracking, returns, warranty or cancellation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID such as SS-10458 (optional)."},
                "status": {
                    "type": "string",
                    "enum": ["processing", "shipped", "delivered", "cancelled"],
                    "description": "Filter by order status (optional).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search the store's current policy documents and return the best matching paragraphs, each with "
            "doc_id and section number. Use it for any question about rules (returns, refunds, warranty, shipping, "
            "cancellation, exchanges, payments) and before explaining why an action is or is not allowed. "
            "Cite only paragraphs you actually used."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Short natural-language search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "check_return_eligibility",
        "description": (
            "Check whether specific items of an order can be returned. Returns status 'eligible' or an error code "
            "(NOT_DELIVERED, NON_RETURNABLE, RETURN_WINDOW_EXPIRED, OPENED_NOT_RETURNABLE, INVALID_ITEM...). "
            "reason must be 'changed_mind' (does not want it) or 'defective' (broken, damaged or wrong item). "
            "Always call this before proposing a return."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "item_ids": {"type": "array", "items": {"type": "string"}, "description": "Item IDs like SS-10458-1."},
                "reason": {"type": "string", "enum": ["changed_mind", "defective"]},
            },
            "required": ["order_id", "item_ids", "reason"],
        },
    },
    {
        "name": "create_return_request",
        "description": (
            "Create (save) a return request. ONLY call this after the customer has explicitly confirmed a proposal "
            "with a clear yes. It re-checks eligibility before saving."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "item_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string", "enum": ["changed_mind", "defective"]},
            },
            "required": ["order_id", "item_ids", "reason"],
        },
    },
    {
        "name": "create_support_ticket",
        "description": (
            "Create (save) a support ticket of type 'warranty' (order delivered and at least one item under warranty) "
            "or 'cancellation' (order still processing). ONLY call this after the customer has explicitly confirmed "
            "a proposal with a clear yes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "ticket_type": {"type": "string", "enum": ["warranty", "cancellation"]},
                "summary": {"type": "string", "description": "One-sentence summary of the problem or request."},
            },
            "required": ["order_id", "ticket_type", "summary"],
        },
    },
]

WRITE_TOOLS = {"create_return_request", "create_support_ticket"}


class ToolBox:
    """Tools bound to one customer (from the header) and one request."""

    def __init__(self, repo: Repository, index: PolicyIndex, customer_id: str) -> None:
        self.repo = repo
        self.index = index
        self.customer_id = customer_id
        self.today = settings.app_today
        self.params: Params = Params.from_table(repo.policy_parameters())
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "get_orders": self.get_orders,
            "search_policy": self.search_policy,
            "check_return_eligibility": self.check_return_eligibility,
            "create_return_request": self.create_return_request,
            "create_support_ticket": self.create_support_ticket,
        }

    # ---- dispatch ----------------------------------------------------------
    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return {"error_code": "UNKNOWN_TOOL", "message": f"Unknown tool {name}"}
        try:
            return handler(**self._clean_args(name, args))
        except RuleError as e:
            return e.to_dict()
        except TypeError as e:  # wrong/missing arguments from the LLM
            return {"error_code": "INVALID_ARGUMENTS", "message": str(e)}
        except Exception as e:  # never crash the request
            return {"error_code": "TOOL_ERROR", "message": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _clean_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = next(s for s in TOOL_SPECS if s["name"] == name)
        allowed = spec["parameters"]["properties"].keys()
        out = {k: v for k, v in (args or {}).items() if k in allowed}
        if "item_ids" in out and isinstance(out["item_ids"], str):
            out["item_ids"] = [s.strip() for s in out["item_ids"].split(",") if s.strip()]
        if "order_id" in out and out["order_id"] is not None:
            out["order_id"] = str(out["order_id"]).strip().upper()
        return out

    # ---- helpers -------------------------------------------------------------
    def _load_order(self, order_id: Optional[str]) -> tuple[dict[str, Any], list[dict[str, Any]], Optional[Any]]:
        """Returns (order, items, delivered_date). Raises ORDER_NOT_FOUND for unknown/foreign orders."""
        if not order_id:
            raise rules.ORDER_NOT_FOUND
        order = self.repo.get_order(self.customer_id, order_id)
        if order is None:
            raise rules.ORDER_NOT_FOUND
        items = self.repo.get_items(order_id)
        shipment = self.repo.get_shipment(order_id)
        delivered = rules.parse_date(shipment.get("delivered_at")) if shipment else None
        return order, items, delivered

    def _enrich_order(self, order: dict[str, Any]) -> dict[str, Any]:
        items = self.repo.get_items(order["order_id"])
        shipment = self.repo.get_shipment(order["order_id"]) or {}
        delivered = rules.parse_date(shipment.get("delivered_at"))
        out = {
            "order_id": order["order_id"],
            "order_date": order.get("order_date"),
            "status": order.get("status"),
            "total": order.get("total"),
            "shipment": {
                "carrier": shipment.get("carrier"),
                "status": shipment.get("status"),
                "eta": shipment.get("eta"),
                "delivered_at": shipment.get("delivered_at"),
            },
            "delivery_date": delivered.isoformat() if delivered else None,
            "days_since_delivery": rules.days_since(delivered, self.today) if delivered else None,
            "return_window_days": self.params.return_window_days,
            "within_return_window": (
                rules.days_since(delivered, self.today) <= self.params.return_window_days if delivered else False
            ),
            "items": [],
        }
        for it in items:
            w = rules.warranty_info(it, delivered, self.today, self.params)
            out["items"].append(
                {
                    "item_id": it["item_id"],
                    "product_id": it["product_id"],
                    "product_name": it["product_name"],
                    "category": it.get("category"),
                    "qty": it.get("qty"),
                    "unit_price": it.get("unit_price"),
                    "opened": int(it.get("opened") or 0) == 1,
                    "returnable": it.get("returnable"),
                    "warranty_class": it.get("warranty_class"),
                    **w,
                }
            )
        return out

    # ---- tools -----------------------------------------------------------------
    def get_orders(self, order_id: Optional[str] = None, status: Optional[str] = None) -> dict[str, Any]:
        if order_id:
            order = self.repo.get_order(self.customer_id, order_id)
            if order is None:
                raise rules.ORDER_NOT_FOUND
            return {"today": self.today.isoformat(), "orders": [self._enrich_order(order)]}
        orders = self.repo.list_orders(self.customer_id, status)
        return {"today": self.today.isoformat(), "orders": [self._enrich_order(o) for o in orders]}

    def search_policy(self, query: str) -> dict[str, Any]:
        return self.index.search(query)

    def check_return_eligibility(self, order_id: str, item_ids: list[str], reason: str) -> dict[str, Any]:
        order, items, delivered = self._load_order(order_id)
        return rules.check_return_eligibility(order, items, delivered, list(item_ids), reason, self.today, self.params)

    def create_return_request(self, order_id: str, item_ids: list[str], reason: str) -> dict[str, Any]:
        order, items, delivered = self._load_order(order_id)
        # Re-check just before saving.
        rules.check_return_eligibility(order, items, delivered, list(item_ids), reason, self.today, self.params)
        row = self.repo.insert_return_request(order_id, self.customer_id, list(item_ids), reason)
        return {"status": "created", "return_request": row}

    def create_support_ticket(self, order_id: str, ticket_type: str, summary: str = "") -> dict[str, Any]:
        if ticket_type not in rules.VALID_TICKET_TYPES:
            raise RuleError("INVALID_TICKET_TYPE", f"ticket_type must be one of {', '.join(rules.VALID_TICKET_TYPES)}.")
        order, items, delivered = self._load_order(order_id)
        if ticket_type == "warranty":
            rules.check_warranty_ticket(order, items, delivered, self.today, self.params)
        else:
            rules.check_cancellation(order)
        row = self.repo.insert_support_ticket(order_id, self.customer_id, ticket_type, summary or f"{ticket_type} request")
        return {"status": "created", "support_ticket": row}
