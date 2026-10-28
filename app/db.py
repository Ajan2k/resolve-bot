"""Thin SQLite data-access layer. Every order query is scoped by customer_id."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .config import settings


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(r) for r in cur.fetchall()]


class Repository:
    def __init__(self, conn: Optional[sqlite3.Connection] = None) -> None:
        self.conn = conn or connect()

    # ---- customers -------------------------------------------------------
    def customer_exists(self, customer_id: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM customers WHERE customer_id = ?", (customer_id,))
        return cur.fetchone() is not None

    def get_customer(self, customer_id: str) -> Optional[dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ---- policy parameters -----------------------------------------------
    def policy_parameters(self) -> dict[str, str]:
        cur = self.conn.execute("SELECT key, value FROM policy_parameters")
        return {r["key"]: str(r["value"]) for r in cur.fetchall()}

    # ---- orders ----------------------------------------------------------
    def get_order(self, customer_id: str, order_id: str) -> Optional[dict[str, Any]]:
        """Returns the order only if it belongs to this customer (else None)."""
        cur = self.conn.execute(
            "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?",
            (order_id, customer_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_orders(self, customer_id: str, status: Optional[str] = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM orders WHERE customer_id = ?"
        params: list[Any] = [customer_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY order_date DESC, order_id DESC"
        return _rows(self.conn.execute(sql, params))

    def get_items(self, order_id: str) -> list[dict[str, Any]]:
        sql = (
            "SELECT oi.item_id, oi.order_id, oi.product_id, oi.qty, oi.unit_price, oi.opened, "
            "p.name AS product_name, p.category, p.price, p.returnable, p.warranty_class "
            "FROM order_items oi JOIN products p ON p.product_id = oi.product_id "
            "WHERE oi.order_id = ? ORDER BY oi.item_id"
        )
        return _rows(self.conn.execute(sql, (order_id,)))

    def get_shipment(self, order_id: str) -> Optional[dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM shipments WHERE order_id = ?", (order_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ---- writes ----------------------------------------------------------
    def insert_return_request(self, order_id: str, customer_id: str, item_ids: list[str], reason: str) -> dict[str, Any]:
        row = {
            "request_id": str(uuid.uuid4()),
            "order_id": order_id,
            "customer_id": customer_id,
            "item_ids": ",".join(item_ids),
            "reason": reason,
            "status": "requested",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.conn.execute(
            "INSERT INTO return_requests (request_id, order_id, customer_id, item_ids, reason, status, created_at) "
            "VALUES (:request_id, :order_id, :customer_id, :item_ids, :reason, :status, :created_at)",
            row,
        )
        self.conn.commit()
        return row

    def insert_support_ticket(self, order_id: str, customer_id: str, ticket_type: str, summary: str) -> dict[str, Any]:
        row = {
            "ticket_id": str(uuid.uuid4()),
            "order_id": order_id,
            "customer_id": customer_id,
            "ticket_type": ticket_type,
            "summary": summary,
            "status": "open",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.conn.execute(
            "INSERT INTO support_tickets (ticket_id, order_id, customer_id, ticket_type, summary, status, created_at) "
            "VALUES (:ticket_id, :order_id, :customer_id, :ticket_type, :summary, :status, :created_at)",
            row,
        )
        self.conn.commit()
        return row
