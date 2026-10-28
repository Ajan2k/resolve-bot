"""Creates a sample data/shopsphere.db with the schema from the assessment.

The graders replace data/ with their own copy; this file exists only so the project
runs locally.  Usage:  python scripts/seed_db.py [path]
"""
from __future__ import annotations

import os
import sqlite3
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DB_PATH", "./data/shopsphere.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (customer_id TEXT PRIMARY KEY, full_name TEXT, email TEXT, city TEXT);
CREATE TABLE IF NOT EXISTS products (product_id TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL,
    returnable TEXT, warranty_class TEXT);
CREATE TABLE IF NOT EXISTS orders (order_id TEXT PRIMARY KEY, customer_id TEXT, order_date TEXT, status TEXT, total REAL);
CREATE TABLE IF NOT EXISTS order_items (item_id TEXT PRIMARY KEY, order_id TEXT, product_id TEXT, qty INTEGER,
    unit_price REAL, opened INTEGER);
CREATE TABLE IF NOT EXISTS shipments (order_id TEXT PRIMARY KEY, carrier TEXT, status TEXT, delivered_at TEXT, eta TEXT);
CREATE TABLE IF NOT EXISTS policy_parameters (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS return_requests (request_id TEXT PRIMARY KEY, order_id TEXT, customer_id TEXT,
    item_ids TEXT, reason TEXT, status TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS support_tickets (ticket_id TEXT PRIMARY KEY, order_id TEXT, customer_id TEXT,
    ticket_type TEXT, summary TEXT, status TEXT, created_at TEXT);
"""

customers = [
    ("C1009", "Priya Raman", "priya@example.com", "Chennai"),
    ("C1010", "Arjun Mehta", "arjun@example.com", "Mumbai"),
]
products = [
    ("P-SPK", "Nimbra Smart Speaker", "speakers", 4999, "yes", "standard"),
    ("P-BUD", "AeroPods Wireless Earbuds", "earbuds", 2999, "unopened_only", "standard"),
    ("P-GC", "ShopSphere Gift Card 1000", "gift cards", 1000, "no", "none"),
    ("P-LAP", "Vectra 14 Laptop", "laptops", 58999, "yes", "extended"),
    ("P-PHN", "Nimbra X2 Phone", "phones", 21999, "yes", "standard"),
    ("P-CHG", "65W USB-C Charger", "chargers", 1499, "yes", "standard"),
    ("P-PB", "10000mAh Power Bank", "power banks", 1799, "yes", "standard"),
]
# APP_TODAY default is 2026-07-01
orders = [
    # order_id, customer, order_date, status, total
    ("SS-10458", "C1009", "2026-06-18", "delivered", 7998),   # delivered 21 June -> day 10
    ("SS-10401", "C1009", "2026-05-25", "delivered", 2999),   # delivered 1 June  -> day 30 (last day)
    ("SS-10377", "C1009", "2026-05-20", "delivered", 1499),   # delivered 31 May  -> day 31 (expired)
    ("SS-10490", "C1009", "2026-06-26", "delivered", 1000),   # gift card, day 2
    ("SS-10512", "C1009", "2026-06-28", "shipped", 58999),    # in transit (laptop)
    ("SS-10530", "C1009", "2026-06-30", "processing", 1499),  # can be cancelled (charger)
    ("SS-10210", "C1009", "2025-03-10", "delivered", 21999),  # phone delivered 15 Mar 2025 -> warranty ended 15 Mar 2026
    ("SS-10300", "C1010", "2026-06-10", "delivered", 1799),   # belongs to another customer
    ("SS-10395", "C1010", "2026-06-20", "processing", 1499),
]
items = [
    ("SS-10458-1", "SS-10458", "P-SPK", 1, 4999, 0),
    ("SS-10458-2", "SS-10458", "P-BUD", 1, 2999, 1),
    ("SS-10401-1", "SS-10401", "P-BUD", 1, 2999, 1),
    ("SS-10377-1", "SS-10377", "P-CHG", 1, 1499, 0),
    ("SS-10490-1", "SS-10490", "P-GC", 1, 1000, 0),
    ("SS-10512-1", "SS-10512", "P-LAP", 1, 58999, 0),
    ("SS-10530-1", "SS-10530", "P-CHG", 1, 1499, 0),
    ("SS-10210-1", "SS-10210", "P-PHN", 1, 21999, 1),
    ("SS-10210-2", "SS-10210", "P-CHG", 1, 1499, 1),
    ("SS-10300-1", "SS-10300", "P-PB", 1, 1799, 0),
    ("SS-10395-1", "SS-10395", "P-CHG", 1, 1499, 0),
]
shipments = [
    ("SS-10458", "BlueDart", "delivered", "2026-06-21", "2026-06-21"),
    ("SS-10401", "Delhivery", "delivered", "2026-06-01", "2026-05-30"),
    ("SS-10377", "Delhivery", "delivered", "2026-05-31", "2026-05-31"),
    ("SS-10490", "BlueDart", "delivered", "2026-06-29", "2026-06-29"),
    ("SS-10512", "BlueDart", "in_transit", None, "2026-07-03"),
    ("SS-10530", "Delhivery", "processing", None, "2026-07-05"),
    ("SS-10210", "BlueDart", "delivered", "2025-03-15", "2025-03-15"),
    ("SS-10300", "Delhivery", "delivered", "2026-06-14", "2026-06-14"),
    ("SS-10395", "Delhivery", "processing", None, "2026-06-25"),
]
params = [
    ("return_window_days", "30"),
    ("standard_warranty_months", "12"),
    ("extended_warranty_months", "24"),
    ("damage_report_hours", "48"),
]


def main() -> None:
    os.makedirs(os.path.dirname(PATH) or ".", exist_ok=True)
    if os.path.exists(PATH):
        os.remove(PATH)
    conn = sqlite3.connect(PATH)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", products)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?)", items)
    conn.executemany("INSERT INTO shipments VALUES (?,?,?,?,?)", shipments)
    conn.executemany("INSERT INTO policy_parameters VALUES (?,?)", params)
    conn.commit()
    conn.close()
    print(f"Seeded {PATH}")


if __name__ == "__main__":
    main()
