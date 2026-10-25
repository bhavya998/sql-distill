"""The sales database the model learns to query.

Same schema and deterministic seed (``random.seed(42)``) as
[analytics-mcp](https://github.com/bhavya998/analytics-mcp), so a model trained here can sit
behind that server's tools unchanged.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_DB = Path("data/sales.db")

SCHEMA_SQL = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    company TEXT,
    region TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'standard',
    signup_date TEXT NOT NULL,
    lifetime_value REAL DEFAULT 0
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    cost REAL NOT NULL,
    stock INTEGER DEFAULT 0
);
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    region TEXT NOT NULL,
    hire_date TEXT NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    employee_id INTEGER REFERENCES employees(id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed',
    total REAL NOT NULL
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
"""

# The schema as the model sees it in every prompt: DDL plus the value sets it cannot guess.
SCHEMA_PROMPT = SCHEMA_SQL.strip() + """
-- customers.region, employees.region: 'North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East'
-- customers.tier: 'standard', 'premium', 'enterprise'
-- products.category: 'Electronics', 'Software', 'Services', 'Hardware', 'Consulting'
-- orders.status: 'completed', 'pending', 'cancelled', 'refunded'
-- employees.role: 'Sales Rep', 'Sales Manager', 'Account Executive'
-- dates are ISO text 'YYYY-MM-DD'; orders span 2024-01-01 to 2025-06-30
-- order_items.unit_price is the price paid; products.cost is unit cost; revenue = quantity * unit_price
"""

REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
TIERS = ["standard", "premium", "enterprise"]
CATEGORIES = ["Electronics", "Software", "Services", "Hardware", "Consulting"]
STATUSES = ["completed", "pending", "cancelled", "refunded"]
ROLES = ["Sales Rep", "Sales Manager", "Account Executive"]

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Aisha", "Wei", "Hiroshi", "Priya",
    "Omar", "Yuki", "Diego", "Fatima", "Chen", "Anya",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Kumar", "Tanaka", "Kim", "Singh", "Patel", "Nguyen", "Chang", "Okafor",
]
COMPANY_PREFIXES = ["Tech", "Data", "Cloud", "Cyber", "Quantum", "Nexus", "Vertex", "Apex",
                    "Prime", "Meta"]
COMPANY_SUFFIXES = ["Corp", "Solutions", "Systems", "Labs", "Group", "Industries", "Partners",
                    "Holdings"]
PRODUCT_NAMES = {
    "Electronics": ["Pro Laptop 15", "Ultra Tablet", "4K Monitor", "Wireless Headset", "Smart Hub"],
    "Software": ["Enterprise Suite", "Security Platform", "Analytics Dashboard", "CRM Pro",
                 "DevOps Toolkit"],
    "Services": ["Premium Support", "Implementation Package", "Training Program", "Health Check",
                 "Migration Service"],
    "Hardware": ["Rack Server X1", "Network Switch 48G", "Storage Array 100TB", "Backup Appliance",
                 "Edge Gateway"],
    "Consulting": ["Strategy Session", "Architecture Review", "Security Audit",
                   "Performance Tuning", "Compliance Check"],
}


def build_database(path: Path | str = DEFAULT_DB, overwrite: bool = False) -> Path:
    """Create the database with the deterministic seed. Idempotent unless ``overwrite``."""
    path = Path(path)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA_SQL)
        _seed(conn.cursor())
        conn.commit()
    finally:
        conn.close()
    return path


def _seed(cur: sqlite3.Cursor) -> None:
    random.seed(42)
    employees = []
    for i, region in enumerate(REGIONS):
        for j in range(3):
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            role = random.choice(ROLES)
            hired = (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))).strftime(
                "%Y-%m-%d")
            employees.append((i * 3 + j + 1, name, role, region, hired))
    cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees)

    customers = []
    for i in range(200):
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        email = (f"{name.lower().replace(' ', '.')}@{random.choice(COMPANY_PREFIXES).lower()}"
                 f"{random.choice(COMPANY_SUFFIXES).lower()}.com")
        company = f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"
        region = random.choice(REGIONS)
        tier = random.choices(TIERS, weights=[60, 30, 10])[0]
        signup = (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))).strftime(
            "%Y-%m-%d")
        customers.append((i + 1, name, email, company, region, tier, signup, 0.0))
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?)", customers)

    products = []
    pid = 1
    for category, names in PRODUCT_NAMES.items():
        for pname in names:
            price = round(random.uniform(99, 9999), 2)
            cost = round(price * random.uniform(0.3, 0.6), 2)
            products.append((pid, pname, category, price, cost, random.randint(10, 500)))
            pid += 1
    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", products)

    orders, items, item_id = [], [], 1
    for order_id in range(1, 2001):
        customer_id = random.randint(1, 200)
        employee_id = random.randint(1, len(employees))
        order_date = (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 546))).strftime(
            "%Y-%m-%d")
        status = random.choices(STATUSES, weights=[80, 10, 5, 5])[0]
        total = 0.0
        for prod in random.sample(products, random.randint(1, 5)):
            qty = random.randint(1, 20)
            total += qty * prod[3]
            items.append((item_id, order_id, prod[0], qty, prod[3]))
            item_id += 1
        orders.append((order_id, customer_id, employee_id, order_date, status, round(total, 2)))
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", orders)
    cur.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
    cur.execute(
        "UPDATE customers SET lifetime_value = (SELECT COALESCE(SUM(o.total), 0) FROM orders o "
        "WHERE o.customer_id = customers.id AND o.status = 'completed')"
    )
