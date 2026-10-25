"""Question/SQL templates over the sales schema, grouped by the SQL skill they exercise.

``TRAIN_TEMPLATES`` feed the training set (and an in-distribution test split with unseen
parameter values). ``HELDOUT_TEMPLATES`` never appear in training: they combine the same
skills in new ways and measure whether the model learned SQL or memorised templates.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sql_distill.db import CATEGORIES, PRODUCT_NAMES, REGIONS, ROLES, STATUSES, TIERS

PRODUCTS = [p for names in PRODUCT_NAMES.values() for p in names]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
QUARTER_BOUNDS = {1: ("01-01", "03-31"), 2: ("04-01", "06-30"), 3: ("07-01", "09-30"),
                  4: ("10-01", "12-31")}


@dataclass(frozen=True)
class Template:
    id: str
    skill: str
    phrasings: tuple[str, ...]
    sql: str
    params: Callable[[random.Random], dict[str, Any]]

    def render(self, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
        p = self.params(rng)
        return rng.choice(self.phrasings).format(**p), self.sql.format(**p), p


def _none(_: random.Random) -> dict[str, Any]:
    return {}


def _region(r: random.Random) -> dict[str, Any]:
    return {"region": r.choice(REGIONS)}


def _tier(r: random.Random) -> dict[str, Any]:
    return {"tier": r.choice(TIERS)}


def _category(r: random.Random) -> dict[str, Any]:
    return {"category": r.choice(CATEGORIES)}


def _status(r: random.Random) -> dict[str, Any]:
    return {"status": r.choice(STATUSES)}


def _year(r: random.Random) -> dict[str, Any]:
    return {"year": r.choice([2024, 2025])}


def _month(r: random.Random) -> dict[str, Any]:
    year = r.choice([2024, 2025])
    month = r.randint(1, 12 if year == 2024 else 6)
    return {"year": year, "mm": f"{month:02d}", "month_name": MONTHS[month - 1]}


def _quarter(r: random.Random) -> dict[str, Any]:
    year = r.choice([2024, 2025])
    q = r.randint(1, 4 if year == 2024 else 2)
    start, end = QUARTER_BOUNDS[q]
    return {"year": year, "q": q, "start": f"{year}-{start}", "end": f"{year}-{end}"}


def _k(r: random.Random) -> dict[str, Any]:
    return {"k": r.choice([3, 5, 10])}


def _merge(*fns: Callable[[random.Random], dict[str, Any]]):
    def params(r: random.Random) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for fn in fns:
            out.update(fn(r))
        return out
    return params


def _date_range(r: random.Random) -> dict[str, Any]:
    start = r.choice(["2024-02-01", "2024-05-15", "2024-09-01", "2025-01-10", "2025-03-01"])
    end = {"2024-02-01": "2024-04-30", "2024-05-15": "2024-08-31", "2024-09-01": "2024-12-31",
           "2025-01-10": "2025-02-28", "2025-03-01": "2025-06-30"}[start]
    return {"d1": start, "d2": end}


T = Template

TRAIN_TEMPLATES: list[Template] = [
    # --- filter + count -------------------------------------------------------------------
    T("cust_in_region", "filter_count",
      ("How many customers are in {region}?", "Count the customers based in {region}.",
       "What's the number of customers from {region}?"),
      "SELECT COUNT(*) FROM customers WHERE region = '{region}'", _region),
    T("cust_in_tier", "filter_count",
      ("How many {tier} tier customers do we have?", "Count customers on the {tier} tier.",
       "Number of {tier} customers?"),
      "SELECT COUNT(*) FROM customers WHERE tier = '{tier}'", _tier),
    T("orders_by_status", "filter_count",
      ("How many orders are {status}?", "Count the {status} orders.",
       "What is the number of orders with status {status}?"),
      "SELECT COUNT(*) FROM orders WHERE status = '{status}'", _status),
    T("orders_in_month", "date",
      ("How many orders were placed in {month_name} {year}?",
       "Count the orders from {month_name} {year}.",
       "Number of orders in {month_name} {year}?"),
      "SELECT COUNT(*) FROM orders WHERE strftime('%Y-%m', order_date) = '{year}-{mm}'", _month),
    T("low_stock", "filter_count",
      ("How many {category} products have fewer than {n} units in stock?",
       "Count {category} products with stock below {n}."),
      "SELECT COUNT(*) FROM products WHERE category = '{category}' AND stock < {n}",
      _merge(_category, lambda r: {"n": r.choice([100, 200, 300])})),
    # --- filter + list --------------------------------------------------------------------
    T("products_above_price", "filter_list",
      ("List the {category} products priced above {price}.",
       "Which {category} products cost more than {price}?"),
      "SELECT name FROM products WHERE category = '{category}' AND price > {price}",
      _merge(_category, lambda r: {"price": r.choice([1000, 2500, 5000])})),
    T("customers_signed_up", "filter_list",
      ("Which {tier} customers signed up in {year}? Give their names and emails.",
       "List names and emails of {tier} customers who joined in {year}."),
      "SELECT name, email FROM customers WHERE tier = '{tier}' "
      "AND strftime('%Y', signup_date) = '{year}'",
      _merge(_tier, lambda r: {"year": r.choice([2023, 2024])})),
    T("employees_in_region", "filter_list",
      ("List the employees in {region} with their roles.",
       "Who works in {region}? Show names and roles."),
      "SELECT name, role FROM employees WHERE region = '{region}'", _region),
    T("employees_by_role", "filter_list",
      ("Which employees are {role}s? Give names and regions.",
       "List every {role} and their region."),
      "SELECT name, region FROM employees WHERE role = '{role}'",
      lambda r: {"role": r.choice(ROLES)}),
    # --- aggregates -----------------------------------------------------------------------
    T("revenue_year", "aggregate",
      ("What was the total revenue from completed orders in {year}?",
       "Sum of completed order totals in {year}?",
       "How much revenue did completed orders bring in during {year}?"),
      "SELECT ROUND(SUM(total), 2) FROM orders WHERE status = 'completed' "
      "AND strftime('%Y', order_date) = '{year}'", _year),
    T("avg_order_value_status", "aggregate",
      ("What is the average order value of {status} orders?",
       "Average total for orders with status {status}?"),
      "SELECT ROUND(AVG(total), 2) FROM orders WHERE status = '{status}'", _status),
    T("most_expensive_in_category", "aggregate",
      ("What is the most expensive product in {category}?",
       "Which {category} product has the highest price?"),
      "SELECT name FROM products WHERE category = '{category}' ORDER BY price DESC LIMIT 1",
      _category),
    T("avg_ltv_tier", "aggregate",
      ("What is the average lifetime value of {tier} customers?",
       "Average customer lifetime value on the {tier} tier?"),
      "SELECT ROUND(AVG(lifetime_value), 2) FROM customers WHERE tier = '{tier}'", _tier),
    T("max_order_total", "aggregate",
      ("What is the largest single order total ever recorded?",
       "Show the highest order total in the database."),
      "SELECT MAX(total) FROM orders", _none),
    # --- group by -------------------------------------------------------------------------
    T("customers_per_region", "group_by",
      ("How many customers are there per region?", "Show the customer count for each region."),
      "SELECT region, COUNT(*) FROM customers GROUP BY region", _none),
    T("orders_per_status", "group_by",
      ("How many orders are in each status?", "Break down order counts by status."),
      "SELECT status, COUNT(*) FROM orders GROUP BY status", _none),
    T("avg_price_per_category", "group_by",
      ("What is the average product price in each category?",
       "Average price per product category?"),
      "SELECT category, ROUND(AVG(price), 2) FROM products GROUP BY category", _none),
    T("monthly_revenue", "group_by",
      ("Show completed revenue per month in {year}, in month order.",
       "Monthly revenue from completed orders for {year}, sorted by month."),
      "SELECT strftime('%Y-%m', order_date) AS month, ROUND(SUM(total), 2) FROM orders "
      "WHERE status = 'completed' AND strftime('%Y', order_date) = '{year}' "
      "GROUP BY month ORDER BY month", _year),
    T("stock_per_category", "group_by",
      ("How many products and how much total stock does each category have?",
       "Per category, count the products and sum their stock."),
      "SELECT category, COUNT(*), SUM(stock) FROM products GROUP BY category", _none),
    # --- joins ----------------------------------------------------------------------------
    T("revenue_by_customer_region", "join",
      ("What was the completed revenue from customers in {region}?",
       "Total of completed orders placed by {region} customers?"),
      "SELECT ROUND(SUM(o.total), 2) FROM orders o JOIN customers c ON c.id = o.customer_id "
      "WHERE o.status = 'completed' AND c.region = '{region}'", _region),
    T("orders_by_tier_year", "join",
      ("How many orders did {tier} customers place in {year}?",
       "Count {year} orders from {tier} tier customers."),
      "SELECT COUNT(*) FROM orders o JOIN customers c ON c.id = o.customer_id "
      "WHERE c.tier = '{tier}' AND strftime('%Y', o.order_date) = '{year}'",
      _merge(_tier, _year)),
    T("top_employees_region", "join",
      ("Which {k} employees in {region} handled the most orders? Show order counts.",
       "Top {k} {region} employees by number of orders handled."),
      "SELECT e.name, COUNT(*) AS n FROM orders o JOIN employees e ON e.id = o.employee_id "
      "WHERE e.region = '{region}' GROUP BY e.id ORDER BY n DESC, e.id LIMIT {k}",
      _merge(_region, lambda r: {"k": r.choice([2, 3])})),
    T("units_of_product", "multi_join",
      ("How many units of {product} were sold in completed orders?",
       "Total quantity of {product} across completed orders?"),
      "SELECT SUM(oi.quantity) FROM order_items oi JOIN orders o ON o.id = oi.order_id "
      "JOIN products p ON p.id = oi.product_id "
      "WHERE p.name = '{product}' AND o.status = 'completed'",
      lambda r: {"product": r.choice(PRODUCTS)}),
    T("revenue_by_category_year", "multi_join",
      ("Show completed revenue by product category for {year}.",
       "Per category, how much revenue came from completed orders in {year}?"),
      "SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) FROM order_items oi "
      "JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id "
      "WHERE o.status = 'completed' AND strftime('%Y', o.order_date) = '{year}' "
      "GROUP BY p.category", _year),
    T("top_products_region", "multi_join",
      ("What are the top {k} products by units sold to {region} customers?",
       "Top {k} products by quantity bought by customers in {region}."),
      "SELECT p.name, SUM(oi.quantity) AS units FROM order_items oi "
      "JOIN orders o ON o.id = oi.order_id JOIN customers c ON c.id = o.customer_id "
      "JOIN products p ON p.id = oi.product_id WHERE c.region = '{region}' "
      "GROUP BY p.id ORDER BY units DESC, p.id LIMIT {k}", _merge(_region, _k)),
    T("category_products_by_tier", "multi_join",
      ("Which {category} products have {tier} customers bought? List distinct names.",
       "Distinct {category} products purchased by {tier} tier customers."),
      "SELECT DISTINCT p.name FROM order_items oi JOIN orders o ON o.id = oi.order_id "
      "JOIN customers c ON c.id = o.customer_id JOIN products p ON p.id = oi.product_id "
      "WHERE p.category = '{category}' AND c.tier = '{tier}'", _merge(_category, _tier)),
    # --- top-k ----------------------------------------------------------------------------
    T("top_customers_orders", "top_k",
      ("Who are the top {k} customers by number of completed orders?",
       "List the {k} customers with the most completed orders, with counts."),
      "SELECT c.name, COUNT(*) AS n FROM orders o JOIN customers c ON c.id = o.customer_id "
      "WHERE o.status = 'completed' GROUP BY c.id ORDER BY n DESC, c.id LIMIT {k}", _k),
    T("top_margin_products", "top_k",
      ("Which {k} products have the highest margin (price minus cost)?",
       "Top {k} products by unit margin."),
      "SELECT name, ROUND(price - cost, 2) AS margin FROM products "
      "ORDER BY margin DESC, id LIMIT {k}", _k),
    T("top_ltv_region", "top_k",
      ("Who are the {k} highest lifetime value customers in {region}?",
       "Top {k} customers in {region} by lifetime value."),
      "SELECT name, lifetime_value FROM customers WHERE region = '{region}' "
      "ORDER BY lifetime_value DESC, id LIMIT {k}", _merge(_region, _k)),
    # --- having ---------------------------------------------------------------------------
    T("customers_many_orders", "having",
      ("Which customers placed more than {n} orders? Show names and counts.",
       "List customers with over {n} orders and how many they placed."),
      "SELECT c.name, COUNT(*) FROM orders o JOIN customers c ON c.id = o.customer_id "
      "GROUP BY c.id HAVING COUNT(*) > {n}", lambda r: {"n": r.choice([15, 17, 19])}),
    T("categories_over_revenue", "having",
      ("Which product categories had more than {amount} in completed revenue in {year}?",
       "Categories whose completed {year} revenue exceeded {amount}?"),
      "SELECT p.category FROM order_items oi JOIN orders o ON o.id = oi.order_id "
      "JOIN products p ON p.id = oi.product_id WHERE o.status = 'completed' "
      "AND strftime('%Y', o.order_date) = '{year}' GROUP BY p.category "
      "HAVING SUM(oi.quantity * oi.unit_price) > {amount}",
      _merge(_year, lambda r: {"amount": r.choice([18000000, 25000000, 40000000])})),
    # --- dates ----------------------------------------------------------------------------
    T("orders_in_quarter", "date",
      ("How many orders were placed in Q{q} {year}?", "Count orders from quarter {q} of {year}."),
      "SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '{start}' AND '{end}'", _quarter),
    T("revenue_between", "date",
      ("What was the completed revenue between {d1} and {d2}?",
       "Completed order revenue from {d1} through {d2}?"),
      "SELECT ROUND(SUM(total), 2) FROM orders WHERE status = 'completed' "
      "AND order_date BETWEEN '{d1}' AND '{d2}'", _date_range),
    T("signups_before", "date",
      ("How many customers signed up before {date}?", "Count customers who joined before {date}."),
      "SELECT COUNT(*) FROM customers WHERE signup_date < '{date}'",
      lambda r: {"date": r.choice(["2023-06-01", "2023-12-31", "2024-06-30"])}),
    # --- subqueries -----------------------------------------------------------------------
    T("above_category_avg", "subquery",
      ("Which products are priced above the average price of their category?",
       "List products more expensive than their category's average."),
      "SELECT p.name FROM products p WHERE p.price > "
      "(SELECT AVG(p2.price) FROM products p2 WHERE p2.category = p.category)", _none),
    T("customers_without_orders", "subquery",
      ("How many customers have never placed an order?",
       "Count customers with no orders at all."),
      "SELECT COUNT(*) FROM customers WHERE id NOT IN (SELECT customer_id FROM orders)", _none),
    # --- ratios ---------------------------------------------------------------------------
    T("cancel_refund_share", "ratio",
      ("What percentage of {year} orders were cancelled or refunded?",
       "Share of orders in {year} that ended cancelled or refunded, in percent?"),
      "SELECT ROUND(100.0 * SUM(CASE WHEN status IN ('cancelled', 'refunded') THEN 1 ELSE 0 END)"
      " / COUNT(*), 2) FROM orders WHERE strftime('%Y', order_date) = '{year}'", _year),
    T("category_gross_margin", "ratio",
      ("What is the gross margin percentage of {category} on completed orders?",
       "Gross margin % for the {category} category, completed orders only?"),
      "SELECT ROUND(100.0 * SUM(oi.quantity * (oi.unit_price - p.cost)) / "
      "SUM(oi.quantity * oi.unit_price), 2) FROM order_items oi "
      "JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id "
      "WHERE o.status = 'completed' AND p.category = '{category}'", _category),
]

HELDOUT_TEMPLATES: list[Template] = [
    T("best_region_year", "heldout",
      ("Which customer region generated the most completed revenue in {year}?",),
      "SELECT c.region FROM orders o JOIN customers c ON c.id = o.customer_id "
      "WHERE o.status = 'completed' AND strftime('%Y', o.order_date) = '{year}' "
      "GROUP BY c.region ORDER BY SUM(o.total) DESC LIMIT 1", _year),
    T("orders_per_role", "heldout",
      ("How many orders were handled by each employee role?",),
      "SELECT e.role, COUNT(*) FROM orders o JOIN employees e ON e.id = o.employee_id "
      "GROUP BY e.role", _none),
    T("avg_items_per_order", "heldout",
      ("On average, how many line items does an order have?",),
      "SELECT ROUND(AVG(n), 2) FROM (SELECT COUNT(*) AS n FROM order_items GROUP BY order_id)",
      _none),
    T("busiest_month", "heldout",
      ("Which month of {year} had the most orders?",),
      "SELECT strftime('%m', order_date) AS m FROM orders "
      "WHERE strftime('%Y', order_date) = '{year}' GROUP BY m ORDER BY COUNT(*) DESC LIMIT 1",
      _year),
    T("distinct_buyers_product", "heldout",
      ("How many different customers have bought {product}?",),
      "SELECT COUNT(DISTINCT o.customer_id) FROM orders o JOIN order_items oi "
      "ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id WHERE p.name = '{product}'",
      lambda r: {"product": r.choice(PRODUCTS)}),
    T("enterprise_share_region", "heldout",
      ("What percentage of customers in {region} are on the enterprise tier?",),
      "SELECT ROUND(100.0 * SUM(CASE WHEN tier = 'enterprise' THEN 1 ELSE 0 END) / COUNT(*), 2) "
      "FROM customers WHERE region = '{region}'", _region),
    T("aov_per_tier", "heldout",
      ("What is the average completed order value for each customer tier?",),
      "SELECT c.tier, ROUND(AVG(o.total), 2) FROM orders o JOIN customers c "
      "ON c.id = o.customer_id WHERE o.status = 'completed' GROUP BY c.tier", _none),
    T("new_hires_busy", "heldout",
      ("Which employees hired in {year} handled more than {n} orders?",),
      "SELECT e.name FROM employees e JOIN orders o ON o.employee_id = e.id "
      "WHERE strftime('%Y', e.hire_date) = '{year}' GROUP BY e.id HAVING COUNT(*) > {n}",
      lambda r: {"year": r.choice([2023, 2024]), "n": r.choice([120, 130, 140])}),
]
