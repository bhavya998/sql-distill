"""Hand-written analyst questions with gold SQL — the out-of-template test set.

Written independently of ``templates.py``: different wording, and many need SQL the
templates never produce (weekday filters, recursive CTEs, correlated subqueries,
cross-region joins). Every query is checked to return rows when the splits are built.
"""

from __future__ import annotations

# (question, gold SQL, skill)
HANDWRITTEN: list[tuple[str, str, str]] = [
    ("Which product has the most units in stock?",
     "SELECT name FROM products ORDER BY stock DESC LIMIT 1", "top_k"),
    ("What is our total inventory value at cost?",
     "SELECT ROUND(SUM(cost * stock), 2) FROM products", "aggregate"),
    ("How many orders were handled by employees based in Europe?",
     "SELECT COUNT(*) FROM orders o JOIN employees e ON e.id = o.employee_id "
     "WHERE e.region = 'Europe'", "join"),
    ("Who is our single most valuable customer by lifetime value?",
     "SELECT name FROM customers ORDER BY lifetime_value DESC LIMIT 1", "top_k"),
    ("What was the average order total in the first half of 2025, all statuses included?",
     "SELECT ROUND(AVG(total), 2) FROM orders "
     "WHERE order_date BETWEEN '2025-01-01' AND '2025-06-30'", "date"),
    ("How many orders contain more than three different line items?",
     "SELECT COUNT(*) FROM (SELECT order_id FROM order_items GROUP BY order_id "
     "HAVING COUNT(*) > 3)", "having"),
    ("How many units of Software products sit in pending orders?",
     "SELECT SUM(oi.quantity) FROM order_items oi JOIN orders o ON o.id = oi.order_id "
     "JOIN products p ON p.id = oi.product_id "
     "WHERE p.category = 'Software' AND o.status = 'pending'", "multi_join"),
    ("Which companies do our enterprise customers in Asia Pacific work for? List each once.",
     "SELECT DISTINCT company FROM customers "
     "WHERE tier = 'enterprise' AND region = 'Asia Pacific'", "filter_list"),
    ("How many Sales Managers do we have in each region?",
     "SELECT region, COUNT(*) FROM employees WHERE role = 'Sales Manager' GROUP BY region",
     "group_by"),
    ("What share of 2024 completed revenue came from Consulting, in percent?",
     "SELECT ROUND(100.0 * SUM(CASE WHEN p.category = 'Consulting' "
     "THEN oi.quantity * oi.unit_price ELSE 0 END) / SUM(oi.quantity * oi.unit_price), 2) "
     "FROM order_items oi JOIN orders o ON o.id = oi.order_id "
     "JOIN products p ON p.id = oi.product_id "
     "WHERE o.status = 'completed' AND strftime('%Y', o.order_date) = '2024'", "ratio"),
    ("Which employee brought in the most completed order value?",
     "SELECT e.name FROM orders o JOIN employees e ON e.id = o.employee_id "
     "WHERE o.status = 'completed' GROUP BY e.id ORDER BY SUM(o.total) DESC LIMIT 1", "join"),
    ("How many customers have had at least one order refunded?",
     "SELECT COUNT(DISTINCT customer_id) FROM orders WHERE status = 'refunded'", "aggregate"),
    ("What is the average list price of products whose unit cost is above 2000?",
     "SELECT ROUND(AVG(price), 2) FROM products WHERE cost > 2000", "aggregate"),
    ("Which products are running low, under 100 units? Show name, category and stock.",
     "SELECT name, category, stock FROM products WHERE stock < 100", "filter_list"),
    ("In which month of 2024 did completed orders bring in the most revenue?",
     "SELECT strftime('%Y-%m', order_date) FROM orders WHERE status = 'completed' "
     "AND strftime('%Y', order_date) = '2024' GROUP BY 1 ORDER BY SUM(total) DESC LIMIT 1",
     "date"),
    ("How many distinct products has customer 17 ever ordered?",
     "SELECT COUNT(DISTINCT oi.product_id) FROM order_items oi "
     "JOIN orders o ON o.id = oi.order_id WHERE o.customer_id = 17", "join"),
    ("Break down completed revenue by customer tier.",
     "SELECT c.tier, ROUND(SUM(o.total), 2) FROM orders o JOIN customers c "
     "ON c.id = o.customer_id WHERE o.status = 'completed' GROUP BY c.tier", "group_by"),
    ("How many people did we hire in 2024?",
     "SELECT COUNT(*) FROM employees WHERE strftime('%Y', hire_date) = '2024'", "date"),
    ("What's the average quantity per order line for Hardware products?",
     "SELECT ROUND(AVG(oi.quantity), 2) FROM order_items oi "
     "JOIN products p ON p.id = oi.product_id WHERE p.category = 'Hardware'", "join"),
    ("Give me the names of customers who joined in December 2024.",
     "SELECT name FROM customers WHERE signup_date BETWEEN '2024-12-01' AND '2024-12-31'",
     "date"),
    ("How much completed revenue did the Pro Laptop 15 generate?",
     "SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) FROM order_items oi "
     "JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id "
     "WHERE p.name = 'Pro Laptop 15' AND o.status = 'completed'", "multi_join"),
    ("Which category has the thinnest average unit margin?",
     "SELECT category FROM products GROUP BY category ORDER BY AVG(price - cost) ASC LIMIT 1",
     "group_by"),
    ("How many orders above 100,000 were placed in 2025?",
     "SELECT COUNT(*) FROM orders WHERE total > 100000 AND order_date >= '2025-01-01'", "date"),
    ("Show order counts per quarter of 2024, in quarter order.",
     "SELECT (CAST(strftime('%m', order_date) AS INTEGER) + 2) / 3 AS q, COUNT(*) FROM orders "
     "WHERE strftime('%Y', order_date) = '2024' GROUP BY q ORDER BY q", "date"),
    ("Which Latin America customers have a lifetime value above 1.5 million? Names and values.",
     "SELECT name, lifetime_value FROM customers "
     "WHERE region = 'Latin America' AND lifetime_value > 1500000", "filter_list"),
    ("On average, how many days pass between a customer's signup and their first order?",
     "SELECT ROUND(AVG(julianday(f.first_order) - julianday(c.signup_date)), 2) "
     "FROM customers c JOIN (SELECT customer_id, MIN(order_date) AS first_order FROM orders "
     "GROUP BY customer_id) f ON f.customer_id = c.id", "subquery"),
    ("For each product category, how many distinct customers have bought from it?",
     "SELECT p.category, COUNT(DISTINCT o.customer_id) FROM order_items oi "
     "JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id "
     "GROUP BY p.category", "multi_join"),
    ("How many orders were placed on a Saturday or Sunday?",
     "SELECT COUNT(*) FROM orders WHERE strftime('%w', order_date) IN ('0', '6')", "date"),
    ("For each employee, how many of their orders came from customers outside their own "
     "region?",
     "SELECT e.name, COUNT(*) FROM orders o JOIN employees e ON e.id = o.employee_id "
     "JOIN customers c ON c.id = o.customer_id WHERE c.region <> e.region GROUP BY e.id",
     "multi_join"),
    ("What percentage of customers are premium or enterprise?",
     "SELECT ROUND(100.0 * SUM(CASE WHEN tier IN ('premium', 'enterprise') THEN 1 ELSE 0 END) "
     "/ COUNT(*), 2) FROM customers", "ratio"),
    ("What are the top 3 products by revenue across all orders, whatever the status?",
     "SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS rev FROM order_items oi "
     "JOIN products p ON p.id = oi.product_id GROUP BY p.id ORDER BY rev DESC LIMIT 3",
     "top_k"),
    ("How many products are priced at least 2.5 times their cost?",
     "SELECT COUNT(*) FROM products WHERE price >= 2.5 * cost", "filter_count"),
    ("How many units were ordered in March 2025?",
     "SELECT SUM(oi.quantity) FROM order_items oi JOIN orders o ON o.id = oi.order_id "
     "WHERE o.order_date BETWEEN '2025-03-01' AND '2025-03-31'", "date"),
    ("How many customers ordered in both 2024 and 2025?",
     "SELECT COUNT(*) FROM (SELECT customer_id FROM orders GROUP BY customer_id "
     "HAVING SUM(strftime('%Y', order_date) = '2024') > 0 "
     "AND SUM(strftime('%Y', order_date) = '2025') > 0)", "having"),
    ("Split completed revenue from European customers by tier.",
     "SELECT c.tier, ROUND(SUM(o.total), 2) FROM orders o JOIN customers c "
     "ON c.id = o.customer_id WHERE o.status = 'completed' AND c.region = 'Europe' "
     "GROUP BY c.tier", "group_by"),
    ("List the emails of customers at companies whose name starts with Quantum.",
     "SELECT email FROM customers WHERE company LIKE 'Quantum%'", "filter_list"),
    ("Among customers who ordered at all, what's the average number of orders each?",
     "SELECT ROUND(AVG(n), 2) FROM (SELECT COUNT(*) AS n FROM orders GROUP BY customer_id)",
     "subquery"),
    ("How many pending orders date from before 2025?",
     "SELECT COUNT(*) FROM orders WHERE status = 'pending' AND order_date < '2025-01-01'",
     "date"),
    ("Which Account Executive handled the most orders?",
     "SELECT e.name FROM orders o JOIN employees e ON e.id = o.employee_id "
     "WHERE e.role = 'Account Executive' GROUP BY e.id ORDER BY COUNT(*) DESC LIMIT 1", "join"),
    ("How much order value did we lose to cancellations and refunds in 2024?",
     "SELECT ROUND(SUM(total), 2) FROM orders WHERE status IN ('cancelled', 'refunded') "
     "AND strftime('%Y', order_date) = '2024'", "aggregate"),
    ("Show the three cheapest products and their prices.",
     "SELECT name, price FROM products ORDER BY price ASC LIMIT 3", "top_k"),
    ("How many orders has our highest-lifetime-value customer placed?",
     "SELECT COUNT(*) FROM orders WHERE customer_id = "
     "(SELECT id FROM customers ORDER BY lifetime_value DESC LIMIT 1)", "subquery"),
    ("Which regions have more than 40 customers?",
     "SELECT region FROM customers GROUP BY region HAVING COUNT(*) > 40", "having"),
    ("What was the average completed order value each month of 2025, in month order?",
     "SELECT strftime('%Y-%m', order_date) AS m, ROUND(AVG(total), 2) FROM orders "
     "WHERE status = 'completed' AND strftime('%Y', order_date) = '2025' "
     "GROUP BY m ORDER BY m", "group_by"),
    ("How many customers signed up each year?",
     "SELECT strftime('%Y', signup_date) AS y, COUNT(*) FROM customers GROUP BY y", "group_by"),
    ("What percentage of all orders were handled by Sales Reps?",
     "SELECT ROUND(100.0 * SUM(CASE WHEN e.role = 'Sales Rep' THEN 1 ELSE 0 END) / COUNT(*), 2) "
     "FROM orders o JOIN employees e ON e.id = o.employee_id", "ratio"),
    ("Which customers have had two or more orders cancelled?",
     "SELECT c.name FROM orders o JOIN customers c ON c.id = o.customer_id "
     "WHERE o.status = 'cancelled' GROUP BY c.id HAVING COUNT(*) >= 2", "having"),
    ("Total units sold per category in completed orders, highest first.",
     "SELECT p.category, SUM(oi.quantity) AS units FROM order_items oi "
     "JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id "
     "WHERE o.status = 'completed' GROUP BY p.category ORDER BY units DESC", "multi_join"),
    ("What's the priciest product in each category?",
     "SELECT p.category, p.name FROM products p WHERE p.price = "
     "(SELECT MAX(p2.price) FROM products p2 WHERE p2.category = p.category)", "subquery"),
    ("How many orders included a Security Audit?",
     "SELECT COUNT(DISTINCT oi.order_id) FROM order_items oi "
     "JOIN products p ON p.id = oi.product_id WHERE p.name = 'Security Audit'", "join"),
    ("Rank regions by average customer lifetime value, highest first.",
     "SELECT region, ROUND(AVG(lifetime_value), 2) AS v FROM customers "
     "GROUP BY region ORDER BY v DESC", "group_by"),
    ("How many North American customers have never completed an order?",
     "SELECT COUNT(*) FROM customers WHERE region = 'North America' "
     "AND id NOT IN (SELECT customer_id FROM orders WHERE status = 'completed')", "subquery"),
    ("Which employee has been with us the longest?",
     "SELECT name FROM employees ORDER BY hire_date ASC LIMIT 1", "top_k"),
    ("Completed revenue by the region of the employee who handled it.",
     "SELECT e.region, ROUND(SUM(o.total), 2) FROM orders o JOIN employees e "
     "ON e.id = o.employee_id WHERE o.status = 'completed' GROUP BY e.region", "group_by"),
    ("How many order lines had the maximum quantity of 20?",
     "SELECT COUNT(*) FROM order_items WHERE quantity = 20", "filter_count"),
    ("What percentage of completed 2025 revenue came from enterprise customers?",
     "SELECT ROUND(100.0 * SUM(CASE WHEN c.tier = 'enterprise' THEN o.total ELSE 0 END) "
     "/ SUM(o.total), 2) FROM orders o JOIN customers c ON c.id = o.customer_id "
     "WHERE o.status = 'completed' AND strftime('%Y', o.order_date) = '2025'", "ratio"),
    ("How many days in 2024 had no orders at all?",
     "WITH RECURSIVE days(d) AS (SELECT '2024-01-01' UNION ALL "
     "SELECT date(d, '+1 day') FROM days WHERE d < '2024-12-31') "
     "SELECT COUNT(*) FROM days WHERE d NOT IN (SELECT order_date FROM orders)", "date"),
    ("Which customers spent more in completed orders than the average customer in their "
     "region? Return names.",
     "SELECT c.name FROM customers c WHERE c.lifetime_value > "
     "(SELECT AVG(c2.lifetime_value) FROM customers c2 WHERE c2.region = c.region)",
     "subquery"),
    ("List the products sold in every one of the five regions.",
     "SELECT p.name FROM products p JOIN order_items oi ON oi.product_id = p.id "
     "JOIN orders o ON o.id = oi.order_id JOIN customers c ON c.id = o.customer_id "
     "GROUP BY p.id HAVING COUNT(DISTINCT c.region) = 5", "having"),
]
