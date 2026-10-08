-- Data-quality gates. Every row must return violations = 0; the pipeline stops otherwise.
SELECT 'cancelled invoices present' AS check_name, count(*) AS violations FROM lines WHERE invoice LIKE 'C%'
UNION ALL
SELECT 'non-product codes present', count(*) FROM lines WHERE NOT regexp_matches(sku, '^[0-9]{5}[A-Za-z]?$')
UNION ALL
SELECT 'non-positive quantity or price', count(*) FROM lines WHERE qty <= 0 OR price <= 0
UNION ALL
SELECT 'missing customer id', count(*) FROM lines WHERE customer IS NULL
UNION ALL
SELECT 'duplicate product-week keys', count(*) - count(DISTINCT (sku, week)) FROM product_week
UNION ALL
SELECT 'revenue does not reconcile with lines',
       CAST(abs((SELECT sum(revenue) FROM product_week) - (SELECT sum(qty * price) FROM lines)) > 0.01 AS INTEGER)
UNION ALL
SELECT 'weeks not on Monday', count(*) FROM product_week WHERE dayofweek(week) <> 1;
