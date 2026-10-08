-- Product-week panel.
-- The retailer sells on a quantity-tier price list: the same product is cheaper per unit in larger orders,
-- and some customers pay a higher small-order price. A product's average price in a week therefore moves
-- with the mix of order sizes, not with any pricing decision. We keep two price measures so the bias can be shown:
--   avg_price   : revenue / units (contaminated by order-size mix)
--   tier_price  : the modal price within the product's main tier (0.8x to 1.25x its overall modal price)
-- The posted price used for modelling is built from tier_price in Python (trailing 4-week mode), see data.py.
CREATE OR REPLACE TABLE product_tier AS
SELECT sku, mode(price) AS modal_price, any_value(description) AS description
FROM lines
GROUP BY sku;

CREATE OR REPLACE TABLE product_week AS
SELECT
    l.sku,
    l.week,
    sum(l.qty)                                   AS units,
    sum(l.qty * l.price)                         AS revenue,
    sum(l.qty * l.price) / sum(l.qty)            AS avg_price,
    mode(l.price) FILTER (WHERE l.price BETWEEN 0.8 * t.modal_price AND 1.25 * t.modal_price) AS tier_price,
    count(DISTINCT l.customer)                   AS customers,
    count(DISTINCT l.invoice)                    AS orders
FROM lines l
JOIN product_tier t USING (sku)
GROUP BY l.sku, l.week;
