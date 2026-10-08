-- Clean invoice lines into sellable product sales.
-- Input: table raw_lines (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country)
-- Rules, each one checked in 03_quality_checks.sql:
--   * cancellations (InvoiceNo starting with 'C') removed
--   * non-product codes removed (postage, fees, manual adjustments, vouchers): product codes are 5 digits plus an optional letter
--   * non-positive quantity or price removed
--   * lines without a customer id removed: they are a separate retail channel with its own price list
--   * single lines of 10,000+ units removed (a handful of bulk orders that were cancelled or are clear outliers)
CREATE OR REPLACE TABLE lines AS
SELECT
    InvoiceNo                       AS invoice,
    StockCode                       AS sku,
    Description                     AS description,
    CAST(Quantity AS DOUBLE)        AS qty,
    CAST(UnitPrice AS DOUBLE)       AS price,
    CAST(CustomerID AS BIGINT)      AS customer,
    Country                         AS country,
    InvoiceDate                     AS ts,
    CAST(date_trunc('week', InvoiceDate) AS DATE) AS week
FROM raw_lines
WHERE InvoiceNo NOT LIKE 'C%'
  AND regexp_matches(StockCode, '^[0-9]{5}[A-Za-z]?$')
  AND Quantity > 0 AND Quantity < 10000
  AND UnitPrice > 0
  AND CustomerID IS NOT NULL
  -- the last week in the data is partial (data ends on a Friday)
  AND date_trunc('week', InvoiceDate) < DATE '2011-12-05';
