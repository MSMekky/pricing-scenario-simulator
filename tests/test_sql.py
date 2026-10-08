import datetime as dt

import pandas as pd
import pytest

from pricesim.data import run_sql


def _raw(rows):
    cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    df = pd.DataFrame(rows, columns=cols)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


BASE = [
    ["536365", "85123A", "HEART HOLDER", 6, "2011-01-03 09:00", 2.95, 17850.0, "United Kingdom"],
    ["536366", "85123A", "HEART HOLDER", 32, "2011-01-04 10:00", 2.55, 13047.0, "United Kingdom"],
    ["536367", "22423", "CAKESTAND", 2, "2011-01-05 11:00", 12.75, 13047.0, "France"],
]


def test_cleaning_rules_drop_every_bad_line():
    bad = [
        ["C536368", "85123A", "HEART HOLDER", -6, "2011-01-05 12:00", 2.95, 17850.0, "United Kingdom"],  # cancellation
        ["536369", "POST", "POSTAGE", 1, "2011-01-05 12:00", 18.0, 12583.0, "France"],                 # non-product
        ["536370", "22423", "CAKESTAND", 0, "2011-01-05 12:00", 12.75, 12583.0, "France"],               # zero qty
        ["536371", "22423", "CAKESTAND", 1, "2011-01-05 12:00", 0.0, 12583.0, "France"],                 # zero price
        ["536372", "22423", "CAKESTAND", 1, "2011-01-05 12:00", 12.75, None, "United Kingdom"],          # no customer
        ["536373", "22423", "CAKESTAND", 12000, "2011-01-05 12:00", 1.0, 12583.0, "France"],             # bulk outlier
        ["536374", "22423", "CAKESTAND", 1, "2011-12-06 12:00", 12.75, 12583.0, "France"],               # partial last week
    ]
    con = run_sql(_raw(BASE + bad))
    assert con.execute("SELECT count(*) FROM lines").fetchone()[0] == len(BASE)


def test_product_week_aggregates_and_reconciles():
    con = run_sql(_raw(BASE))
    pw = con.execute("SELECT * FROM product_week WHERE sku = '85123A'").df()
    assert len(pw) == 1
    row = pw.iloc[0]
    assert row.units == 38
    assert row.revenue == pytest.approx(6 * 2.95 + 32 * 2.55)
    assert row.avg_price == pytest.approx(row.revenue / row.units)
    assert pd.Timestamp(row.week).date() == dt.date(2011, 1, 3)   # Monday


def test_tier_price_ignores_other_tiers():
    rows = [["5000%d" % i, "21000", "MUG", q, "2011-02-0%d 10:00" % (i + 1), p, 12000.0 + i, "United Kingdom"]
            for i, (q, p) in enumerate([(1, 1.25), (1, 1.25), (2, 1.25), (48, 0.85), (6, 2.50)])]
    con = run_sql(_raw(rows))
    tp = con.execute("SELECT tier_price FROM product_week").fetchone()[0]
    assert tp == pytest.approx(1.25)   # modal tier; 0.85 (bulk) and 2.50 (small-order) are outside 0.8x-1.25x


def test_quality_gate_stops_the_pipeline(monkeypatch):
    import pricesim.data as d
    real = d.SQL_DIR

    class Fake:
        def __truediv__(self, name):
            path = real / name
            if name == "01_clean_lines.sql":   # sabotage: keep cancellations
                return type("P", (), {"read_text": lambda self: path.read_text().replace("InvoiceNo NOT LIKE 'C%'", "TRUE")})()
            return path
    monkeypatch.setattr(d, "SQL_DIR", Fake())
    bad = BASE + [["C1", "85123A", "HEART HOLDER", 6, "2011-01-03 09:00", 2.95, 17850.0, "United Kingdom"]]
    with pytest.raises(RuntimeError, match="cancelled invoices present"):
        d.run_sql(_raw(bad))
