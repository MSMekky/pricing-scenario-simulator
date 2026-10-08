"""Load the UCI Online Retail data, run the SQL layer and build the modelling panel."""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import SQL_DIR, PROCESSED, POSTED_PRICE_WINDOW

RAW_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity",
               "InvoiceDate", "UnitPrice", "CustomerID", "Country"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw invoice lines from the official .xlsx, a .csv, a .parquet or the R package .rda."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str}, encoding="latin1")
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".rda":
        import pyreadr
        df = next(iter(pyreadr.read_r(str(path)).values()))
    else:
        raise ValueError(f"unsupported file type: {suffix}")
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"raw file is missing columns: {sorted(missing)}")
    df = df[RAW_COLUMNS].copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def run_sql(raw: pd.DataFrame, con: duckdb.DuckDBPyConnection | None = None) -> duckdb.DuckDBPyConnection:
    """Run the SQL files in order and enforce the quality gates."""
    con = con or duckdb.connect()
    con.register("raw_df", raw)
    con.execute("CREATE OR REPLACE TABLE raw_lines AS SELECT * FROM raw_df")
    for name in ("01_clean_lines.sql", "02_product_week.sql"):
        con.execute((SQL_DIR / name).read_text())
    checks = con.execute((SQL_DIR / "03_quality_checks.sql").read_text()).df()
    failed = checks[checks["violations"] != 0]
    if len(failed):
        raise RuntimeError(f"data-quality checks failed:\n{failed.to_string(index=False)}")
    return con


def trailing_mode(values: np.ndarray, window: int) -> np.ndarray:
    """Mode of the last `window` observations (inclusive), sticky on ties.

    Used to turn a noisy weekly tier price into a posted price: a one-week flip caused by
    order mix does not move the posted price, a real list-price change does once it holds a
    majority of the window. On a tie the previous posted price is kept. Breaking ties towards the
    current week would let a single big-order week set the price, which is exactly the
    order-mix bias this measure exists to remove.
    """
    out = np.empty(len(values), dtype=float)
    prev = np.nan
    for i in range(len(values)):
        win = values[max(0, i - window + 1): i + 1]
        win = win[~np.isnan(win)]
        if len(win) == 0:
            out[i] = prev
            continue
        uniq, counts = np.unique(win, return_counts=True)
        best = uniq[counts == counts.max()]
        if len(best) == 1:
            prev = best[0]
        elif not (prev in best):
            # no previous value among the tied ones (start of the series): take the earliest in the window
            prev = next(v for v in win if v in best)
        out[i] = prev
    return out


def build_panel(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Product-week panel of selling weeks with the posted price attached."""
    pw = con.execute("""
        SELECT pw.*, t.modal_price, t.description
        FROM product_week pw JOIN product_tier t USING (sku)
        ORDER BY sku, week
    """).df()
    pw["week"] = pd.to_datetime(pw["week"])
    pw["posted_price"] = (
        pw.groupby("sku", group_keys=False)["tier_price"]
          .apply(lambda s: pd.Series(trailing_mode(s.to_numpy(dtype=float), POSTED_PRICE_WINDOW), index=s.index))
    )
    return pw


def build(raw_path: str | Path, out_dir: Path = PROCESSED) -> pd.DataFrame:
    raw = load_raw(raw_path)
    con = run_sql(raw)
    panel = build_panel(con)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / "product_week.parquet", index=False)
    return panel


def load_panel(path: Path = PROCESSED / "product_week.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["week"] = pd.to_datetime(df["week"])
    return df
