"""Demand baseline for the planning quarter, and an honest test of whether ML beats it.

Grid: full product x week matrix of units (zero when a product did not sell) for the simulated product set.

Two candidates for next quarter's units per product:
  * naive: the last 13 weeks repeated
  * gradient boosting (Poisson loss) on lagged demand features, rolled forward week by week
Both are scored on two held-out quarters. With one year of history the naive baseline wins at the
quarter level, so the simulator uses it; see reports/results.md.

Uncertainty in the baseline comes from the empirical quarter-on-quarter growth of every product,
measured at each weekly origin before the planning quarter and resampled as whole vectors so the
correlation between products (a good or bad quarter for the whole shop) is kept.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import SPLIT, REGULAR_WEEKS, SEED

FEATURES = ["lag1", "lag2", "lag3", "lag4", "mean4", "mean8", "mean13", "sell_rate8",
            "weeks_since_sale", "age", "agg_lag1", "agg_mean4", "log_modal_price"]
WARMUP = 8
H = SPLIT.horizon


@dataclass
class Grid:
    weeks: pd.DatetimeIndex     # trading weeks (the shop's closure weeks are absent from the data)
    skus: np.ndarray
    units: np.ndarray           # weeks x products
    revenue: np.ndarray         # weeks x products
    log_modal_price: np.ndarray # products
    base_price: np.ndarray      # products: realised revenue per unit over the 8 weeks before the test quarter

    def row(self, date) -> int:
        return int(self.weeks.get_loc(pd.Timestamp(date)))

    @property
    def Y(self) -> np.ndarray:
        return np.log1p(self.units)


def make_grid(panel: pd.DataFrame, min_weeks: int = REGULAR_WEEKS) -> Grid:
    test_start = pd.Timestamp(SPLIT.test_start)
    pre = panel[panel["week"] < test_start]
    n = pre.groupby("sku")["week"].size()
    skus = np.sort(n[n >= min_weeks].index.to_numpy())
    weeks = pd.DatetimeIndex(np.sort(panel["week"].unique()))
    p = panel[panel["sku"].isin(skus)]
    piv = lambda col: p.pivot_table(index="week", columns="sku", values=col, aggfunc="sum").reindex(
        index=weeks, columns=skus).fillna(0.0).to_numpy()
    units, rev = piv("units"), piv("revenue")
    modal = p.groupby("sku")["modal_price"].first().reindex(skus).to_numpy()
    t0 = int(weeks.get_loc(test_start))
    u8, r8 = units[t0 - 8:t0].sum(0), rev[t0 - 8:t0].sum(0)
    # prices are fixed at the eight weeks before the planning quarter; in the earlier backtest quarter they only
    # weight units into revenue, and list prices barely move over the year
    base = np.where(u8 > 0, r8 / np.maximum(u8, 1e-9), modal)
    return Grid(weeks, skus, units, rev, np.log(modal), base)


# ---------------------------------------------------------------- gradient boosting candidate
def features(Y: np.ndarray, t: int, log_modal_price: np.ndarray) -> np.ndarray:
    """Features (products x 13) for week t, from rows < t of Y = log1p(units)."""
    hist = Y[:t]
    sold = hist > 0
    N = hist.shape[1]
    mean = lambda k: hist[max(0, t - k):].mean(axis=0)
    since = np.where(sold.any(axis=0), np.argmax(sold[::-1], axis=0) + 1, 52)
    first = np.where(sold.any(axis=0), np.argmax(sold, axis=0), t)
    agg = hist.mean(axis=1)
    cols = [hist[t - k] for k in (1, 2, 3, 4)] + [
        mean(4), mean(8), mean(13), sold[max(0, t - 8):].mean(axis=0),
        np.minimum(since, 13), np.minimum(t - first, 52),
        np.full(N, agg[t - 1]), np.full(N, agg[t - 4:t].mean()), log_modal_price]
    return np.column_stack(cols)


def fit_gbm(grid: Grid, end_row: int) -> HistGradientBoostingRegressor:
    Y = grid.Y
    X = np.vstack([features(Y, t, grid.log_modal_price) for t in range(WARMUP, end_row)])
    y = np.concatenate([grid.units[t] for t in range(WARMUP, end_row)])
    m = HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.05, min_samples_leaf=50,
                                      l2_regularization=1.0, random_state=SEED)
    return m.fit(X, y)


def gbm_quarter(grid: Grid, model, origin: int) -> np.ndarray:
    """Roll the model forward H weeks from `origin`, feeding its own predictions back as lags."""
    Y = grid.Y[:origin].copy()
    out = []
    for h in range(H):
        mu = model.predict(features(Y, origin + h, grid.log_modal_price))
        out.append(mu)
        Y = np.vstack([Y, np.log1p(mu)])
    return np.array(out)


def gbm_one_step(grid: Grid, model, origin: int) -> np.ndarray:
    return np.array([model.predict(features(grid.Y, t, grid.log_modal_price)) for t in range(origin, origin + H)])


# ---------------------------------------------------------------- naive baseline and its uncertainty
def naive_quarter(grid: Grid, origin: int) -> np.ndarray:
    return grid.units[origin - H:origin].sum(axis=0)


def growth_vectors(grid: Grid, origin: int) -> np.ndarray:
    """Quarter-on-quarter growth ratios (origins x products) for every weekly origin whose next quarter ends before `origin`."""
    out = []
    for t in range(H, origin - H + 1):
        prev = grid.units[t - H:t].sum(0)
        nxt = grid.units[t:t + H].sum(0)
        g = (nxt + 1.0) / (prev + 1.0)
        # rescale so the vector reproduces that origin's total growth exactly: product-level ratios
        # on small counts are skewed upwards, and without this the shop-level forecast drifts high
        g *= nxt.sum() / (prev * g).sum()
        out.append(g)
    return np.array(out)


def baseline_paths(grid: Grid, origin: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """n x products samples of next-quarter units: naive baseline times a resampled growth vector."""
    g = growth_vectors(grid, origin)
    base = naive_quarter(grid, origin)
    return np.maximum((base + 1.0)[None, :] * g[rng.integers(0, len(g), size=n)] - 1.0, 0.0)


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(actual - pred).sum() / np.abs(actual).sum())
