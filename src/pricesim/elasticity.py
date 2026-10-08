"""Price elasticity of demand: the naive estimate, why it is wrong, and the estimates the simulator uses.

All specifications are log-log two-way fixed-effects regressions
    log(units_it) = beta * log(price_it) + product_i + week_t + e_it
with standard errors clustered by product, estimated only on weeks before the test quarter.

Why the naive estimate is wrong: the retailer prices on quantity tiers, so a product's average price in
a week falls when large orders arrive. Units and average price then move together mechanically and
the regression reads order-size mix as price sensitivity. The posted price (data.trailing_mode over the
within-tier weekly price) only moves when the list price actually changes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import pyfixest as pf

from .config import SPLIT, MIN_SELLING_WEEKS, REGULAR_WEEKS
from .data import trailing_mode

WINDOWS = (3, 4, 6, 8)   # posted-price windows averaged over in the simulator (model uncertainty)


@dataclass
class Estimate:
    spec: str
    beta: float
    se: float
    n_obs: int
    n_products_with_price_change: int
    note: str = ""

    def as_row(self) -> dict:
        d = asdict(self)
        d["ci_low"], d["ci_high"] = self.beta - 1.96 * self.se, self.beta + 1.96 * self.se
        return d


@dataclass
class SegmentFit:
    window: int
    coef: np.ndarray            # [occasional, regular]
    vcov: np.ndarray            # 2x2
    n_obs: int
    n_changers: dict = field(default_factory=dict)


def with_posted(panel: pd.DataFrame, window: int) -> pd.Series:
    return panel.groupby("sku", group_keys=False)["tier_price"].apply(
        lambda s: pd.Series(trailing_mode(s.to_numpy(dtype=float), window), index=s.index))


def sample(panel: pd.DataFrame, window: int, end=SPLIT.test_start, min_weeks: int = MIN_SELLING_WEEKS) -> pd.DataFrame:
    df = panel.assign(posted=with_posted(panel, window)).dropna(subset=["posted", "tier_price"])
    if end is not None:
        df = df[df["week"] < pd.Timestamp(end)]
    n_weeks = df.groupby("sku")["week"].transform("size")
    df = df[n_weeks >= min_weeks].copy()
    df["regular"] = (df.groupby("sku")["week"].transform("size") >= REGULAR_WEEKS).astype(int)
    df["log_units"] = np.log(df["units"])
    for c in ("avg_price", "tier_price", "posted"):
        df[f"log_{c}"] = np.log(df[c])
    df["log_posted_reg"] = df["log_posted"] * df["regular"]
    df["log_posted_occ"] = df["log_posted"] * (1 - df["regular"])
    df["week_id"] = df["week"].dt.strftime("%Y-%m-%d")
    return df


def _fit(df: pd.DataFrame, rhs: str):
    return pf.feols(f"log_units ~ {rhs} | sku + week_id", data=df, vcov={"CRV1": "sku"})


def changers(df: pd.DataFrame, col: str = "posted") -> int:
    return int((df.groupby("sku")[col].nunique() > 1).sum())


def comparison(panel: pd.DataFrame, window: int = 4) -> list[Estimate]:
    """The table that tells the story: naive -> tier price -> posted price -> by segment -> Q4 diagnostic."""
    pre = sample(panel, window)
    out = []
    for spec, col, note in [
        ("A. naive: average price paid", "avg_price", "order-size mix read as price sensitivity"),
        ("B. weekly tier price", "tier_price", "still moves with week-to-week tier flips"),
        ("C. posted price (4-week mode)", "posted", "moves only when the list price changes"),
    ]:
        f = _fit(pre, f"log_{col}")
        t = f.tidy()
        out.append(Estimate(spec, float(t["Estimate"].iloc[0]), float(t["Std. Error"].iloc[0]), int(f._N),
                            changers(pre, col), note))
    f = _fit(pre, "log_posted_occ + log_posted_reg")
    t = f.tidy()
    reg = pre[pre["regular"] == 1]
    occ = pre[pre["regular"] == 0]
    out.append(Estimate("D. posted price, occasional sellers", float(t.loc["log_posted_occ", "Estimate"]),
                        float(t.loc["log_posted_occ", "Std. Error"]), int(f._N), changers(occ),
                        f"{MIN_SELLING_WEEKS}-{REGULAR_WEEKS - 1} selling weeks before the test quarter"))
    out.append(Estimate("E. posted price, regular sellers", float(t.loc["log_posted_reg", "Estimate"]),
                        float(t.loc["log_posted_reg", "Std. Error"]), int(f._N), changers(reg),
                        f"{REGULAR_WEEKS}+ selling weeks before the test quarter"))
    full = sample(panel, window, end=None)
    f = _fit(full, "log_posted")
    t = f.tidy()
    out.append(Estimate("F. diagnostic: posted price incl. Q4", float(t["Estimate"].iloc[0]), float(t["Std. Error"].iloc[0]),
                        int(f._N), changers(full), "Q4 price rises on seasonal items confound the estimate; not used"))
    return out


def segment_fits(panel: pd.DataFrame, windows=WINDOWS) -> list[SegmentFit]:
    fits = []
    for w in windows:
        df = sample(panel, w)
        f = _fit(df, "log_posted_occ + log_posted_reg")
        names = list(f.coef().index)
        if names != ["log_posted_occ", "log_posted_reg"]:
            raise RuntimeError(f"segment model did not estimate both segments (got {names}); check the sample")
        fits.append(SegmentFit(w, f.coef().to_numpy(), np.asarray(f._vcov),
                               int(f._N), {"occasional": changers(df[df.regular == 0]), "regular": changers(df[df.regular == 1])}))
    return fits


def draw(fits: list[SegmentFit], n: int, rng: np.random.Generator, nonpositive: bool = True) -> np.ndarray:
    """n x 2 draws [occasional, regular]: pick a posted-price window at random, then draw from its sampling distribution.

    Averaging over windows carries model uncertainty (how the posted price is defined) into the simulation,
    on top of sampling uncertainty. With `nonpositive` (the default) draws above zero are redrawn: demand that
    rises with price is ruled out a priori. This moves mass towards more elastic values, so it is the
    conservative choice for evaluating price rises; the untruncated version is reported as a sensitivity.
    """
    def raw(k: int) -> np.ndarray:
        which = rng.integers(0, len(fits), size=k)
        out = np.empty((k, 2))
        for i, f in enumerate(fits):
            m = which == i
            out[m] = rng.multivariate_normal(f.coef, f.vcov, size=int(m.sum()))
        return out

    out = raw(n)
    if nonpositive:
        bad = (out > 0).any(axis=1)
        for _ in range(1000):
            if not bad.any():
                break
            out[bad] = raw(int(bad.sum()))
            bad = (out > 0).any(axis=1)
    return out
