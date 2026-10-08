"""Monte Carlo pricing scenarios.

Two sources of uncertainty are combined in every draw:
  * demand: one sample of next-quarter units per product (forecast.baseline_paths)
  * price response: one draw of the elasticity (elasticity.draw: posted-price definition and sampling error)
A scenario's result is always compared with the status quo on the same path and the same elasticity
draw, so the difference isolates the effect of the price change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    name: str
    price_change: np.ndarray  # per product, e.g. 0.05 for +5%


def uniform(name: str, change: float, n_products: int) -> Scenario:
    return Scenario(name, np.full(n_products, change))


def targeted(name: str, change: float, mask: np.ndarray) -> Scenario:
    return Scenario(name, np.where(mask, change, 0.0))


def run(quarter_units: np.ndarray, base_price: np.ndarray, elasticity_draws: np.ndarray,
        scenario: Scenario, cost_ratio: float, rng: np.random.Generator) -> pd.DataFrame:
    """Return one row per draw: revenue and gross profit for the scenario and the status quo.

    quarter_units: samples x products, status-quo units for the quarter
    cost_ratio:  unit cost as a share of the current price (an assumption: the data has no costs)
    """
    k = rng.integers(0, len(quarter_units), size=len(elasticity_draws))
    Qk = quarter_units[k]                                          # draws x products
    eps = elasticity_draws[:, None]
    d = scenario.price_change[None, :]
    p = base_price[None, :]
    q_new = Qk * (1.0 + d) ** eps
    rev_base = (Qk * p).sum(axis=1)
    rev_new = (q_new * p * (1.0 + d)).sum(axis=1)
    gp_base = (Qk * p * (1.0 - cost_ratio)).sum(axis=1)
    gp_new = (q_new * p * (1.0 + d - cost_ratio)).sum(axis=1)
    return pd.DataFrame({"elasticity": elasticity_draws, "revenue_base": rev_base, "revenue": rev_new,
                         "gp_base": gp_base, "gp": gp_new})


def summarise(name: str, draws: pd.DataFrame) -> dict:
    d_rev = draws["revenue"] - draws["revenue_base"]
    d_gp = draws["gp"] - draws["gp_base"]
    q = lambda s, a: float(np.quantile(s, a))
    return {
        "scenario": name,
        "revenue_p50": q(draws["revenue"], 0.5),
        "revenue_p05": q(draws["revenue"], 0.05),
        "revenue_p95": q(draws["revenue"], 0.95),
        "delta_revenue_p50": q(d_rev, 0.5),
        "delta_revenue_p05": q(d_rev, 0.05),
        "delta_revenue_p95": q(d_rev, 0.95),
        "p_revenue_up": float((d_rev > 0).mean()),
        "delta_gp_p50": q(d_gp, 0.5),
        "delta_gp_p05": q(d_gp, 0.05),
        "delta_gp_p95": q(d_gp, 0.95),
        "p_gp_up": float((d_gp > 0).mean()),
    }


def breakeven_elasticity(change: float, cost_ratio: float) -> dict:
    """Elasticity at which a uniform price change leaves revenue or gross profit unchanged.

    Revenue: (1+d)^(1+e) = 1                 -> e = -1
    Gross profit: (1+d)^e * (1+d-c) = 1-c   -> e = ln((1-c)/(1+d-c)) / ln(1+d)
    A price rise pays off whenever demand is less elastic (closer to zero) than the break-even value.
    """
    gp = np.log((1 - cost_ratio) / (1 + change - cost_ratio)) / np.log(1 + change)
    return {"revenue": -1.0, "gross_profit": float(gp)}
