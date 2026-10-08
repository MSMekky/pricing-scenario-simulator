"""End-to-end run: data -> SQL -> elasticity -> forecast backtests -> scenarios -> report artifacts.

    python -m pricesim.pipeline --raw "data/raw/Online Retail.xlsx"
    python -m pricesim.pipeline            # reuse data/processed/product_week.parquet
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import data, elasticity, forecast, simulate
from .config import SPLIT, PROCESSED, REPORTS, SEED, REGULAR_WEEKS

N_DRAWS = 10000
N_BASELINE = 2000
COST_RATIO = 0.60
MIN_GROWTH_ORIGINS = 8
BACKTEST_ORIGINS = {"Q3 (Jun-Aug 2011)": "2011-06-06", "Q4 to date (Sep-Nov 2011)": "2011-09-05"}


def backtests(grid: forecast.Grid, rng: np.random.Generator) -> list[dict]:
    rows = []
    H = forecast.H
    p = grid.base_price
    for label, date in BACKTEST_ORIGINS.items():
        o = grid.row(date)
        actual_w = grid.units[o:o + H]
        actual_q = actual_w.sum(0)
        model = forecast.fit_gbm(grid, o)
        one = forecast.gbm_one_step(grid, model, o)
        mean4 = np.array([grid.units[t - 4:t].mean(0) for t in range(o, o + H)])
        gq = forecast.gbm_quarter(grid, model, o).sum(0)
        nq = forecast.naive_quarter(grid, o)
        n_origins = len(forecast.growth_vectors(grid, o))
        actual_rev = float(grid.revenue[o:o + H].sum())
        if n_origins >= MIN_GROWTH_ORIGINS:
            samples = forecast.baseline_paths(grid, o, N_BASELINE, rng)
            lo, hi = np.quantile(samples, [0.05, 0.95], axis=0)
            tot = (samples * p).sum(1)
            cov = float(((actual_q >= lo) & (actual_q <= hi)).mean())
            r05, r50, r95 = (float(np.quantile(tot, a)) for a in (0.05, 0.5, 0.95))
        else:  # not enough history before this origin to measure growth uncertainty
            cov = r05 = r50 = r95 = None
        rows.append({
            "quarter": label,
            "growth_origins_available": int(n_origins),
            "weekly_wape_gbm": forecast.wape(actual_w, one),
            "weekly_wape_mean4": forecast.wape(actual_w, mean4),
            "quarter_wape_gbm": forecast.wape(actual_q, gq),
            "quarter_wape_naive": forecast.wape(actual_q, nq),
            "coverage_90_product": cov,
            "revenue_actual": actual_rev,
            "revenue_naive": float((nq * p).sum()),
            "revenue_p05": r05, "revenue_p50": r50, "revenue_p95": r95,
        })
    return rows


def main(raw_path: str | None) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    panel = data.build(raw_path) if raw_path else data.load_panel()

    # 1. elasticity: the comparison table, then the regular-seller draws used by the simulator
    comp = elasticity.comparison(panel)
    pd.DataFrame([e.as_row() for e in comp]).to_csv(REPORTS / "elasticity.csv", index=False)
    fits = elasticity.segment_fits(panel)
    eps = elasticity.draw(fits, N_DRAWS, rng)[:, 1]                         # column 1 = regular sellers
    eps_raw = elasticity.draw(fits, N_DRAWS, rng, nonpositive=False)[:, 1]  # sensitivity: no sign restriction
    by_window = [{"window_weeks": f.window, "occasional": float(f.coef[0]), "occasional_se": float(np.sqrt(f.vcov[0, 0])),
                  "regular": float(f.coef[1]), "regular_se": float(np.sqrt(f.vcov[1, 1])), **{f"changers_{k}": v for k, v in f.n_changers.items()}}
                 for f in fits]

    # 2. demand baseline: product set = regular sellers; backtests on two held-out quarters
    grid = forecast.make_grid(panel, REGULAR_WEEKS)
    bt = backtests(grid, rng)
    pd.DataFrame(bt).to_csv(REPORTS / "backtest.csv", index=False)

    # 3. planning quarter: the 13 weeks from the test origin, at current prices
    origin = grid.row(SPLIT.test_start)
    q = forecast.baseline_paths(grid, origin, N_BASELINE, rng)
    p = grid.base_price
    point = forecast.naive_quarter(grid, origin)
    test_end = grid.weeks[origin + forecast.H - 1]
    in_test = (panel["week"] >= pd.Timestamp(SPLIT.test_start)) & (panel["week"] <= test_end)
    share = float(grid.revenue[origin:origin + forecast.H].sum() / panel.loc[in_test, "revenue"].sum())

    top = (point * p) >= np.quantile(point * p, 0.8)
    n = len(grid.skus)
    scenarios = [simulate.uniform(f"{c:+.0%} on all products", c, n) for c in (-0.10, -0.05, 0.05, 0.10)]
    scenarios += [simulate.targeted("+5% on the top 20% by revenue", 0.05, top),
                  simulate.targeted("+5% on the other 80%", 0.05, ~top)]
    rows, store = [], []
    for sc in scenarios:
        d = simulate.run(q, p, eps, sc, COST_RATIO, rng)
        rows.append(simulate.summarise(sc.name, d))
        store.append(d.assign(scenario=sc.name).sample(4000, random_state=SEED))
    pd.DataFrame(rows).to_csv(REPORTS / "scenarios.csv", index=False)

    # sensitivity: elasticity draws capped at zero (no upward-sloping demand), and the cost assumption
    plus5 = scenarios[2]
    capped = simulate.summarise(plus5.name + ", no sign restriction on elasticity",
                                simulate.run(q, p, eps_raw, plus5, COST_RATIO, rng))
    cost_rows = [{"cost_ratio": c, "breakeven_elasticity_gp": simulate.breakeven_elasticity(0.05, c)["gross_profit"],
                  "p_gp_up": simulate.summarise("", simulate.run(q, p, eps, plus5, c, rng))["p_gp_up"]}
                 for c in (0.4, 0.5, 0.6, 0.7, 0.8)]
    pd.concat(store).to_parquet(PROCESSED / "scenario_draws.parquet", index=False)

    np.savez_compressed(PROCESSED / "simulation_inputs.npz",
                        quarter_units=q.astype(np.float32), base_price=p, skus=grid.skus, point=point, top=top,
                        elasticity_draws=eps, weekly_revenue=grid.revenue.sum(1),
                        weeks=np.array([str(w.date()) for w in grid.weeks]), origin=origin)

    results = {
        "data": {"product_weeks": int(len(panel)), "products": int(panel["sku"].nunique()),
                 "simulated_products": n, "share_of_quarter_revenue_simulated": share},
        "elasticity_comparison": [e.as_row() for e in comp],
        "elasticity_by_window": by_window,
        "elasticity_used": {"segment": "regular sellers", "mean": float(eps.mean()),
                            "p05": float(np.quantile(eps, 0.05)), "p95": float(np.quantile(eps, 0.95)),
                            "p_above_minus_1": float((eps > -1).mean()),
                            "unrestricted_mean": float(eps_raw.mean()), "unrestricted_p_positive": float((eps_raw > 0).mean())},
        "breakeven_plus5": simulate.breakeven_elasticity(0.05, COST_RATIO),
        "sensitivity_capped_elasticity": capped,
        "sensitivity_cost_ratio": cost_rows,
        "cost_ratio_assumption": COST_RATIO,
        "backtests": bt,
        "scenarios": rows,
        "settings": {"draws": N_DRAWS, "baseline_samples": N_BASELINE, "seed": SEED},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", help="raw Online Retail file (.xlsx/.csv/.parquet/.rda)")
    r = main(ap.parse_args().raw)
    print(json.dumps({k: r[k] for k in ("data", "elasticity_used", "backtests", "runtime_seconds")}, indent=2))
