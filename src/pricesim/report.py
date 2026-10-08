"""Figures and the markdown results page, built from reports/results.json and the saved draws."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import REPORTS, FIGURES, PROCESSED

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def fig_elasticity(res: dict) -> None:
    rows = res["elasticity_comparison"]
    labels = [r["spec"] for r in rows][::-1]
    b = np.array([r["beta"] for r in rows])[::-1]
    lo = np.array([r["ci_low"] for r in rows])[::-1]
    hi = np.array([r["ci_high"] for r in rows])[::-1]
    used = [r["spec"].startswith("E.") for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(rows))
    for i in range(len(rows)):
        c = BLUE if used[i] else MUTED
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=c, lw=2, solid_capstyle="round")
        ax.plot(b[i], y[i], "o", color=c, ms=8, mec=SURFACE, mew=2)
        ax.text(hi[i] + 0.25, y[i], f"{b[i]:.2f}", va="center", color=INK if used[i] else MUTED, fontsize=9)
    be = res["breakeven_plus5"]["gross_profit"]
    for x, lab, ha, dx in [(-1, "revenue\nbreak-even", "left", 0.08),
                           (be, f"gross-profit break-even\n(+5%, cost {res['cost_ratio_assumption']:.0%} of price)", "right", -0.08)]:
        ax.axvline(x, color=ORANGE, lw=1.5, ls=(0, (4, 3)))
        ax.text(x + dx, len(rows) - 0.45, lab, color=MUTED, fontsize=8, ha=ha, va="bottom")
    ax.set_yticks(y, labels)
    ax.set_xlabel("price elasticity of demand (95% CI, clustered by product)")
    ax.set_ylim(-0.6, len(rows) + 0.5)
    ax.set_title("Naive estimates read order-size mix as price sensitivity. Corrected, regular sellers show\nlittle measurable sensitivity, well inside the gross-profit break-even.",
                 loc="left", fontsize=11, color=INK, pad=22)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "elasticity.png", dpi=160)
    plt.close(fig)


def fig_scenarios(res: dict) -> None:
    d = pd.read_parquet(PROCESSED / "scenario_draws.parquet")
    names = [s["scenario"] for s in res["scenarios"]][:4]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (col, base, title) in zip(axes, [("revenue", "revenue_base", "Change in quarterly revenue"),
                                             ("gp", "gp_base", "Change in quarterly gross profit")]):
        data = [(d.loc[d.scenario == n, col] - d.loc[d.scenario == n, base]).to_numpy() / 1e3 for n in names]
        parts = ax.violinplot(data, orientation="horizontal", showextrema=False, widths=0.8)
        for pc, n in zip(parts["bodies"], names):
            pc.set_facecolor(BLUE if n.startswith("+") else ORANGE)
            pc.set_edgecolor(SURFACE)
            pc.set_alpha(0.85)
        for i, x in enumerate(data, start=1):
            q05, q50, q95 = np.quantile(x, [0.05, 0.5, 0.95])
            ax.plot([q05, q95], [i, i], color=INK, lw=1)
            ax.plot(q50, i, "o", color=INK, ms=5)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("GBP thousand vs current prices (median and 90% interval)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(1, len(names) + 1), names)
    fig.tight_layout()
    fig.savefig(FIGURES / "scenarios.png", dpi=160)
    plt.close(fig)


def fig_backtest(res: dict) -> None:
    z = np.load(PROCESSED / "simulation_inputs.npz", allow_pickle=True)
    weeks = pd.to_datetime(z["weeks"])
    rev = z["weekly_revenue"] / 1e3
    o = int(z["origin"])
    bt = [b for b in res["backtests"] if b["revenue_p50"] is not None][-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2.3, 1]})
    a.plot(weeks[:o], rev[:o], color=BLUE, lw=2, label="history used")
    a.plot(weeks[o - 1:o + 13], rev[o - 1:o + 13], color=ORANGE, lw=2, label="held-out quarter")
    a.set_title("Weekly revenue, simulated product set", loc="left", fontsize=10.5)
    a.set_ylabel("GBP thousand")
    a.legend(frameon=False, loc="upper left")
    vals = [bt["revenue_naive"], bt["revenue_p50"], bt["revenue_actual"]]
    labs = ["last quarter\n(naive)", "simulated\nmedian", "actual"]
    b.bar(range(3), np.array(vals) / 1e3, color=[MUTED, BLUE, ORANGE], width=0.6)
    b.errorbar(1, bt["revenue_p50"] / 1e3, yerr=[[(bt["revenue_p50"] - bt["revenue_p05"]) / 1e3],
                                                [(bt["revenue_p95"] - bt["revenue_p50"]) / 1e3]],
               color=INK, capsize=6, lw=1.2)
    b.set_xticks(range(3), labs)
    b.set_title("Held-out quarter total (90% interval)", loc="left", fontsize=10.5)
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "backtest.png", dpi=160)
    plt.close(fig)


def pct(x: float) -> str:
    """Percent that never rounds a non-certain probability to 0% or 100%."""
    if 0 < x < 0.005:
        return "<1%"
    if 0.995 < x < 1:
        return ">99%"
    return f"{x:.0%}"


def gbp(x: float) -> str:
    return f"£{x/1e3:,.0f}k" if abs(x) >= 1e3 else f"£{x:,.0f}"


def results_md(res: dict) -> str:
    L = ["# Results", "", f"Generated by `python -m pricesim.report` from `results.json` (seed {res['settings']['seed']}).", ""]
    L += ["## Elasticity", "", "| specification | estimate | 95% CI | products with a price change | note |", "|---|---|---|---|---|"]
    for r in res["elasticity_comparison"]:
        L.append(f"| {r['spec']} | {r['beta']:.2f} | {r['ci_low']:.2f} to {r['ci_high']:.2f} | {r['n_products_with_price_change']} | {r['note']} |")
    L += ["", "Robustness to the posted-price window (regular sellers, used by the simulator):", "",
          "| window (weeks) | regular sellers | occasional sellers |", "|---|---|---|"]
    for w in res["elasticity_by_window"]:
        L.append(f"| {w['window_weeks']} | {w['regular']:.2f} (se {w['regular_se']:.2f}) | {w['occasional']:.2f} (se {w['occasional_se']:.2f}) |")
    e = res["elasticity_used"]
    L += ["", f"Simulator draws (four windows pooled, positive values ruled out): mean {e['mean']:.2f}, "
          f"90% interval {e['p05']:.2f} to {e['p95']:.2f}; P(less elastic than -1) = {e['p_above_minus_1']:.0%}. "
          f"Without the sign restriction the pooled mean is {e['unrestricted_mean']:.2f} and {e['unrestricted_p_positive']:.0%} of draws are positive.", ""]
    L += ["## Forecast backtests", "", "| quarter | weekly WAPE, GBM | weekly WAPE, 4-week mean | quarter WAPE, GBM | quarter WAPE, naive | 90% product coverage | actual | naive | simulated p05 / p50 / p95 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for b in res["backtests"]:
        cov = "n/a" if b["coverage_90_product"] is None else f"{b['coverage_90_product']:.0%}"
        sim = "n/a (too little history)" if b["revenue_p50"] is None else f"{gbp(b['revenue_p05'])} / {gbp(b['revenue_p50'])} / {gbp(b['revenue_p95'])}"
        L.append(f"| {b['quarter']} | {b['weekly_wape_gbm']:.3f} | {b['weekly_wape_mean4']:.3f} | {b['quarter_wape_gbm']:.3f} | {b['quarter_wape_naive']:.3f} | {cov} | {gbp(b['revenue_actual'])} | {gbp(b['revenue_naive'])} | {sim} |")
    L += ["", "## Scenarios (planning quarter, regular sellers)", "",
          "| scenario | median revenue change | 90% interval | P(revenue up) | median gross-profit change | 90% interval | P(gross profit up) |", "|---|---|---|---|---|---|---|"]
    for s in res["scenarios"] + [res["sensitivity_capped_elasticity"]]:
        L.append(f"| {s['scenario']} | {gbp(s['delta_revenue_p50'])} | {gbp(s['delta_revenue_p05'])} to {gbp(s['delta_revenue_p95'])} | {pct(s['p_revenue_up'])} | "
                 f"{gbp(s['delta_gp_p50'])} | {gbp(s['delta_gp_p05'])} to {gbp(s['delta_gp_p95'])} | {pct(s['p_gp_up'])} |")
    L += ["", f"Gross profit assumes unit cost = {res['cost_ratio_assumption']:.0%} of current price (the data has no costs). Sensitivity for +5% on all products:", "",
          "| cost ratio | break-even elasticity | P(gross profit up) |", "|---|---|---|"]
    for c in res["sensitivity_cost_ratio"]:
        L.append(f"| {c['cost_ratio']:.0%} | {c['breakeven_elasticity_gp']:.2f} | {pct(c['p_gp_up'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    res = json.loads((REPORTS / "results.json").read_text())
    fig_elasticity(res)
    fig_scenarios(res)
    fig_backtest(res)
    (REPORTS / "results.md").write_text(results_md(res))


if __name__ == "__main__":
    main()
