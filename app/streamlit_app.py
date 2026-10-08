"""Interactive pricing scenarios.  Run:  streamlit run app/streamlit_app.py"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pricesim import simulate  # noqa: E402

st.set_page_config(page_title="Pricing scenario simulator", layout="wide")


@st.cache_data
def load():
    z = np.load(ROOT / "data/processed/simulation_inputs.npz", allow_pickle=True)
    res = json.loads((ROOT / "reports/results.json").read_text())
    return {k: z[k] for k in ("quarter_units", "base_price", "top", "elasticity_draws")}, res


inp, res = load()
q, p, top, eps_data = inp["quarter_units"].astype(float), inp["base_price"], inp["top"], inp["elasticity_draws"]

st.title("Pricing scenario simulator")
st.caption(f"{len(p):,} regular-selling products of a UK online wholesaler, next 13 weeks. "
           "Demand uncertainty and elasticity uncertainty are simulated together; every scenario is compared "
           "with current prices on the same draw.")

with st.sidebar:
    st.header("Scenario")
    d_top = st.slider("Price change, top 20% of products by revenue", -15, 15, 5, 1, format="%d%%") / 100
    d_rest = st.slider("Price change, other 80%", -15, 15, 5, 1, format="%d%%") / 100
    cost = st.slider("Unit cost as a share of current price", 0.30, 0.90, res["cost_ratio_assumption"], 0.05,
                     help="The data has no costs, so gross profit depends on this assumption.")
    st.header("Price response")
    source = st.radio("Elasticity", ["Estimated from the data", "Set my own"],
                      help="The estimate pools four posted-price definitions and rules out positive values.")
    if source == "Set my own":
        mu = st.number_input("Mean elasticity", -6.0, 0.0, -1.5, 0.1)
        sd = st.number_input("Uncertainty (standard deviation)", 0.0, 3.0, 0.5, 0.1)
    n = st.select_slider("Draws", [2000, 5000, 10000], 5000)
    seed = st.number_input("Seed", 0, 10_000, 7)

rng = np.random.default_rng(int(seed))
if source == "Set my own":
    eps = np.minimum(rng.normal(mu, sd, n), 0.0)
else:
    eps = rng.choice(eps_data, n)
sc = simulate.Scenario("custom", np.where(top, d_top, d_rest))
draws = simulate.run(q, p, eps, sc, cost, rng)
s = simulate.summarise("custom", draws)
be = simulate.breakeven_elasticity(d_top if d_top == d_rest and d_top != 0 else 0.05, cost)["gross_profit"]


def pct(x: float) -> str:
    """Percent that never rounds a non-certain probability to 0% or 100%."""
    if 0 < x < 0.005:
        return "<1%"
    if 0.995 < x < 1:
        return ">99%"
    return f"{x:.0%}"

k = lambda x: f"£{x/1e3:,.0f}k"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Median revenue change", k(s["delta_revenue_p50"]))
c1.caption(f"90% interval {k(s['delta_revenue_p05'])} to {k(s['delta_revenue_p95'])}")
c2.metric("Chance revenue rises", pct(s['p_revenue_up']))
c3.metric("Median gross-profit change", k(s["delta_gp_p50"]))
c3.caption(f"90% interval {k(s['delta_gp_p05'])} to {k(s['delta_gp_p95'])}")
c4.metric("Chance gross profit rises", pct(s['p_gp_up']))

left, right = st.columns(2)
for col, (name, a, b) in zip((left, right), [("Revenue change (£k)", "revenue", "revenue_base"),
                                              ("Gross-profit change (£k)", "gp", "gp_base")]):
    delta = (draws[a] - draws[b]) / 1e3
    counts, edges = np.histogram(delta, bins=40)
    col.subheader(name)
    col.bar_chart(pd.DataFrame({"draws": counts}, index=np.round((edges[:-1] + edges[1:]) / 2, 1)))

if d_top == d_rest and d_top > 0:
    st.info(f"A uniform {d_top:+.0%} rise raises gross profit unless elasticity is below {be:.2f} "
            f"(at {cost:.0%} cost). Share of elasticity draws below that: {(eps < be).mean():.1%}.")

with st.expander("What the numbers rest on"):
    st.markdown((ROOT / "reports/results.md").read_text())
