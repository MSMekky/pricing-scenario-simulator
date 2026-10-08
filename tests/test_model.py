import numpy as np
import pandas as pd
import pytest

from pricesim import simulate
from pricesim.data import trailing_mode
from pricesim.forecast import Grid, growth_vectors, baseline_paths, naive_quarter, features, H, wape

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- posted price
def test_trailing_mode_ignores_one_week_flips():
    x = np.array([1.25, 1.25, 1.25, 0.85, 1.25, 1.25])
    assert np.all(trailing_mode(x, 4) == 1.25)


def test_trailing_mode_follows_a_persistent_change():
    x = np.array([1.25, 1.25, 1.25, 1.45, 1.45, 1.45, 1.45])
    out = trailing_mode(x, 4)
    assert out[2] == 1.25 and out[-1] == 1.45
    assert out[4] == 1.25   # 2-2 tie: the posted price is sticky until the new price has a majority
    assert out[5] == 1.45


def test_trailing_mode_tie_does_not_follow_a_big_order_week():
    # alternating flips create ties; a tie must never hand the price to the current (big-order) week
    x = np.array([1.25, 1.25, 0.85, 1.25, 0.85, 0.85, 1.25])
    out = trailing_mode(x, 4)
    assert np.all(out[:5] == 1.25)


def test_trailing_mode_uses_only_the_past():
    x = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
    full = trailing_mode(x, 3)
    for i in range(len(x)):
        assert trailing_mode(x[: i + 1], 3)[-1] == full[i]


# ---------------------------------------------------------------- simulator invariants
def _inputs(n_products=50, n_samples=200):
    q = RNG.gamma(2.0, 50.0, size=(n_samples, n_products))
    p = RNG.uniform(0.5, 10.0, size=n_products)
    return q, p


def test_no_price_change_means_no_change():
    q, p = _inputs()
    d = simulate.run(q, p, RNG.normal(-1, 0.5, 1000), simulate.uniform("0", 0.0, len(p)), 0.6, RNG)
    assert np.allclose(d.revenue, d.revenue_base) and np.allclose(d.gp, d.gp_base)


def test_unit_elastic_demand_keeps_revenue_constant():
    q, p = _inputs()
    d = simulate.run(q, p, np.full(500, -1.0), simulate.uniform("+10", 0.10, len(p)), 0.6, RNG)
    assert np.allclose(d.revenue, d.revenue_base)


def test_perfectly_inelastic_demand_passes_the_rise_straight_to_profit():
    q, p = _inputs()
    d = simulate.run(q, p, np.zeros(500), simulate.uniform("+5", 0.05, len(p)), 0.6, RNG)
    assert np.allclose(d.revenue / d.revenue_base, 1.05)
    assert np.allclose(d.gp - d.gp_base, 0.05 * d.revenue_base)


def test_breakeven_elasticity_is_exactly_breakeven():
    q, p = _inputs()
    for c in (0.4, 0.6, 0.8):
        e = simulate.breakeven_elasticity(0.05, c)["gross_profit"]
        d = simulate.run(q, p, np.full(200, e), simulate.uniform("+5", 0.05, len(p)), c, RNG)
        assert np.allclose(d.gp, d.gp_base)


def test_targeted_scenario_only_touches_its_products():
    q, p = _inputs()
    mask = np.zeros(len(p), bool)
    mask[:10] = True
    sc = simulate.targeted("t", 0.05, mask)
    assert np.all(sc.price_change[mask] == 0.05) and np.all(sc.price_change[~mask] == 0.0)


def test_summary_probabilities_are_consistent():
    q, p = _inputs()
    d = simulate.run(q, p, RNG.normal(-0.5, 0.5, 4000), simulate.uniform("+5", 0.05, len(p)), 0.6, RNG)
    s = simulate.summarise("x", d)
    assert 0.0 <= s["p_revenue_up"] <= 1.0
    assert s["delta_revenue_p05"] <= s["delta_revenue_p50"] <= s["delta_revenue_p95"]
    # with elasticity draws centred above -1 a rise should raise revenue more often than not
    assert s["p_revenue_up"] > 0.5


# ---------------------------------------------------------------- baseline and features
def _grid(T=40, N=30):
    units = RNG.poisson(5.0, size=(T, N)).astype(float)
    weeks = pd.date_range("2011-01-03", periods=T, freq="W-MON")
    return Grid(weeks, np.arange(N).astype(str), units, units * 2.0, np.zeros(N), np.full(N, 2.0))


def test_growth_vectors_reproduce_total_growth():
    g = _grid()
    vecs = growth_vectors(g, 40)
    for i, t in enumerate(range(H, 40 - H + 1)):
        prev, nxt = g.units[t - H:t].sum(0), g.units[t:t + H].sum(0)
        assert (prev * vecs[i]).sum() == pytest.approx(nxt.sum())


def test_baseline_never_negative_and_naive_is_last_quarter():
    g = _grid()
    s = baseline_paths(g, 40, 500, RNG)
    assert s.shape == (500, g.units.shape[1]) and (s >= 0).all()
    assert np.allclose(naive_quarter(g, 40), g.units[40 - H:40].sum(0))


def test_features_do_not_see_the_future():
    g = _grid()
    Y = np.log1p(g.units)
    t = 20
    f1 = features(Y, t, g.log_modal_price)
    Y2 = Y.copy()
    Y2[t:] = 99.0   # tamper with week t and later
    assert np.allclose(f1, features(Y2, t, g.log_modal_price))


def test_wape():
    assert wape(np.array([10.0, 10.0]), np.array([10.0, 10.0])) == 0.0
    assert wape(np.array([10.0, 10.0]), np.array([5.0, 15.0])) == pytest.approx(0.5)
