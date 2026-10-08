"""The identification argument, tested on data where the true elasticity is known."""
import numpy as np
import pandas as pd

from pricesim import elasticity as E

TRUE_BETA = -1.5


def synthetic_panel(seed=1, n_products=150, n_weeks=38):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2010-12-06", periods=n_weeks, freq="W-MON")
    week_fx = rng.normal(0, 0.2, n_weeks)
    rows = []
    for i in range(n_products):
        list_price = rng.uniform(0.5, 10)
        change_at = rng.integers(8, n_weeks - 8)
        new_price = list_price * rng.choice([0.85, 1.15, 1.25])
        a = rng.normal(3, 0.5)
        # half the products are occasional sellers: they sell in only 25 of the weeks
        selling = set(range(n_weeks)) if i % 2 == 0 else set(rng.choice(n_weeks, 25, replace=False))
        for t, w in enumerate(weeks):
            if t not in selling:
                continue
            lp = list_price if t < change_at else new_price
            big_order_week = rng.random() < 0.25
            units = np.exp(a + week_fx[t] + TRUE_BETA * np.log(lp) + rng.normal(0, 0.3))
            if big_order_week:      # bulk buyers arrive: many more units, paid at the cheaper tier
                units *= 2.5
            tier_price = lp * (0.85 if big_order_week else 1.0)
            avg_price = tier_price * (0.95 if big_order_week else 1.0)
            rows.append((f"p{i}", w, units, avg_price, tier_price))
    return pd.DataFrame(rows, columns=["sku", "week", "units", "avg_price", "tier_price"])


def test_naive_is_biased_and_pooled_posted_price_recovers_the_truth():
    """Known truth: -1.5. Order-mix weeks push the naive estimate to about -3.

    The posted price brackets the truth: short windows still let some tier flips through (too
    elastic), long windows lag real changes (attenuated). Pooling the windows, as the simulator
    does, lands close to the truth, slightly attenuated: on real data the simulator may therefore
    understate price sensitivity a little, which is stated in the README.
    """
    for seed in (1, 2, 3):
        panel = synthetic_panel(seed=seed)
        naive = E.comparison(panel, window=4)[0]
        assert naive.beta < TRUE_BETA - 1.0
        fits = E.segment_fits(panel)
        pooled = np.mean([f.coef for f in fits], axis=0)
        assert np.all(np.abs(pooled - TRUE_BETA) < 0.25)
        windows = np.array([f.coef for f in fits])
        assert windows.min() < TRUE_BETA < windows.max()     # the windows bracket the truth


def test_draws_pool_windows_and_match_their_fits():
    panel = synthetic_panel()
    fits = E.segment_fits(panel, windows=(3, 4))
    d = E.draw(fits, 20000, np.random.default_rng(0))
    assert d.shape == (20000, 2)
    for col in (0, 1):                                       # 0 = occasional, 1 = regular
        expected = np.mean([f.coef[col] for f in fits])
        assert abs(d[:, col].mean() - expected) < 0.05


def test_sign_restriction_removes_positive_draws_only():
    panel = synthetic_panel()
    fits = E.segment_fits(panel, windows=(4,))
    fits[0].coef = np.array([-0.1, 0.05])          # centre one segment near zero so positives occur
    rng = np.random.default_rng(3)
    d = E.draw(fits, 5000, rng)
    assert (d <= 0).all()
    d_raw = E.draw(fits, 5000, np.random.default_rng(3), nonpositive=False)
    assert (d_raw > 0).any()
