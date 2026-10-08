"""Single place for dates, paths and run settings."""
from dataclasses import dataclass
from pathlib import Path
import datetime as dt

ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = ROOT / "sql"
DATA_DIR = ROOT / "data"
PROCESSED = DATA_DIR / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


@dataclass(frozen=True)
class Split:
    # model used to produce calibration residuals is trained on weeks before cal_start
    cal_start: dt.date = dt.date(2011, 6, 6)
    # final model, elasticity and simulation origin: everything before test_start
    test_start: dt.date = dt.date(2011, 9, 5)
    # 13 test weeks: 2011-09-05 .. 2011-11-28 (one quarter)
    horizon: int = 13


SPLIT = Split()
MIN_SELLING_WEEKS = 20      # products need this many selling weeks before test_start
REGULAR_WEEKS = 30          # "regular sellers": 30+ selling weeks before test_start; the rest are "occasional"
POSTED_PRICE_WINDOW = 4     # trailing window (selling weeks) for the posted-price mode
SEED = 7
