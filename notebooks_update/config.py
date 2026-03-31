"""
config.py  –  Project-wide constants for the CARMA quanto pipeline.

Import at the top of every notebook:
    from config import *
"""
from pathlib import Path

# ── Time units ───────────────────────────────────────────────────────────────
HOURS_PER_YEAR = 8760.0          # 1 year = 8760 hours
DT_YEARS       = 1.0 / HOURS_PER_YEAR   # one hourly step expressed in years
UTC            = "UTC"

# ── Price transformation ─────────────────────────────────────────────────────
# German day-ahead prices can be negative; we work with log(S_t + PRICE_SHIFT).
# The canonical transformed asset is
#   S_shifted(t) = S_market(t) + PRICE_SHIFT = exp(Lambda_S(t) + X_t).
# Downstream pricing on the observed market price must therefore map
#   K_market -> K_shifted = K_market + PRICE_SHIFT
# and
#   F_market = E[S_market(T) | F_t] = E[S_shifted(T) | F_t] - PRICE_SHIFT.
# PRICE_SHIFT must satisfy  min(prices) + PRICE_SHIFT > 0.
PRICE_SHIFT = 1000.0   # EUR/MWh

# ── Directory layout ─────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent            # Code/notebooks/
CODE_DIR   = _HERE
DATA_DIR   = _HERE.parent / "data"            # Code/data/
RAW_DIR    = DATA_DIR / "raw"
HOURLY_DIR = DATA_DIR / "hourly"
DESEAS_DIR = DATA_DIR / "deseasonalised"
INCR_DIR   = DATA_DIR / "increments"
RES_DIR    = _HERE / "results"
FIG_DIR    = _HERE / "figures"

for _d in (HOURLY_DIR, DESEAS_DIR, INCR_DIR, RES_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Raw data file paths ───────────────────────────────────────────────────────
PRICES_CSV = RAW_DIR / "prices.csv"           # column: price_eur_mwh, mixed hourly/15min in raw file
TEMP_CSV   = RAW_DIR / "temperature.csv"      # column: temperature_c, hourly
PRICES_HOURLY_CSV = HOURLY_DIR / "prices_hourly.csv"
TEMP_HOURLY_CSV   = HOURLY_DIR / "temperature_hourly.csv"

# ── Processed data file paths ─────────────────────────────────────────────────
TEMP_FIT_CSV          = DESEAS_DIR / "temp_fit.csv"
PRICE_LOGFIT_CSV      = DESEAS_DIR / "price_logfit.csv"
TEMP_RESID_CSV        = DESEAS_DIR / "temp_resid.csv"
PRICE_LOGRESID_CSV    = DESEAS_DIR / "price_logresid.csv"
TEMP_RESID_TRAIN_CSV  = DESEAS_DIR / "temp_resid_train.csv"
TEMP_RESID_TEST_CSV   = DESEAS_DIR / "temp_resid_test.csv"
PRICE_RESID_TRAIN_CSV = DESEAS_DIR / "price_logresid_train.csv"
PRICE_RESID_TEST_CSV  = DESEAS_DIR / "price_logresid_test.csv"
TEMP_SEASONAL_MODEL_JSON  = RES_DIR / "temp_seasonal_model.json"
PRICE_SEASONAL_MODEL_JSON = RES_DIR / "price_logseasonal_model.json"

TEMP_INC_CSV  = INCR_DIR / "temp_increments.csv"
PRICE_INC_CSV = INCR_DIR / "price_increments.csv"

# ── Data date ranges ──────────────────────────────────────────────────────────
PRICE_START = "2023-01-01"
PRICE_END   = "2025-12-31"
TEMP_START  = "2020-01-01"
TEMP_END    = "2025-12-31"

# ── Train / validation / test split ──────────────────────────────────────────
TRAIN_END            = "2024-12-31"
TEST_START           = "2025-01-01"
TRAIN_END_TIMESTAMP  = "2024-12-31 23:00:00+00:00"
TEST_START_TIMESTAMP = "2025-01-01 00:00:00+00:00"

# ── Seasonal regression: Fourier orders ──────────────────────────────────────
# Temperature: hourly series, two periodic components
K_HOUR_TEMP  = 3   # harmonics for 24-h intraday cycle
K_YEAR_TEMP  = 3   # harmonics for 8760-h annual cycle

# Log-price: hourly series, two periodic components + day-of-week dummies
K_HOUR_PRICE = 3
K_YEAR_PRICE = 3

# ── CARMA candidate model orders ─────────────────────────────────────────────
CARMA_ORDERS_TEMP  = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)]
CARMA_ORDERS_PRICE = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]

# ── MLE optimisation settings ─────────────────────────────────────────────────
MLE_METHOD   = "L-BFGS-B"
MLE_MAXITER  = 15_000
MLE_FTOL     = 1e-14
MLE_GTOL     = 1e-8
MLE_N_STARTS = 5      # number of multi-start attempts

# ── Kalman filter jitter (tiny observation noise for numerical stability) ─────
KF_JITTER = 1e-8

# ── Canonical model/result outputs ────────────────────────────────────────────
CARMA_SELECTION_CSV      = RES_DIR / "carma_model_selection.csv"
CARMA_TEMP_MODEL_JSON    = RES_DIR / "carma_temp_selected.json"
CARMA_PRICE_MODEL_JSON   = RES_DIR / "carma_price_selected.json"
OU_TEMP_MODEL_JSON       = RES_DIR / "ou_temp_selected.json"
OU_PRICE_MODEL_JSON      = RES_DIR / "ou_price_selected.json"
TEMP_FILTERED_STATES_CSV = RES_DIR / "temp_filtered_states.csv"
PRICE_FILTERED_STATES_CSV = RES_DIR / "price_filtered_states.csv"

LEVY_TEMP_JSON   = RES_DIR / "levy_params_temperature.json"
LEVY_PRICE_JSON  = RES_DIR / "levy_params_logprice.json"
COUPLING_JSON    = RES_DIR / "coupling_params.json"
PRICING_COMPARISON_CSV = RES_DIR / "pricing_comparison.csv"
HEDGING_BACKTEST_CSV   = RES_DIR / "hedging_backtest.csv"
HEDGING_SUMMARY_CSV    = RES_DIR / "hedging_summary.csv"
