"""
Validate Fourier pricing against Monte Carlo on market-price payoffs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import (
    PRICE_SHIFT,
    TEST_START_TIMESTAMP,
    TEMP_FILTERED_STATES_CSV,
    PRICE_FILTERED_STATES_CSV,
    TEMP_SEASONAL_MODEL_JSON,
    PRICE_SEASONAL_MODEL_JSON,
    TEMP_HOURLY_CSV,
    PRICING_COMPARISON_CSV,
    CARMA_TEMP_MODEL_JSON,
    CARMA_PRICE_MODEL_JSON,
    LEVY_TEMP_JSON,
    LEVY_PRICE_JSON,
    COUPLING_JSON,
)
from carma_utils import simulate_coupled_carma
from pipeline_models import (
    load_state_frame,
    load_system,
    market_call_price,
    market_forward_price,
    market_indicator_quanto_price,
    market_spot_from_factor,
    seasonal_level,
    state_vector,
)


def _load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _current_actual_temperature(timestamp: pd.Timestamp) -> float:
    df = pd.read_csv(TEMP_HOURLY_CSV, index_col=0, parse_dates=True)
    return float(df.iloc[:, 0].loc[timestamp])


def main() -> None:
    _, A_X, B_X, c_X = load_system(CARMA_PRICE_MODEL_JSON)
    _, A_Y, B_Y, c_Y = load_system(CARMA_TEMP_MODEL_JSON)
    nig_X = _load_json(LEVY_PRICE_JSON)["nig"]
    nig_Y = _load_json(LEVY_TEMP_JSON)["nig"]
    gamma0 = float(_load_json(COUPLING_JSON)["gamma0"])
    Gamma = gamma0 * B_X

    x_states = load_state_frame(PRICE_FILTERED_STATES_CSV)
    y_states = load_state_frame(TEMP_FILTERED_STATES_CSV)
    t0 = pd.Timestamp(TEST_START_TIMESTAMP)
    z_x0 = state_vector(x_states, t0)
    z_y0 = state_vector(y_states, t0)
    temp_now_actual = _current_actual_temperature(t0)

    maturities = {
        "1w": pd.Timedelta(days=7),
        "1m": pd.Timedelta(days=30),
        "3m": pd.Timedelta(days=90),
    }
    rows: list[dict] = []
    rng = np.random.default_rng(42)
    n_paths = 25_000

    for label, delta in maturities.items():
        expiry = t0 + delta
        tau = delta / pd.Timedelta(hours=8760)
        lambda_x_T = seasonal_level(PRICE_SEASONAL_MODEL_JSON, expiry, "price_logfit")
        lambda_y_T = seasonal_level(TEMP_SEASONAL_MODEL_JSON, expiry, "temp_fit")

        strike_market = market_forward_price(
            tau,
            z_x0,
            A_X,
            c_X,
            B_X,
            A_Y,
            c_Y,
            B_Y,
            Gamma,
            nig_X,
            nig_Y,
            lambda_shifted_price=lambda_x_T,
            price_shift=PRICE_SHIFT,
        )
        threshold_y_actual = temp_now_actual

        call_fft = market_call_price(
            tau,
            z_x0,
            A_X,
            c_X,
            B_X,
            A_Y,
            c_Y,
            B_Y,
            Gamma,
            nig_X,
            nig_Y,
            strike_market=strike_market,
            lambda_shifted_price=lambda_x_T,
            price_shift=PRICE_SHIFT,
        )
        indicator_fft = market_indicator_quanto_price(
            tau,
            z_x0,
            z_y0,
            A_X,
            c_X,
            B_X,
            A_Y,
            c_Y,
            B_Y,
            Gamma,
            nig_X,
            nig_Y,
            strike_market=strike_market,
            threshold_secondary_actual=threshold_y_actual,
            lambda_shifted_price=lambda_x_T,
            lambda_secondary=lambda_y_T,
            price_shift=PRICE_SHIFT,
        )

        n_steps = max(int(delta / pd.Timedelta(days=1)), 1)
        dt = tau / n_steps
        x_paths, y_paths, _, _ = simulate_coupled_carma(
            A_X,
            B_X,
            c_X,
            A_Y,
            B_Y,
            c_Y,
            Gamma,
            n_steps=n_steps,
            dt=dt,
            nig_X=nig_X,
            nig_Y=nig_Y,
            n_paths=n_paths,
            z0_X=z_x0,
            z0_Y=z_y0,
            rng=rng,
        )
        s_T_market = market_spot_from_factor(x_paths[:, -1], lambda_x_T, PRICE_SHIFT)
        y_T_actual = lambda_y_T + y_paths[:, -1]

        call_payoff = np.maximum(s_T_market - strike_market, 0.0)
        indicator_payoff = (s_T_market > strike_market).astype(float) * (y_T_actual > threshold_y_actual).astype(float)

        rows.append({
            "valuation_timestamp": str(t0),
            "expiry_timestamp": str(expiry),
            "maturity": label,
            "tau_years": float(tau),
            "strike_market": float(strike_market),
            "threshold_y_actual": float(threshold_y_actual),
            "call_fft": float(call_fft),
            "call_mc": float(call_payoff.mean()),
            "call_mc_se": float(call_payoff.std(ddof=0) / np.sqrt(n_paths)),
            "indicator_fft": float(indicator_fft),
            "indicator_mc": float(indicator_payoff.mean()),
            "indicator_mc_se": float(indicator_payoff.std(ddof=0) / np.sqrt(n_paths)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(PRICING_COMPARISON_CSV, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
