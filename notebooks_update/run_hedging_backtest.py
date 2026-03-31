"""
Rolling fixed-expiry hedging backtest on the 2025 test window.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import (
    PRICE_SHIFT,
    TEST_START_TIMESTAMP,
    PRICES_HOURLY_CSV,
    TEMP_HOURLY_CSV,
    PRICE_LOGRESID_CSV,
    TEMP_RESID_CSV,
    PRICE_FILTERED_STATES_CSV,
    TEMP_FILTERED_STATES_CSV,
    PRICE_SEASONAL_MODEL_JSON,
    TEMP_SEASONAL_MODEL_JSON,
    CARMA_PRICE_MODEL_JSON,
    CARMA_TEMP_MODEL_JSON,
    OU_PRICE_MODEL_JSON,
    OU_TEMP_MODEL_JSON,
    LEVY_PRICE_JSON,
    LEVY_TEMP_JSON,
    COUPLING_JSON,
    HEDGING_BACKTEST_CSV,
    HEDGING_SUMMARY_CSV,
    DT_YEARS,
)
from carma_utils import kalman_filter
from pipeline_models import (
    build_system,
    load_state_frame,
    load_system,
    market_forward_price,
    market_indicator_hedge_ratio,
    market_indicator_quanto_price,
    seasonal_level,
    state_vector,
)


def _load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _load_series(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0].astype(float)


def _ou_state_frame(model_json: str, residual_csv: str) -> pd.DataFrame:
    params = _load_json(model_json)
    residual = _load_series(residual_csv)
    t_years = np.arange(len(residual), dtype=float) * DT_YEARS
    res = kalman_filter(
        t_years=t_years,
        y=residual.to_numpy(dtype=float),
        a_coeffs=params["a_coeffs"],
        b_coeffs=params["b_coeffs"],
        sigma=params["sigma"],
    )
    frame = pd.DataFrame(index=residual.index)
    frame["x_filt_1"] = res["x_filt"][:, 0]
    return frame


def _contract_grid(index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]]:
    starts = list(index[::24 * 7])
    contracts: list[tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]] = []
    for start in starts:
        expiry = start + pd.Timedelta(days=30)
        if expiry not in index:
            continue
        rehedge = [start + j * pd.Timedelta(days=7) for j in range(5)]
        rehedge = [ts for ts in rehedge if ts < expiry and ts in index]
        path = rehedge + [expiry]
        if len(path) >= 2:
            contracts.append((start, expiry, path))
    return contracts


def main() -> None:
    _, A_X, B_X, c_X = load_system(CARMA_PRICE_MODEL_JSON)
    _, A_Y, B_Y, c_Y = load_system(CARMA_TEMP_MODEL_JSON)
    ou_price_params = _load_json(OU_PRICE_MODEL_JSON)
    ou_temp_params = _load_json(OU_TEMP_MODEL_JSON)
    _, A_X_ou, B_X_ou, c_X_ou = load_system(OU_PRICE_MODEL_JSON)
    _, A_Y_ou, B_Y_ou, c_Y_ou = load_system(OU_TEMP_MODEL_JSON)

    nig_X = _load_json(LEVY_PRICE_JSON)["nig"]
    nig_Y = _load_json(LEVY_TEMP_JSON)["nig"]
    gamma0 = float(_load_json(COUPLING_JSON)["gamma0"])
    Gamma = gamma0 * B_X
    Gamma_ou = gamma0 * B_X_ou

    x_states = load_state_frame(PRICE_FILTERED_STATES_CSV)
    y_states = load_state_frame(TEMP_FILTERED_STATES_CSV)
    x_states_ou = _ou_state_frame(OU_PRICE_MODEL_JSON, PRICE_LOGRESID_CSV)
    y_states_ou = _ou_state_frame(OU_TEMP_MODEL_JSON, TEMP_RESID_CSV)

    price_actual = _load_series(PRICES_HOURLY_CSV)
    temp_actual = _load_series(TEMP_HOURLY_CSV)

    common_index = x_states.index
    for idx in [y_states.index, x_states_ou.index, y_states_ou.index, price_actual.index, temp_actual.index]:
        common_index = common_index.intersection(idx)
    common_index = common_index[common_index >= pd.Timestamp(TEST_START_TIMESTAMP)]
    contracts = _contract_grid(common_index)

    rows: list[dict] = []
    errors_unhedged: list[float] = []
    errors_carma: list[float] = []
    errors_ou: list[float] = []

    for contract_id, (start, expiry, path) in enumerate(contracts):
        z_x0 = state_vector(x_states, start)
        z_y0 = state_vector(y_states, start)
        z_x0_ou = state_vector(x_states_ou, start)
        z_y0_ou = state_vector(y_states_ou, start)

        lambda_x_T = seasonal_level(PRICE_SEASONAL_MODEL_JSON, expiry, "price_logfit")
        lambda_y_T = seasonal_level(TEMP_SEASONAL_MODEL_JSON, expiry, "temp_fit")

        tau0 = (expiry - start) / pd.Timedelta(hours=8760)
        strike_market = market_forward_price(
            tau0,
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
        threshold_y_actual = float(temp_actual.loc[start])

        premium_carma = market_indicator_quanto_price(
            tau0,
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
        premium_ou = market_indicator_quanto_price(
            tau0,
            z_x0_ou,
            z_y0_ou,
            A_X_ou,
            c_X_ou,
            B_X_ou,
            A_Y_ou,
            c_Y_ou,
            B_Y_ou,
            Gamma_ou,
            nig_X,
            nig_Y,
            strike_market=strike_market,
            threshold_secondary_actual=threshold_y_actual,
            lambda_shifted_price=lambda_x_T,
            lambda_secondary=lambda_y_T,
            price_shift=PRICE_SHIFT,
        )

        gains_carma = 0.0
        gains_ou = 0.0
        for t_j, t_next in zip(path[:-1], path[1:]):
            tau_j = (expiry - t_j) / pd.Timedelta(hours=8760)
            tau_next = (expiry - t_next) / pd.Timedelta(hours=8760)

            z_x_j = state_vector(x_states, t_j)
            z_y_j = state_vector(y_states, t_j)
            z_x_next = state_vector(x_states, t_next)
            z_x_j_ou = state_vector(x_states_ou, t_j)
            z_y_j_ou = state_vector(y_states_ou, t_j)
            z_x_next_ou = state_vector(x_states_ou, t_next)

            xi_carma = market_indicator_hedge_ratio(
                tau_j,
                z_x_j,
                z_y_j,
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
            xi_ou = market_indicator_hedge_ratio(
                tau_j,
                z_x_j_ou,
                z_y_j_ou,
                A_X_ou,
                c_X_ou,
                B_X_ou,
                A_Y_ou,
                c_Y_ou,
                B_Y_ou,
                Gamma_ou,
                nig_X,
                nig_Y,
                strike_market=strike_market,
                threshold_secondary_actual=threshold_y_actual,
                lambda_shifted_price=lambda_x_T,
                lambda_secondary=lambda_y_T,
                price_shift=PRICE_SHIFT,
            )

            F_j = market_forward_price(
                tau_j,
                z_x_j,
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
            F_next = market_forward_price(
                tau_next,
                z_x_next,
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
            F_j_ou = market_forward_price(
                tau_j,
                z_x_j_ou,
                A_X_ou,
                c_X_ou,
                B_X_ou,
                A_Y_ou,
                c_Y_ou,
                B_Y_ou,
                Gamma_ou,
                nig_X,
                nig_Y,
                lambda_shifted_price=lambda_x_T,
                price_shift=PRICE_SHIFT,
            )
            F_next_ou = market_forward_price(
                tau_next,
                z_x_next_ou,
                A_X_ou,
                c_X_ou,
                B_X_ou,
                A_Y_ou,
                c_Y_ou,
                B_Y_ou,
                Gamma_ou,
                nig_X,
                nig_Y,
                lambda_shifted_price=lambda_x_T,
                price_shift=PRICE_SHIFT,
            )
            gains_carma += xi_carma * (F_next - F_j)
            gains_ou += xi_ou * (F_next_ou - F_j_ou)

        payoff = float((price_actual.loc[expiry] > strike_market) and (temp_actual.loc[expiry] > threshold_y_actual))
        error_unhedged = premium_carma - payoff
        error_carma = premium_carma + gains_carma - payoff
        error_ou = premium_ou + gains_ou - payoff

        errors_unhedged.append(error_unhedged)
        errors_carma.append(error_carma)
        errors_ou.append(error_ou)
        rows.append({
            "contract_id": contract_id,
            "start_timestamp": str(start),
            "expiry_timestamp": str(expiry),
            "strike_market": float(strike_market),
            "threshold_y_actual": float(threshold_y_actual),
            "premium_unhedged": float(premium_carma),
            "premium_ou": float(premium_ou),
            "premium_carma": float(premium_carma),
            "payoff": payoff,
            "hedge_error_unhedged": float(error_unhedged),
            "hedge_error_ou": float(error_ou),
            "hedge_error_carma": float(error_carma),
        })

    df = pd.DataFrame(rows)
    df.to_csv(HEDGING_BACKTEST_CSV, index=False)

    var_unhedged = float(np.var(errors_unhedged))
    var_ou = float(np.var(errors_ou))
    var_carma = float(np.var(errors_carma))
    summary = pd.DataFrame([
        {
            "strategy": "unhedged",
            "n_contracts": len(df),
            "variance": var_unhedged,
            "rmse": float(np.sqrt(np.mean(np.square(errors_unhedged)))),
            "mean_error": float(np.mean(errors_unhedged)),
            "vrr": np.nan,
        },
        {
            "strategy": "ou",
            "n_contracts": len(df),
            "variance": var_ou,
            "rmse": float(np.sqrt(np.mean(np.square(errors_ou)))),
            "mean_error": float(np.mean(errors_ou)),
            "vrr": float(1.0 - var_ou / var_unhedged) if var_unhedged > 0 else np.nan,
        },
        {
            "strategy": "carma",
            "n_contracts": len(df),
            "variance": var_carma,
            "rmse": float(np.sqrt(np.mean(np.square(errors_carma)))),
            "mean_error": float(np.mean(errors_carma)),
            "vrr": float(1.0 - var_carma / var_unhedged) if var_unhedged > 0 else np.nan,
        },
    ])
    summary.to_csv(HEDGING_SUMMARY_CSV, index=False)

    print(df.head().to_string(index=False))
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
