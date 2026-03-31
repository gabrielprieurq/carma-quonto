"""
Shared model-loading and market-price wrappers for pricing/backtesting scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from carma_utils import (
    build_companion,
    compute_forward_price,
    compute_hedge_ratio_fd,
    fourier_price_1d,
    fourier_price_indicator_quanto,
    load_params,
)
from preprocess_utils import SeasonalModel


def build_system(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(params["a_coeffs"], dtype=float)
    b = np.asarray(params["b_coeffs"], dtype=float)
    A = build_companion(a)
    B = np.zeros((len(a), 1))
    B[-1, 0] = float(params["sigma"])
    c = np.zeros(len(a))
    c[:len(b)] = b
    return A, B, c


def load_system(model_json: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    params = load_params(model_json)
    A, B, c = build_system(params)
    return params, A, B, c


def load_state_frame(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def state_vector(frame: pd.DataFrame, timestamp: pd.Timestamp) -> np.ndarray:
    row = frame.loc[timestamp]
    cols = [c for c in frame.columns if c.startswith("x_filt_")]
    return row[cols].to_numpy(dtype=float)


def seasonal_level(model_json: str, timestamp: pd.Timestamp, name: str) -> float:
    model = SeasonalModel.from_json(model_json)
    index = pd.DatetimeIndex([timestamp])
    return float(model.evaluate(index, name=name).iloc[0])


def market_forward_price(
    tau: float,
    z_x: np.ndarray,
    A_X: np.ndarray,
    c_X: np.ndarray,
    B_X: np.ndarray,
    A_Y: np.ndarray,
    c_Y: np.ndarray,
    B_Y: np.ndarray,
    Gamma: np.ndarray,
    nig_X: dict,
    nig_Y: dict,
    *,
    lambda_shifted_price: float,
    price_shift: float,
) -> float:
    shifted_forward = compute_forward_price(
        tau,
        z_x,
        A_X,
        c_X,
        B_X,
        A_Y,
        c_Y,
        B_Y,
        Gamma,
        nig_X,
        nig_Y,
        Lambda_X=lambda_shifted_price,
    )
    return float(shifted_forward - price_shift)


def market_call_price(
    tau: float,
    z_x: np.ndarray,
    A_X: np.ndarray,
    c_X: np.ndarray,
    B_X: np.ndarray,
    A_Y: np.ndarray,
    c_Y: np.ndarray,
    B_Y: np.ndarray,
    Gamma: np.ndarray,
    nig_X: dict,
    nig_Y: dict,
    *,
    strike_market: float,
    lambda_shifted_price: float,
    price_shift: float,
) -> float:
    return fourier_price_1d(
        tau,
        z_x,
        A_X,
        c_X,
        B_X,
        A_Y,
        c_Y,
        B_Y,
        Gamma,
        nig_X,
        nig_Y,
        K=strike_market + price_shift,
        Lambda_X=lambda_shifted_price,
    )


def market_indicator_quanto_price(
    tau: float,
    z_x: np.ndarray,
    z_y: np.ndarray,
    A_X: np.ndarray,
    c_X: np.ndarray,
    B_X: np.ndarray,
    A_Y: np.ndarray,
    c_Y: np.ndarray,
    B_Y: np.ndarray,
    Gamma: np.ndarray,
    nig_X: dict,
    nig_Y: dict,
    *,
    strike_market: float,
    threshold_secondary_actual: float,
    lambda_shifted_price: float,
    lambda_secondary: float,
    price_shift: float,
) -> float:
    return fourier_price_indicator_quanto(
        tau,
        z_x,
        z_y,
        A_X,
        c_X,
        B_X,
        A_Y,
        c_Y,
        B_Y,
        Gamma,
        nig_X,
        nig_Y,
        K_S=strike_market + price_shift,
        K_Y=threshold_secondary_actual,
        Lambda_X=lambda_shifted_price,
        Lambda_Y=lambda_secondary,
    )


def market_indicator_hedge_ratio(
    tau: float,
    z_x: np.ndarray,
    z_y: np.ndarray,
    A_X: np.ndarray,
    c_X: np.ndarray,
    B_X: np.ndarray,
    A_Y: np.ndarray,
    c_Y: np.ndarray,
    B_Y: np.ndarray,
    Gamma: np.ndarray,
    nig_X: dict,
    nig_Y: dict,
    *,
    strike_market: float,
    threshold_secondary_actual: float,
    lambda_shifted_price: float,
    lambda_secondary: float,
    price_shift: float,
) -> float:
    return compute_hedge_ratio_fd(
        tau,
        z_x,
        z_y,
        A_X,
        c_X,
        B_X,
        A_Y,
        c_Y,
        B_Y,
        Gamma,
        nig_X,
        nig_Y,
        K_S=strike_market + price_shift,
        K_Y=threshold_secondary_actual,
        Lambda_X=lambda_shifted_price,
        Lambda_Y=lambda_secondary,
    )


def market_spot_from_factor(x: np.ndarray, lambda_shifted_price: float, price_shift: float) -> np.ndarray:
    return np.exp(lambda_shifted_price + x) - price_shift
