"""
Recover timestamped latent increments from the canonical filtered states.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CARMA_TEMP_MODEL_JSON,
    CARMA_PRICE_MODEL_JSON,
    TEMP_FILTERED_STATES_CSV,
    PRICE_FILTERED_STATES_CSV,
    TEMP_RESID_CSV,
    PRICE_LOGRESID_CSV,
    TEMP_INC_CSV,
    PRICE_INC_CSV,
    DT_YEARS,
)
from carma_utils import build_companion, load_params, recover_increments_exact


def _load_states(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _load_series(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0].astype(float)


def _recover(label: str, model_json: str, state_csv: str, series_csv: str, out_csv: str, increment_col: str) -> None:
    params = load_params(model_json)
    states = _load_states(state_csv)
    series = _load_series(series_csv)

    if not states.index.equals(series.index):
        raise ValueError(f"{label}: state and series timestamps do not match")

    x_cols = [c for c in states.columns if c.startswith("x_filt_")]
    x_filt = states[x_cols].to_numpy(dtype=float)

    a = np.asarray(params["a_coeffs"], dtype=float)
    A = build_companion(a)
    B = np.zeros((len(a), 1))
    B[-1, 0] = float(params["sigma"])

    increments = recover_increments_exact(A, B, x_filt, DT_YEARS)
    out = pd.DataFrame({
        "timestamp_start": states.index[:-1],
        "timestamp_end": states.index[1:],
        increment_col: increments,
    })
    out.to_csv(out_csv, index=False)
    print(f"{label}: saved {len(out):,} increments -> {out_csv}")


def main() -> None:
    _recover(
        "temperature",
        CARMA_TEMP_MODEL_JSON,
        TEMP_FILTERED_STATES_CSV,
        TEMP_RESID_CSV,
        TEMP_INC_CSV,
        "dL_temp",
    )
    _recover(
        "logprice",
        CARMA_PRICE_MODEL_JSON,
        PRICE_FILTERED_STATES_CSV,
        PRICE_LOGRESID_CSV,
        PRICE_INC_CSV,
        "dL_price",
    )


if __name__ == "__main__":
    main()
