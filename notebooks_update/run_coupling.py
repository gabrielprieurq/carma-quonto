"""
Estimate the contemporaneous coupling on timestamp-aligned training increments.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    TEMP_INC_CSV,
    PRICE_INC_CSV,
    TRAIN_END_TIMESTAMP,
    COUPLING_JSON,
)
from preprocess_utils import align_on_intersection, lagged_correlation


def _load_increment_series(path: str, value_col: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp_start", "timestamp_end"])
    df = df.loc[df["timestamp_end"] <= pd.Timestamp(TRAIN_END_TIMESTAMP)].copy()
    series = pd.Series(df[value_col].to_numpy(dtype=float), index=pd.to_datetime(df["timestamp_end"], utc=True))
    series.name = value_col
    return series


def main() -> None:
    dly = _load_increment_series(TEMP_INC_CSV, "dL_temp")
    dlx = _load_increment_series(PRICE_INC_CSV, "dL_price")
    dly, dlx = align_on_intersection(dly, dlx)

    gamma0 = float(np.dot(dly.to_numpy(), dlx.to_numpy()) / np.dot(dly.to_numpy(), dly.to_numpy()))
    eps = dlx - gamma0 * dly

    ss_res = float(np.dot(eps.to_numpy(), eps.to_numpy()))
    sigma_eps = np.sqrt(ss_res / max(len(dly) - 1, 1))
    se_gamma0 = float(sigma_eps / np.sqrt(np.dot(dly.to_numpy(), dly.to_numpy())))
    t_stat = float(gamma0 / se_gamma0)
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=max(len(dly) - 1, 1)))

    lags = np.arange(-72, 73)
    ccf = lagged_correlation(dly, dlx, lags)
    ci = float(1.96 / np.sqrt(len(dly)))

    payload = {
        "training_end_timestamp": TRAIN_END_TIMESTAMP,
        "n_obs": int(len(dly)),
        "sample_start": str(dly.index.min()),
        "sample_end": str(dly.index.max()),
        "gamma0": gamma0,
        "se_gamma0": se_gamma0,
        "t_stat": t_stat,
        "p_value": p_value,
        "lag0_correlation": float(ccf.loc[0]),
        "ccf_confidence_band": ci,
        "ccf_lags": lags.tolist(),
        "ccf_values": ccf.fillna(np.nan).tolist(),
        "residual_corr_with_driver": float(np.corrcoef(eps.to_numpy(), dly.to_numpy())[0, 1]),
    }
    with open(COUPLING_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
