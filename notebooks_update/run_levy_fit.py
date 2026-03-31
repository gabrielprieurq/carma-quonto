"""
Fit Gaussian and NIG distributions to timestamped training increments.
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
    LEVY_TEMP_JSON,
    LEVY_PRICE_JSON,
)
from carma_utils import fit_nig_mle, nig_logpdf


def _gaussian_fit(x: np.ndarray) -> dict:
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))
    ll = float(stats.norm.logpdf(x, loc=mu, scale=sigma).sum())
    n = len(x)
    return {
        "mu": mu,
        "sigma": sigma,
        "loglik": ll,
        "aic": float(-2.0 * ll + 2.0 * 2.0),
        "bic": float(-2.0 * ll + np.log(n) * 2.0),
    }


def _fit_increment_file(path: str, col: str, out_json: str) -> None:
    df = pd.read_csv(path, parse_dates=["timestamp_start", "timestamp_end"])
    train_end = pd.Timestamp(TRAIN_END_TIMESTAMP)
    train = df.loc[df["timestamp_end"] <= train_end].copy()
    x = train[col].to_numpy(dtype=float)

    gaussian = _gaussian_fit(x)
    nig = fit_nig_mle(x, verbose=False)
    ll_nig = float(nig_logpdf(x, nig["alpha"], nig["beta"], nig["mu"], nig["delta"]).sum())
    nig["loglik"] = ll_nig
    nig["bic"] = float(-2.0 * ll_nig + np.log(len(x)) * 4.0)

    payload = {
        "time_unit": "hour",
        "training_end_timestamp": TRAIN_END_TIMESTAMP,
        "n_obs": int(len(x)),
        "sample_start": str(train["timestamp_end"].min()),
        "sample_end": str(train["timestamp_end"].max()),
        "gaussian": gaussian,
        "nig": nig,
    }
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


def main() -> None:
    _fit_increment_file(TEMP_INC_CSV, "dL_temp", LEVY_TEMP_JSON)
    _fit_increment_file(PRICE_INC_CSV, "dL_price", LEVY_PRICE_JSON)


if __name__ == "__main__":
    main()
