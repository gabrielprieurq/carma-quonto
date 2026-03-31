"""
Canonical preprocessing for the CARMA quanto pipeline.

This script replaces the exploratory deseasonalisation notebook with a
timestamp-safe, reproducible preprocessing stage.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from config import (
    PRICE_SHIFT,
    PRICE_START,
    PRICE_END,
    TEMP_START,
    TEMP_END,
    PRICES_CSV,
    TEMP_CSV,
    PRICES_HOURLY_CSV,
    TEMP_HOURLY_CSV,
    PRICE_LOGFIT_CSV,
    TEMP_FIT_CSV,
    PRICE_LOGRESID_CSV,
    TEMP_RESID_CSV,
    PRICE_RESID_TRAIN_CSV,
    PRICE_RESID_TEST_CSV,
    TEMP_RESID_TRAIN_CSV,
    TEMP_RESID_TEST_CSV,
    PRICE_SEASONAL_MODEL_JSON,
    TEMP_SEASONAL_MODEL_JSON,
    TRAIN_END_TIMESTAMP,
    TEST_START_TIMESTAMP,
    K_HOUR_PRICE,
    K_YEAR_PRICE,
    K_HOUR_TEMP,
    K_YEAR_TEMP,
)
from preprocess_utils import (
    SeasonalModel,
    fit_seasonal_model,
    load_hourly_series,
    split_series,
)


def _summary(label: str, series: pd.Series) -> dict:
    return {
        "label": label,
        "start": str(series.index.min()),
        "end": str(series.index.max()),
        "n_obs": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "skew": float(stats.skew(series)),
        "kurtosis_excess": float(stats.kurtosis(series, fisher=True)),
        "adf_stat": float(adfuller(series.dropna(), maxlag=48)[0]),
        "adf_pvalue": float(adfuller(series.dropna(), maxlag=48)[1]),
    }


def main() -> None:
    price_hourly, price_meta = load_hourly_series(PRICES_CSV, "price_eur_mwh")
    temp_hourly, temp_meta = load_hourly_series(TEMP_CSV, "temperature_c")

    price_hourly = price_hourly.loc[PRICE_START:PRICE_END].copy()
    temp_hourly = temp_hourly.loc[TEMP_START:TEMP_END].copy()

    price_hourly.name = "price_eur_mwh"
    temp_hourly.name = "temperature_c"

    if float(price_hourly.min() + PRICE_SHIFT) <= 0.0:
        raise ValueError("PRICE_SHIFT is too small for the observed minimum price")

    log_shifted_price = np.log(price_hourly + PRICE_SHIFT)
    log_shifted_price.name = "log_shifted_price"

    temp_train_raw, _ = split_series(
        temp_hourly,
        train_end_timestamp=TRAIN_END_TIMESTAMP,
        test_start_timestamp=TEST_START_TIMESTAMP,
    )
    price_train_raw, _ = split_series(
        log_shifted_price,
        train_end_timestamp=TRAIN_END_TIMESTAMP,
        test_start_timestamp=TEST_START_TIMESTAMP,
    )

    temp_model = fit_seasonal_model(
        temp_train_raw,
        k_hour=K_HOUR_TEMP,
        k_year=K_YEAR_TEMP,
        include_dow=False,
    )
    price_model = fit_seasonal_model(
        price_train_raw,
        k_hour=K_HOUR_PRICE,
        k_year=K_YEAR_PRICE,
        include_dow=True,
    )

    temp_fit = temp_model.evaluate(temp_hourly.index, name="temp_fit")
    price_logfit = price_model.evaluate(log_shifted_price.index, name="price_logfit")

    temp_resid = (temp_hourly - temp_fit).rename("temp_deseasoned")
    price_logresid = (log_shifted_price - price_logfit).rename("logprice_deseasoned")

    temp_train, temp_test = split_series(
        temp_resid,
        train_end_timestamp=TRAIN_END_TIMESTAMP,
        test_start_timestamp=TEST_START_TIMESTAMP,
    )
    price_train, price_test = split_series(
        price_logresid,
        train_end_timestamp=TRAIN_END_TIMESTAMP,
        test_start_timestamp=TEST_START_TIMESTAMP,
    )

    PRICES_HOURLY_CSV.parent.mkdir(parents=True, exist_ok=True)
    price_hourly.to_frame().to_csv(PRICES_HOURLY_CSV)
    temp_hourly.to_frame().to_csv(TEMP_HOURLY_CSV)

    temp_fit.to_frame().to_csv(TEMP_FIT_CSV)
    price_logfit.to_frame().to_csv(PRICE_LOGFIT_CSV)
    temp_resid.to_frame().to_csv(TEMP_RESID_CSV)
    price_logresid.to_frame().to_csv(PRICE_LOGRESID_CSV)
    temp_train.to_frame().to_csv(TEMP_RESID_TRAIN_CSV)
    temp_test.to_frame().to_csv(TEMP_RESID_TEST_CSV)
    price_train.to_frame().to_csv(PRICE_RESID_TRAIN_CSV)
    price_test.to_frame().to_csv(PRICE_RESID_TEST_CSV)

    temp_model.to_json(TEMP_SEASONAL_MODEL_JSON)
    price_model.to_json(PRICE_SEASONAL_MODEL_JSON)

    summary = {
        "price_hourly": price_meta,
        "temperature_hourly": temp_meta,
        "temp_residuals": _summary("temperature", temp_resid),
        "price_log_residuals": _summary("log_shifted_price", price_logresid),
        "train_end_timestamp": TRAIN_END_TIMESTAMP,
        "test_start_timestamp": TEST_START_TIMESTAMP,
        "price_shift": PRICE_SHIFT,
    }
    with open(TEMP_SEASONAL_MODEL_JSON.with_name("preprocessing_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
