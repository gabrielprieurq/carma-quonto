"""
Utilities for deterministic preprocessing and timestamp-safe alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fourier_block(x: np.ndarray, period: float, k_harmonics: int) -> np.ndarray:
    """Build cosine/sine Fourier columns for a scalar time grid."""
    omega = 2.0 * np.pi / period
    cols: list[np.ndarray] = []
    for k in range(1, k_harmonics + 1):
        cols.append(np.cos(k * omega * x))
        cols.append(np.sin(k * omega * x))
    return np.column_stack(cols) if cols else np.empty((len(x), 0))


def _hour_grid(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    return ((index - origin) / pd.Timedelta(hours=1)).to_numpy(dtype=float)


def load_hourly_series(
    path: str | Path,
    value_col: str,
    *,
    aggregation: str = "mean",
    interpolate_missing: bool = True,
) -> tuple[pd.Series, dict]:
    """
    Load a possibly irregular intraday series and convert it to a clean hourly series.

    If a given hour contains multiple observations, they are aggregated using
    `aggregation`. Any missing hourly slots are reintroduced and optionally
    interpolated in time.
    """
    path = Path(path)
    df = pd.read_csv(path, parse_dates=["datetime"])
    index = pd.to_datetime(df["datetime"], utc=True)
    values = pd.Series(df[value_col].to_numpy(dtype=float), index=index).sort_index()

    counts_per_hour = values.groupby(values.index.floor("h")).size().sort_index()
    hourly = getattr(values.groupby(values.index.floor("h")), aggregation)().sort_index()
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", tz="UTC")
    hourly = hourly.reindex(full_index)

    missing_hours = int(hourly.isna().sum())
    if interpolate_missing and missing_hours:
        hourly = hourly.interpolate(method="time").ffill().bfill()

    metadata = {
        "source_path": str(path),
        "value_col": value_col,
        "aggregation": aggregation,
        "n_raw_obs": int(len(values)),
        "n_hourly_obs": int(len(hourly)),
        "hours_with_single_obs": int((counts_per_hour == 1).sum()),
        "hours_with_multiple_obs": int((counts_per_hour > 1).sum()),
        "hours_with_four_obs": int((counts_per_hour == 4).sum()),
        "missing_hours_before_interpolation": missing_hours,
        "start": str(hourly.index.min()),
        "end": str(hourly.index.max()),
    }
    hourly.name = value_col
    return hourly, metadata


@dataclass
class SeasonalModel:
    origin_timestamp: str
    beta: list[float]
    k_hour: int
    k_year: int
    include_dow: bool

    def design_matrix(self, index: pd.DatetimeIndex) -> np.ndarray:
        origin = pd.Timestamp(self.origin_timestamp)
        x = _hour_grid(index, origin)
        cols: list[np.ndarray] = [np.ones(len(index)), x]

        day_block = fourier_block(x, 24.0, self.k_hour)
        year_block = fourier_block(x, 365.25 * 24.0, self.k_year)
        cols.extend([day_block, year_block])

        if day_block.size and year_block.size:
            inter_cols = [day_block[:, i] * year_block[:, j]
                          for i in range(day_block.shape[1])
                          for j in range(year_block.shape[1])]
            cols.append(np.column_stack(inter_cols))

        if self.include_dow:
            dow = index.dayofweek
            cols.append(np.column_stack([(dow == d).astype(float) for d in range(1, 7)]))

        return np.column_stack(cols)

    def evaluate(self, index: pd.DatetimeIndex, name: str = "seasonal_fit") -> pd.Series:
        beta = np.asarray(self.beta, dtype=float)
        fitted = self.design_matrix(index) @ beta
        return pd.Series(fitted, index=index, name=name)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({
                "origin_timestamp": self.origin_timestamp,
                "beta": self.beta,
                "k_hour": self.k_hour,
                "k_year": self.k_year,
                "include_dow": self.include_dow,
            }, fh, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "SeasonalModel":
        with open(path) as fh:
            payload = json.load(fh)
        return cls(**payload)


def fit_seasonal_model(
    series: pd.Series,
    *,
    k_hour: int,
    k_year: int,
    include_dow: bool,
) -> SeasonalModel:
    """Fit the deterministic seasonal regression used throughout the pipeline."""
    model = SeasonalModel(
        origin_timestamp=str(series.index[0]),
        beta=[],
        k_hour=k_hour,
        k_year=k_year,
        include_dow=include_dow,
    )
    design = model.design_matrix(series.index)
    beta, *_ = np.linalg.lstsq(design, series.to_numpy(dtype=float), rcond=None)
    model.beta = beta.tolist()
    return model


def split_series(
    series: pd.Series,
    *,
    train_end_timestamp: str,
    test_start_timestamp: str,
) -> tuple[pd.Series, pd.Series]:
    """Split a timestamp-indexed series using explicit UTC boundaries."""
    train_end = pd.Timestamp(train_end_timestamp)
    test_start = pd.Timestamp(test_start_timestamp)
    train = series.loc[series.index <= train_end].copy()
    test = series.loc[series.index >= test_start].copy()
    return train, test


def align_on_intersection(
    left: pd.Series,
    right: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align two timestamp-indexed series on the exact timestamp intersection."""
    common = left.index.intersection(right.index)
    return left.loc[common].copy(), right.loc[common].copy()


def lagged_correlation(
    x: pd.Series,
    y: pd.Series,
    lags: np.ndarray,
) -> pd.Series:
    """
    Compute Corr(x_t, y_{t+k}) for each lag k on aligned timestamped series.

    Positive k means y is shifted forward relative to x, so x leads y.
    """
    if not x.index.equals(y.index):
        raise ValueError("lagged_correlation expects aligned indices")

    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    out: list[float] = []
    for lag in lags:
        if lag == 0:
            xs = x_values
            ys = y_values
        elif lag > 0:
            xs = x_values[:-lag]
            ys = y_values[lag:]
        else:
            xs = x_values[-lag:]
            ys = y_values[:lag]
        if len(xs) < 3:
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(xs, ys)[0, 1]))
    return pd.Series(out, index=lags, name="correlation")
