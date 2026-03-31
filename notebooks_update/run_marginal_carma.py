"""
Canonical marginal CARMA fitting and model selection.
"""

from __future__ import annotations

import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import (
    DT_YEARS,
    TEMP_RESID_CSV,
    PRICE_LOGRESID_CSV,
    TEMP_RESID_TRAIN_CSV,
    PRICE_RESID_TRAIN_CSV,
    CARMA_ORDERS_TEMP,
    CARMA_ORDERS_PRICE,
    CARMA_SELECTION_CSV,
    CARMA_TEMP_MODEL_JSON,
    CARMA_PRICE_MODEL_JSON,
    OU_TEMP_MODEL_JSON,
    OU_PRICE_MODEL_JSON,
    TEMP_FILTERED_STATES_CSV,
    PRICE_FILTERED_STATES_CSV,
    MLE_MAXITER,
)
from carma_utils import (
    fit_carma_mle,
    kalman_filter,
    save_params,
    stable_carma_init,
)

GRID_N_STARTS = 2
GRID_MAXITER = min(MLE_MAXITER, 400)
FINAL_N_STARTS = 4
FINAL_MAXITER = min(MLE_MAXITER, 2_000)
OU_N_STARTS = 3
OU_MAXITER = min(MLE_MAXITER, 1_000)
GRID_WORKERS = max(1, min(2, os.cpu_count() or 1))
SCREENING_STEP_HOURS = 6


def _load_series(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0].astype(float)


def _diagnostics(std_innov: np.ndarray) -> dict:
    lb = acorr_ljungbox(std_innov, lags=[24, 48, 72], return_df=True)
    return {
        "lb_pvalue_24": float(lb["lb_pvalue"].iloc[0]),
        "lb_pvalue_48": float(lb["lb_pvalue"].iloc[1]),
        "lb_pvalue_72": float(lb["lb_pvalue"].iloc[2]),
        "innov_mean": float(np.mean(std_innov)),
        "innov_std": float(np.std(std_innov)),
    }


def _save_state_frame(path: str, index: pd.DatetimeIndex, res: dict) -> None:
    frame = pd.DataFrame(index=index)
    for i in range(res["x_filt"].shape[1]):
        frame[f"x_filt_{i+1}"] = res["x_filt"][:, i]
    frame["y_pred"] = res["y_pred"]
    frame["innov"] = res["innov"]
    frame["std_innov"] = res["std_innov"]
    frame.to_csv(path)


def _write_partial_selection(selection_path: str, label: str, rows: list[dict]) -> None:
    current = pd.DataFrame(rows).sort_values(["AIC", "BIC"]).reset_index(drop=True)
    if pd.io.common.file_exists(selection_path):
        existing = pd.read_csv(selection_path)
        existing = existing.loc[existing["series"] != label]
        current = pd.concat([existing, current], ignore_index=True)
    current.to_csv(selection_path, index=False)


def _fit_order_grid(
    label: str,
    full_series: pd.Series,
    train_series: pd.Series,
    orders: list[tuple[int, int]],
    selection_path: str,
) -> tuple[pd.DataFrame, dict]:
    y_train = train_series.to_numpy(dtype=float)
    t_train = np.arange(len(y_train), dtype=float) * DT_YEARS
    y_screen = y_train[::SCREENING_STEP_HOURS]
    t_screen = np.arange(len(y_screen), dtype=float) * DT_YEARS * SCREENING_STEP_HOURS
    y_full = full_series.to_numpy(dtype=float)
    t_full = np.arange(len(y_full), dtype=float) * DT_YEARS

    rows: list[dict] = []
    best_record: dict | None = None

    print(
        f"[{label}] screening {len(orders)} orders with {GRID_WORKERS} worker(s) "
        f"on a {SCREENING_STEP_HOURS}-hour subsample via discrete ARMA",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=GRID_WORKERS) as pool:
        future_map = {
            pool.submit(_screen_single_order, label, y_screen, t_screen, p, q): (p, q)
            for p, q in orders
        }
        for future in as_completed(future_map):
            p, q = future_map[future]
            row = future.result()
            rows.append(row)
            _write_partial_selection(selection_path, label, rows)
            print(
                f"[{label}] CARMA({p},{q}) done: "
                f"loglik={row['loglik']:.3f}, AIC={row['AIC']:.3f}, "
                f"LB24={row['lb_pvalue_24']:.4f}, success={row['optimizer_success']}",
                flush=True,
            )

    df = pd.DataFrame(rows).sort_values(["AIC", "BIC"]).reset_index(drop=True)
    passing = df[df["lb_pvalue_24"] > 0.05]
    selected_row = passing.iloc[0] if not passing.empty else df.iloc[0]
    p_sel = int(selected_row["p"])
    q_sel = int(selected_row["q"])
    print(f"[{label}] selected CARMA({p_sel},{q_sel}) for final refit", flush=True)

    theta0_list = stable_carma_init(
        y_train,
        p=p_sel,
        q=q_sel,
        n_starts=FINAL_N_STARTS,
        sigma2_arma=float(np.var(y_train)),
    )
    res_opt, params = fit_carma_mle(
        t_train,
        y_train,
        p=p_sel,
        q=q_sel,
        theta0_list=theta0_list,
        maxiter=FINAL_MAXITER,
        verbose=False,
    )
    kf_train = kalman_filter(
        t_years=t_train,
        y=y_train,
        a_coeffs=params["a_coeffs"],
        b_coeffs=params["b_coeffs"],
        sigma=params["sigma"],
    )
    kf_full = kalman_filter(
        t_years=t_full,
        y=y_full,
        a_coeffs=params["a_coeffs"],
        b_coeffs=params["b_coeffs"],
        sigma=params["sigma"],
    )
    params.update({
        "series": label,
        "estimation_method": "gaussian_qmle",
        "selected_by": "AIC subject to Ljung-Box(24) > 0.05",
        "optimizer_success": bool(res_opt.success),
        **_diagnostics(kf_train["std_innov"]),
    })

    best_record = {
        "params": params,
        "full_filter": kf_full,
        "train_filter": kf_train,
    }
    print(
        f"[{label}] final refit done: CARMA({p_sel},{q_sel}), "
        f"AIC={params['aic']:.3f}, LB24={params['lb_pvalue_24']:.4f}",
        flush=True,
    )
    return df, best_record


def _screen_single_order(
    label: str,
    y_train: np.ndarray,
    t_train: np.ndarray,
    p: int,
    q: int,
) -> dict:
    del t_train
    model = ARIMA(
        y_train,
        order=(p, 0, q),
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    res = model.fit(method_kwargs={"warn_convergence": False})
    resid = np.asarray(res.resid, dtype=float)
    diag = _diagnostics(resid[1:] if len(resid) > 1 else resid)
    return {
        "series": label,
        "p": p,
        "q": q,
        "screening_step_hours": SCREENING_STEP_HOURS,
        "screening_model": "arma",
        "loglik": float(res.llf),
        "AIC": float(res.aic),
        "BIC": float(res.bic),
        "optimizer_success": bool(res.mle_retvals.get("converged", True)),
        **diag,
    }


def _save_model_bundle(model_json: str, state_csv: str, index: pd.DatetimeIndex, bundle: dict) -> None:
    save_params(bundle["params"], model_json)
    _save_state_frame(state_csv, index, bundle["full_filter"])


def _fit_ou_model(label: str, train_series: pd.Series) -> dict:
    y_train = train_series.to_numpy(dtype=float)
    t_train = np.arange(len(y_train), dtype=float) * DT_YEARS
    theta0_list = stable_carma_init(
        y_train,
        p=1,
        q=0,
        n_starts=OU_N_STARTS,
        sigma2_arma=float(np.var(y_train)),
    )
    _, params = fit_carma_mle(
        t_train,
        y_train,
        p=1,
        q=0,
        theta0_list=theta0_list,
        maxiter=OU_MAXITER,
        verbose=False,
    )
    params.update({
        "series": label,
        "estimation_method": "gaussian_qmle",
        "benchmark": "OU",
    })
    return params


def main() -> None:
    temp_full = _load_series(TEMP_RESID_CSV)
    price_full = _load_series(PRICE_LOGRESID_CSV)
    temp_train = _load_series(TEMP_RESID_TRAIN_CSV)
    price_train = _load_series(PRICE_RESID_TRAIN_CSV)

    print("[temperature] starting model grid", flush=True)
    temp_grid, temp_bundle = _fit_order_grid(
        "temperature",
        temp_full,
        temp_train,
        CARMA_ORDERS_TEMP,
        str(CARMA_SELECTION_CSV),
    )
    print("[logprice] starting model grid", flush=True)
    price_grid, price_bundle = _fit_order_grid(
        "logprice",
        price_full,
        price_train,
        CARMA_ORDERS_PRICE,
        str(CARMA_SELECTION_CSV),
    )

    selection = pd.concat([temp_grid, price_grid], ignore_index=True)
    selection.to_csv(CARMA_SELECTION_CSV, index=False)

    _save_model_bundle(CARMA_TEMP_MODEL_JSON, TEMP_FILTERED_STATES_CSV, temp_full.index, temp_bundle)
    _save_model_bundle(CARMA_PRICE_MODEL_JSON, PRICE_FILTERED_STATES_CSV, price_full.index, price_bundle)

    save_params(_fit_ou_model("temperature", temp_train), OU_TEMP_MODEL_JSON)
    save_params(_fit_ou_model("logprice", price_train), OU_PRICE_MODEL_JSON)

    print("[marginal] saved selected CARMA and OU models", flush=True)
    print(selection.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(json.dumps({
        "temp_selected": temp_bundle["params"]["p"],
        "price_selected": price_bundle["params"]["p"],
        "selection_csv": str(CARMA_SELECTION_CSV),
    }, indent=2))


if __name__ == "__main__":
    main()
