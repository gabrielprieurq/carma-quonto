"""
Standalone script to run CARMA MLE fitting for notebooks 05 and 06.
Results are cached to JSON files; subsequent notebook runs will load from cache.
Run this with:  python run_carma_mle.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from config import (
    TEMP_RESID_TRAIN_CSV, PRICE_RESID_TRAIN_CSV,
    RES_DIR, HOURS_PER_YEAR, DT_YEARS,
    CARMA_ORDERS_TEMP, CARMA_ORDERS_PRICE,
)
from carma_utils import fit_carma_mle, save_params, load_params, arma_to_carma_init

RES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
temp_df  = pd.read_csv(TEMP_RESID_TRAIN_CSV,  index_col=0, parse_dates=True)
price_df = pd.read_csv(PRICE_RESID_TRAIN_CSV, index_col=0, parse_dates=True)
y_temp   = temp_df.iloc[:, 0].dropna().to_numpy(float)
y_price  = price_df.iloc[:, 0].dropna().to_numpy(float)
t_temp   = np.arange(len(y_temp),  dtype=float) * DT_YEARS
t_price  = np.arange(len(y_price), dtype=float) * DT_YEARS

# Load ARMA init
with open(RES_DIR / 'arma_selected.json') as f:
    arma_data = json.load(f)

with open(RES_DIR / 'carma_init.json') as f:
    init_data = json.load(f)

def run_fit(label, y, t_yr, p, q, theta0_list, out_path):
    if out_path.exists():
        print(f'[{label} CARMA({p},{q})] Loading cached result from {out_path.name}')
        return load_params(out_path)
    t0 = time.time()
    print(f'[{label} CARMA({p},{q})] Fitting ({len(theta0_list)} starts, {len(y):,} obs) ...')
    _, params = fit_carma_mle(t_yr, y, p, q, theta0_list, verbose=True)
    elapsed = time.time() - t0
    save_params(params, out_path)
    print(f'  => loglik={params["loglik"]:.3f}  AIC={params["aic"]:.1f}  [{elapsed:.0f}s]')
    return params

# ── Primary fits ─────────────────────────────────────────────────────────────
p_t = init_data['temperature']['p']; q_t = init_data['temperature']['q']
p_p = init_data['logprice']['p'];    q_p = init_data['logprice']['q']
theta0_temp  = [np.array(th) for th in init_data['temperature']['theta0_list']]
theta0_price = [np.array(th) for th in init_data['logprice']['theta0_list']]

params_temp  = run_fit('Temperature', y_temp,  t_temp,  p_t, q_t, theta0_temp,
                        RES_DIR / 'carma_temp_mle.json')
params_price = run_fit('Log-price',   y_price, t_price, p_p, q_p, theta0_price,
                        RES_DIR / 'carma_price_mle.json')

# ── Model comparison ─────────────────────────────────────────────────────────
COMP_CSV = RES_DIR / 'model_comparison.csv'
if COMP_CSV.exists():
    print('Model comparison CSV already exists, skipping.')
else:
    rows = []
    for label, y, t_yr, arma_key, orders in [
            ('Temperature', y_temp, t_temp, 'temperature', CARMA_ORDERS_TEMP),
            ('Log-price',   y_price, t_price, 'logprice',   CARMA_ORDERS_PRICE),
        ]:
        arma_pars = arma_data[arma_key]
        for p_c, q_c in orders:
            cache_path = RES_DIR / f'model_comp_{label.lower().replace("-","_")}_{p_c}_{q_c}.json'
            if cache_path.exists():
                pars = load_params(cache_path)
                rows.append({'series': label, 'p': p_c, 'q': q_c,
                             'loglik': pars['loglik'], 'AIC': pars['aic'], 'BIC': pars['bic']})
                print(f'  [{label} ({p_c},{q_c})] loaded from cache  AIC={pars["aic"]:.1f}')
                continue
            phi_init = arma_pars['phi'][:p_c] + [0.0]*max(0, p_c-len(arma_pars['phi']))
            th0_list = arma_to_carma_init(phi_init[:p_c], arma_pars['sigma2'], p_c, q_c, n_random=2)
            try:
                t0 = time.time()
                _, pars = fit_carma_mle(t_yr, y, p_c, q_c, th0_list, verbose=False)
                elapsed = time.time() - t0
                save_params(pars, cache_path)
                rows.append({'series': label, 'p': p_c, 'q': q_c,
                             'loglik': pars['loglik'], 'AIC': pars['aic'], 'BIC': pars['bic']})
                print(f'  [{label} ({p_c},{q_c})] AIC={pars["aic"]:.1f}  [{elapsed:.0f}s]')
            except Exception as e:
                rows.append({'series': label, 'p': p_c, 'q': q_c,
                             'loglik': float('nan'), 'AIC': float('nan'), 'BIC': float('nan')})
                print(f'  [{label} ({p_c},{q_c})] FAILED: {e}')
    
    df = pd.DataFrame(rows)
    df.to_csv(COMP_CSV, index=False)
    print(f'Saved model comparison -> {COMP_CSV}')

print('\nAll done.')
