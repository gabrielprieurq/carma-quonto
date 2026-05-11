# Numerical and Empirical Experiments: Remaining Work

This note tracks what is still left in the numerical code workflow. It replaces
the earlier generic experiment plan with the current repository state.

The intended canonical execution order is:

1. `run_preprocessing.py`
2. `run_marginal_carma.py`
3. `run_increment_recovery.py`
4. `run_levy_fit.py`
5. `run_coupling.py`
6. `run_pricing_validation.py`
7. `run_hedging_backtest.py`

## 1. Current verified state

Verified from `Code/notebooks/results/`:

- Present:
  - `preprocessing_summary.json`
  - `temp_seasonal_model.json`
  - `price_logseasonal_model.json`
  - `carma_model_selection.csv`
  - `full_pipeline.log`
  - legacy files `arma_selected.json`, `carma_init.json`
- Missing:
  - `carma_temp_selected.json`
  - `carma_price_selected.json`
  - `ou_temp_selected.json`
  - `ou_price_selected.json`
  - recovered-increment outputs
  - Levy-fit outputs
  - coupling outputs
  - pricing validation outputs
  - hedging backtest outputs

The latest verified log state is:

- preprocessing completed successfully,
- temperature CARMA screening completed,
- the screening step selected `CARMA(3,1)` for temperature on the 6-hour
  subsample,
- there is no evidence yet that the final hourly refit finished,
- there is no evidence yet that the price-side CARMA stage started or finished.

Conclusion: the numerical pipeline has not run through end to end yet.

## 2. Completed work

### 2.1 Data cleaning and preprocessing

Completed:

- raw price data were cleaned into an hourly series,
- mixed quarter-hour observations were aggregated to hourly means,
- deterministic seasonality/trend models were fit for temperature and shifted
  log-price,
- deseasonalized residual series were written,
- training and holdout split timestamps were fixed to
  `2024-12-31 23:00:00+00:00` and `2025-01-01 00:00:00+00:00`.

Main output:

- `Code/notebooks/results/preprocessing_summary.json`

### 2.2 Initial marginal-model screening

Completed:

- temperature model screening over the candidate CARMA orders,
- partial model-comparison table written to
  `Code/notebooks/results/carma_model_selection.csv`.

Current temperature screening rows:

1. `CARMA(1,0)`
2. `CARMA(2,0)`
3. `CARMA(3,0)`
4. `CARMA(3,1)`
5. `CARMA(2,1)`

Current screening winner:

- temperature `CARMA(3,1)` by AIC among the screened models.

Important limitation:

- this is only the screening stage, not the final selected fitted model object.

## 3. Work still left in the numerical code

### 3.1 Finish marginal CARMA fitting

This is the immediate blocker.

Still required:

1. finish the final full-hour refit for the selected temperature model,
2. run the price-side order screening,
3. run the final full-hour refit for the selected price model,
4. fit and save the OU benchmark models for both series,
5. save filtered-state outputs for both selected CARMA fits.

Expected outputs that should appear when this stage is complete:

- `Code/notebooks/results/carma_temp_selected.json`
- `Code/notebooks/results/carma_price_selected.json`
- `Code/notebooks/results/ou_temp_selected.json`
- `Code/notebooks/results/ou_price_selected.json`
- `Code/data/processed/temp_filtered_states.csv`
- `Code/data/processed/price_filtered_states.csv`

Nothing downstream should be treated as final before these files exist.

### 3.2 Recover latent increments

Still required:

1. recover model-implied increments from the selected CARMA fits,
2. preserve timestamps in the recovered series,
3. write aligned increment files for temperature and price.

This is needed before any Levy fit or coupling estimation can be trusted.

### 3.3 Fit innovation distributions

Still required:

1. fit the innovation law to the recovered increments,
2. write the final Gaussian and NIG parameter outputs,
3. confirm the hourly-to-year unit handling remains correct in the saved
   parameters,
4. compare Gaussian versus NIG fit quality.

This stage should produce the marginal heavy-tail calibration used by pricing
and hedging.

### 3.4 Estimate the coupling channel

Still required:

1. estimate the temperature-to-price coupling on timestamp-aligned increment
   series,
2. save the coupling parameter estimates,
3. validate the fitted lead-lag structure against the empirical cross-series
   diagnostics.

This stage is needed before any coupled quanto-pricing result can be reported.

### 3.5 Run pricing validation

Still required:

1. compare Fourier pricing against Monte Carlo,
2. save pricing comparisons and Monte Carlo confidence intervals,
3. verify that the implemented payoff mapping uses observed price levels rather
   than only the shifted latent log-price.

Expected output:

- a pricing-comparison result table under `Code/notebooks/results/`.

### 3.6 Run the hedging backtest

Still required:

1. run the fixed-expiry hedging backtest,
2. compare unhedged, OU, and coupled-CARMA strategies,
3. save hedging error summaries and pathwise backtest outputs.

Expected output:

- a hedging summary table under `Code/notebooks/results/`.

## 4. Figure generation still left

The current canonical pipeline mainly writes JSON and CSV outputs. It does not
yet generate the full figure set expected by the paper.

Still required:

1. add figure-export code to the canonical `run_*.py` scripts, or
2. add one dedicated post-processing script that reads the saved outputs and
   writes all paper figures to `Code/notebooks/figures/`.

At minimum, the production figure workflow still needs:

- preprocessing diagnostics figure,
- CARMA fit diagnostics figure,
- Levy-fit figure for the log-price innovations,
- coupling cross-correlation figure,
- pricing validation figure,
- hedging backtest figure.

Until those figures exist, the paper can compile only with placeholder graphics
or commented-out figure dependencies.

## 5. Paper integration still left

After the numerical pipeline has actually finished, the paper still needs:

1. the result tables in `empirical.tex` updated from the saved outputs,
2. the narrative claims in the introduction and conclusion aligned with the
   generated numbers,
3. placeholder `\tbd{...}` entries removed or replaced,
4. `graphicx` demo-mode removed once the real figures exist,
5. the missing bibliography entry `Bar97` fixed.

## 6. Execution checklist

The remaining work should be executed in this order:

1. finish `run_marginal_carma.py`,
2. run `run_increment_recovery.py`,
3. run `run_levy_fit.py`,
4. run `run_coupling.py`,
5. run `run_pricing_validation.py`,
6. run `run_hedging_backtest.py`,
7. generate/export figures,
8. update `empirical.tex` from the produced outputs,
9. remove paper compile workarounds.

## 7. Immediate next step

The immediate next step is still:

1. get `run_marginal_carma.py` to complete,
2. confirm the selected model JSON files are written,
3. only then continue to the increment, Levy, coupling, pricing, and hedging
   stages.

Right now the numerical work is blocked at the marginal-fit stage, not at
pricing or hedging.
