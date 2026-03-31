"""
carma_utils.py  –  Core utilities for the CARMA quanto pipeline.

Notation follows GK24.tex:
  State equation :  dZ = A Z dt + B dL_t
  Output         :  Y_t = c^T Z_t
  CARMA(p,q)     :  A = companion matrix (p×p),
                    c = [b0, …, bq, 0, …, 0]^T  with b_q = 1 (normalisation),
                    B = sigma * e_p  (Lévy noise enters the last state component)
  Time unit      :  YEARS throughout.  Hourly data must be divided by 8760
                    before being passed to any function here.
"""

from __future__ import annotations

import warnings
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import minimize

HOURS_PER_YEAR = 8760.0


# ─────────────────────────────────────────────────────────────────────────────
# 1.  State-space construction
# ─────────────────────────────────────────────────────────────────────────────

def build_companion(a: np.ndarray) -> np.ndarray:
    """
    Build the p×p companion matrix from AR coefficients a = [a1, …, ap].

    The characteristic polynomial of A is
        det(λ I - A) = λ^p + a1 λ^{p-1} + … + ap = 0,
    whose roots are the CARMA eigenvalues (all with strictly negative real
    part for a stable model).

    Layout:
        A[i, i+1] = 1   for i = 0, …, p-2
        A[-1, :]  = [-ap, -a_{p-1}, …, -a1]
    """
    a = np.asarray(a, dtype=float)
    p = len(a)
    A = np.zeros((p, p))
    for i in range(p - 1):
        A[i, i + 1] = 1.0
    A[-1, :] = -a[::-1]
    return A


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Discretisation and stationary covariance
# ─────────────────────────────────────────────────────────────────────────────

def van_loan_discretize(A: np.ndarray, B: np.ndarray, dt: float):
    """
    Exact zero-order-hold discretisation via the Van Loan (1978) method.

    Solves the block-matrix exponential
        exp( [[A,  B B^T], [0, -A^T]] * dt )
    to obtain the discrete transition F = exp(A dt) and the process-noise
    covariance Q_d = int_0^dt exp(As) B B^T exp(A^T s) ds.

    Parameters
    ----------
    A  : (p, p)  companion matrix (year^{-1} units)
    B  : (p, m)  noise-loading matrix
    dt : float   time step in years

    Returns
    -------
    F  : (p, p)  state-transition matrix
    Qd : (p, p)  discrete process-noise covariance
    """
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n]   =  A
    M[:n, n:]   =  B @ B.T
    M[n:, n:]   = -A.T
    E = expm(M * dt)
    F  = E[:n, :n]
    Qd = E[:n, n:] @ F.T
    # Symmetrise to prevent numerical drift
    Qd = 0.5 * (Qd + Qd.T)
    return F, Qd


def stationary_cov(A: np.ndarray, Q_cont: np.ndarray) -> np.ndarray:
    """
    Solve A P + P A^T + Q_cont = 0  →  return stationary state covariance P.

    Requires all eigenvalues of A to have strictly negative real parts.
    """
    return solve_continuous_lyapunov(A, -Q_cont)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Kalman filter
# ─────────────────────────────────────────────────────────────────────────────

def kalman_filter(
    t_years:  np.ndarray,
    y:        np.ndarray,
    a_coeffs: np.ndarray,
    b_coeffs: np.ndarray,
    sigma:    float,
    jitter:   float = 1e-8,
) -> dict:
    """
    Gaussian Kalman filter for a CARMA(p, q) model observed at times t_years.

    Parameters
    ----------
    t_years  : (n,) observation times in YEARS (equally or unequally spaced)
    y        : (n,) observations
    a_coeffs : (p,) AR companion-matrix coefficients [a1, …, ap]
    b_coeffs : (q+1,) output-vector entries [b0, …, bq]  (bq = 1 convention)
    sigma    : continuous noise scale  (year^{-1/2} units)
    jitter   : small observation-noise variance for numerical stability

    Returns
    -------
    dict with keys:
        loglik, x_filt, x_pred, P_pred, y_pred, innov, std_innov, S,
        A, B, H, F, Qd
    """
    t_years  = np.asarray(t_years,  dtype=float)
    y        = np.asarray(y,        dtype=float)
    n        = len(y)
    p        = len(a_coeffs)
    q1       = len(b_coeffs)   # = q + 1

    # ── Continuous-time matrices ──────────────────────────────────────────────
    A = build_companion(a_coeffs)                    # p × p
    B = np.zeros((p, 1)); B[-1, 0] = sigma           # p × 1
    c = np.zeros(p);      c[:q1]   = b_coeffs        # output vector
    H = c.reshape(1, -1)                             # 1 × p
    R = float(jitter)                                # scalar obs noise

    # ── Stationary initial covariance ────────────────────────────────────────
    Q_cont = B @ B.T
    try:
        P0 = stationary_cov(A, Q_cont)
        eigv = np.linalg.eigvalsh(P0)
        if np.any(eigv < -1e-10):
            raise ValueError("P0 not PSD")
        P0 = 0.5 * (P0 + P0.T)
    except Exception:
        P0 = np.eye(p) * 1e3   # fallback: diffuse prior

    # ── Precompute F, Qd for equal-spacing case ───────────────────────────────
    dt_vals = np.diff(t_years)
    if len(dt_vals) > 0 and np.allclose(dt_vals, dt_vals[0], rtol=1e-6, atol=1e-14):
        F, Qd      = van_loan_discretize(A, B, dt_vals[0])
        equal_dt   = True
    else:
        F, Qd      = None, None
        equal_dt   = False

    # ── Storage ───────────────────────────────────────────────────────────────
    x_pred_arr    = np.zeros((n, p))
    P_pred_arr    = np.zeros((n, p, p))
    x_filt_arr    = np.zeros((n, p))
    y_pred_arr    = np.zeros(n)
    innov_arr     = np.zeros(n)
    std_innov_arr = np.zeros(n)
    S_arr         = np.zeros(n)

    loglik = 0.0
    x_filt = np.zeros(p)   # initial state = 0  (mean-zero process)
    P_filt = P0.copy()

    for k in range(n):
        # ── Predict ──────────────────────────────────────────────────────────
        if k == 0:
            x_pred = x_filt.copy()
            P_pred = P_filt.copy()
        else:
            if not equal_dt:
                F, Qd = van_loan_discretize(A, B, dt_vals[k - 1])
            x_pred = F @ x_filt
            P_pred = F @ P_filt @ F.T + Qd

        # ── Innovation ───────────────────────────────────────────────────────
        mu = (H @ x_pred).item()
        S  = (H @ P_pred @ H.T).item() + R
        v  = y[k] - mu

        # ── Kalman gain ───────────────────────────────────────────────────────
        Kv   = (P_pred @ H.T) / S          # (p, 1)
        K    = Kv                           # alias for clarity

        # ── Joseph-form covariance update (numerically stable) ────────────────
        IKH    = np.eye(p) - K @ H
        x_filt = x_pred + K[:, 0] * v
        P_filt = IKH @ P_pred @ IKH.T + R * (K @ K.T)
        P_filt = 0.5 * (P_filt + P_filt.T)

        # ── Log-likelihood contribution ───────────────────────────────────────
        loglik += -0.5 * (np.log(2.0 * np.pi * S) + v ** 2 / S)

        # ── Store ─────────────────────────────────────────────────────────────
        x_pred_arr[k]    = x_pred
        P_pred_arr[k]    = P_pred
        x_filt_arr[k]    = x_filt
        y_pred_arr[k]    = mu
        innov_arr[k]     = v
        std_innov_arr[k] = v / np.sqrt(max(S, 1e-30))
        S_arr[k]         = S

    return {
        "loglik":     loglik,
        "x_filt":     x_filt_arr,
        "x_pred":     x_pred_arr,
        "P_pred":     P_pred_arr,
        "y_pred":     y_pred_arr,
        "innov":      innov_arr,
        "std_innov":  std_innov_arr,
        "S":          S_arr,
        "A": A, "B": B, "H": H,
        "F": F, "Qd": Qd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MLE objective and multi-start optimiser
# ─────────────────────────────────────────────────────────────────────────────

def _kalman_loglik_fast(
    y:       np.ndarray,
    F:       np.ndarray,
    H_row:   np.ndarray,
    Qd:      np.ndarray,
    R:       float,
    P0:      np.ndarray,
    n_transient: int = 50,
) -> float:
    """
    Fast Kalman log-likelihood for equal-spacing case.

    Runs a full P-update loop for n_transient steps to let the filter converge,
    then switches to the steady-state gain (solved via DARE) for the remaining
    steps.  For the steady-state phase the state recursion is vectorised using
    scipy.signal.dlsim, giving a ~10x speedup over the full loop.
    """
    from scipy.linalg import solve_discrete_are
    from scipy.signal import dlsim

    n  = len(y)
    p  = F.shape[0]
    H1 = H_row.ravel()           # (p,)

    # ── Steady-state predicted covariance via DARE ────────────────────────────
    try:
        P_ss = solve_discrete_are(F, H1[:, None], Qd, np.array([[R]]))
        S_ss = float(H1 @ P_ss @ H1) + R
        K_ss = (P_ss @ H1) / S_ss         # (p,) filter gain
        # closed-loop: x_{k+1|k} = Phi x_{k|k-1} + G y_k
        Phi  = F @ (np.eye(p) - np.outer(K_ss, H1))  # (p, p)
        Gvec = F @ K_ss                                # (p,)
        dare_ok = True
    except Exception:
        dare_ok = False

    # ── Transient phase (full Kalman with P update) ───────────────────────────
    x_pred = np.zeros(p)
    P_pred = P0.copy()
    ll = 0.0
    n_tr = min(n_transient, n) if dare_ok else n

    for k in range(n_tr):
        mu  = float(H1 @ x_pred)
        S_k = float(H1 @ P_pred @ H1) + R
        v   = y[k] - mu
        ll += -0.5 * (np.log(2.0 * np.pi * S_k) + v ** 2 / S_k)
        K_k    = P_pred @ H1 / S_k
        IKH    = np.eye(p) - np.outer(K_k, H1)
        P_filt = IKH @ P_pred @ IKH.T + R * np.outer(K_k, K_k)
        P_pred = F @ P_filt @ F.T + Qd
        x_pred = F @ (x_pred + K_k * v)

    if n_tr >= n or not dare_ok:
        return ll

    # ── Steady-state phase: vectorised dlsim ─────────────────────────────────
    # System:  x_{k+1} = Phi x_k + Gvec * y_k
    #          out_k   = H1 @ x_k   (predicted mean)
    y_rem  = y[n_tr:]
    _, y_ss, _ = dlsim(
        (Phi, Gvec[:, None], H1[None, :], np.zeros((1, 1)), 1),
        y_rem[:, None],
        x0=x_pred,
    )
    innov_ss = y_rem - y_ss.ravel()
    n_ss     = len(y_rem)
    ll      += -0.5 * n_ss * np.log(2.0 * np.pi * S_ss) \
               - 0.5 * np.sum(innov_ss ** 2) / S_ss
    return ll


def _unpack_theta(theta: np.ndarray, p: int, q: int):
    """
    Unpack the optimisation parameter vector into CARMA components.

    Layout:  theta = [a1, …, ap,  b0, …, b_{q-1},  log_sigma]
    Fixed:   b_q = 1  (normalisation)
    """
    a_coeffs  = theta[:p]
    b_free    = theta[p: p + q]          # q free MA params
    log_sigma = theta[p + q]
    b_coeffs  = np.append(b_free, 1.0)  # append fixed bq = 1
    sigma     = np.exp(log_sigma)
    return a_coeffs, b_coeffs, sigma


def neg_loglik_carma(
    theta:   np.ndarray,
    t_years: np.ndarray,
    y:       np.ndarray,
    p:       int,
    q:       int,
    jitter:  float = 1e-8,
) -> float:
    """Negative Kalman log-likelihood for CARMA(p, q). Used by scipy.optimize."""
    a_coeffs, b_coeffs, sigma = _unpack_theta(theta, p, q)

    # Stability constraint: all eigenvalues of A must have Re < 0
    A    = build_companion(a_coeffs)
    eigs = np.linalg.eigvals(A)
    if np.any(eigs.real >= -1e-8):
        return 1e12

    try:
        res = kalman_filter(t_years, y, a_coeffs, b_coeffs, sigma, jitter=jitter)
        ll  = res["loglik"]
        return float(-ll) if np.isfinite(ll) else 1e12
    except Exception:
        return 1e12


def fit_carma_mle(
    t_years:      np.ndarray,
    y:            np.ndarray,
    p:            int,
    q:            int,
    theta0_list:  list,
    method:       str   = "L-BFGS-B",
    maxiter:      int   = 15_000,
    ftol:         float = 1e-14,
    gtol:         float = 1e-8,
    jitter:       float = 1e-8,
    verbose:      bool  = True,
) -> tuple:
    """
    Fit CARMA(p, q) by maximum likelihood using the Kalman filter log-likelihood.

    Runs from every starting point in theta0_list and returns the best result.

    Parameters
    ----------
    theta0_list : list of (p+q+1,) arrays

    Returns
    -------
    best_res : scipy.OptimizeResult  (best optimisation result)
    params   : dict  with keys a_coeffs, b_coeffs, sigma, eigenvalues_*,
                               loglik, aic, bic, p, q, theta_opt
    """
    best_res = None
    best_ll  = -np.inf
    best_failed_res = None
    best_failed_ll = -np.inf

    for i, theta0 in enumerate(theta0_list):
        try:
            res = minimize(
                neg_loglik_carma,
                x0      = np.asarray(theta0, dtype=float),
                args    = (t_years, y, p, q, jitter),
                method  = method,
                options = {"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
            )
            ll = -float(res.fun)
            if verbose:
                tag = "OK " if res.success else "   "
                msg = res.message[:45] if not res.success else ""
                print(f"  start {i+1}/{len(theta0_list)}: loglik={ll:12.3f}  [{tag}] {msg}")
            if res.success and ll > best_ll:
                best_ll  = ll
                best_res = res
            elif (not res.success) and ll > best_failed_ll:
                best_failed_ll = ll
                best_failed_res = res
        except Exception as exc:
            if verbose:
                print(f"  start {i+1}/{len(theta0_list)}: FAILED  ({exc})")

    if best_res is None:
        if best_failed_res is None:
            raise RuntimeError("All starting points failed in fit_carma_mle.")
        best_res = best_failed_res
        best_ll = best_failed_ll

    a_opt, b_opt, sigma_opt = _unpack_theta(best_res.x, p, q)
    A_opt  = build_companion(a_opt)
    eigs   = np.linalg.eigvals(A_opt)
    n_obs  = len(y)

    params = {
        "a_coeffs":        a_opt.tolist(),
        "b_coeffs":        b_opt.tolist(),
        "sigma":           float(sigma_opt),
        "eigenvalues_real": eigs.real.tolist(),
        "eigenvalues_imag": eigs.imag.tolist(),
        "loglik":          float(best_ll),
        "aic":             float(-2 * best_ll + 2 * (p + q + 1)),
        "bic":             float(-2 * best_ll + np.log(n_obs) * (p + q + 1)),
        "n_obs":           int(n_obs),
        "p": p, "q": q,
        "theta_opt":       best_res.x.tolist(),
    }
    return best_res, params


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Initialisation from discrete ARMA parameters
# ─────────────────────────────────────────────────────────────────────────────

def arma_to_carma_init(
    phi_ar:      np.ndarray,
    sigma2_arma: float,
    p:           int,
    q:           int,
    hours_per_year: float = 8760.0,
    n_random:    int   = 4,
    rng_seed:    int   = 0,
) -> list:
    """
    Generate CARMA(p, q) MLE starting points from a discrete ARMA(p, q) fit.

    Steps
    -----
    1. Find roots of the ARMA AR polynomial  phi(z) = 1 - phi1 z - … - phip z^p.
    2. Convert each root to a CARMA eigenvalue: lambda = -log(root) / h  where
       h = 1/hours_per_year (each observation is one hour apart in years).
    3. Recover real AR coefficients via np.poly (exact for conjugate pairs).
    4. Scale sigma from ARMA to continuous-time units.
    5. Generate additional random perturbations for multi-start robustness.

    Parameters
    ----------
    phi_ar       : (p,) ARMA AR coefficients [phi1, …, phip]
    sigma2_arma  : ARMA innovation variance (from the discrete ARMA fit)
    p, q         : CARMA model orders
    hours_per_year : time-unit conversion factor (default 8760)
    n_random     : total number of starting points (1 ARMA-derived + n_random-1 random)

    Returns
    -------
    list of (p+q+1,) arrays suitable for neg_loglik_carma / fit_carma_mle
    """
    phi_ar = np.asarray(phi_ar, dtype=float)
    if len(phi_ar) != p:
        raise ValueError(f"Expected {p} AR coefficients, got {len(phi_ar)}.")

    # ── Step 1: roots of phi(z) = 1 - phi1*z - … - phip*z^p
    # np.roots expects descending powers: coefficients of -phip z^p - … - phi1 z + 1
    poly_coeffs = np.array([-phi_ar[p - 1 - i] for i in range(p)] + [1.0])
    arma_roots  = np.roots(poly_coeffs)

    # ── Step 2: CARMA eigenvalues in per-year units
    # lambda = -log(root) per hour, then multiply by hours_per_year
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lambda_yr = -np.log(arma_roots) * hours_per_year

    # ── Step 3: real AR coefficients from eigenvalues
    # np.poly([lam1,...,lamp]) gives [1, a1, ..., ap] such that
    # (z - lam1)...(z - lamp) = z^p + a1 z^{p-1} + ... + ap
    # Taking .real is exact when eigenvalues come in conjugate pairs.
    poly_from_eigs = np.poly(lambda_yr)
    a_coeffs_init  = np.real(poly_from_eigs[1:])   # [a1, …, ap]

    # ── Step 4: sigma initialisation
    # Rough heuristic: sigma^2 * dt ≈ sigma2_arma / ||b||^2
    # so sigma ≈ sqrt(sigma2_arma * hours_per_year)
    b_init    = np.append(np.ones(q), 1.0)          # placeholder b = [1,…,1]
    b_norm_sq = float(np.sum(b_init ** 2))
    sigma_sq  = sigma2_arma * hours_per_year / max(b_norm_sq, 1.0)
    sigma_init = float(np.sqrt(max(sigma_sq, 1e-4)))
    log_sig    = np.log(sigma_init)

    # b0 initialisation: start at 1 (will be refined by MLE)
    b_free_init = np.ones(q)

    theta0_arma = np.concatenate([a_coeffs_init, b_free_init, [log_sig]])

    # ── Step 5: random perturbations
    rng        = np.random.default_rng(rng_seed)
    theta_list = [theta0_arma]
    for _ in range(n_random - 1):
        perturb = rng.normal(0.0, 0.3, size=theta0_arma.shape)
        t_try   = theta0_arma + perturb
        # Keep b-free positive (any sign is valid but positive is a safe default)
        t_try[p: p + q] = np.abs(t_try[p: p + q]) + 0.1
        theta_list.append(t_try)

    return theta_list


def stable_carma_init(
    y: np.ndarray,
    p: int,
    q: int,
    *,
    hours_per_year: float = 8760.0,
    n_starts: int = 5,
    arma_phi: np.ndarray | None = None,
    sigma2_arma: float | None = None,
    rng_seed: int = 0,
) -> list[np.ndarray]:
    """
    Generate stable CARMA starting points without relying on invalid root logs.

    If discrete AR coefficients are available, only the root magnitudes are used
    to infer decay rates. This avoids introducing spurious complex frequencies
    from negative discrete roots, which do not map cleanly to a same-order real
    continuous-time CARMA model.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        raise ValueError("stable_carma_init requires at least 4 observations")

    rng = np.random.default_rng(rng_seed)
    y_scale = float(np.std(y))
    if sigma2_arma is not None and sigma2_arma > 0:
        sigma_guess = float(np.sqrt(sigma2_arma * hours_per_year / max(q + 1, 1)))
    else:
        sigma_guess = max(y_scale * np.sqrt(hours_per_year), 1e-2)

    rates: np.ndarray | None = None
    if arma_phi is not None:
        arma_phi = np.asarray(arma_phi, dtype=float)
        if len(arma_phi) == p:
            poly_coeffs = np.array([-arma_phi[p - 1 - i] for i in range(p)] + [1.0])
            roots = np.roots(poly_coeffs)
            mags = np.clip(np.abs(roots), 1.0 + 1e-6, None)
            rates = np.abs(np.log(mags)) * hours_per_year
            rates = np.clip(rates, 1.0, 24.0 * hours_per_year)

    if rates is None or np.any(~np.isfinite(rates)):
        rho1 = np.corrcoef(y[:-1], y[1:])[0, 1]
        rho1 = float(np.clip(abs(rho1), 0.05, 0.995))
        base_rate = max(-np.log(rho1) * hours_per_year, 1.0)
        rates = np.geomspace(base_rate / 2.0, base_rate * 8.0, num=p)

    rates = np.sort(np.asarray(rates, dtype=float))
    rates = np.maximum(rates, 1.0)
    base_eigs = -rates
    base_a = np.poly(base_eigs).real[1:]
    base_b = np.linspace(0.2, 1.0, q + 1)
    base_b[-1] = 1.0
    theta0 = np.concatenate([base_a, base_b[:-1], [np.log(max(sigma_guess, 1e-6))]])

    theta_list = [theta0]
    for _ in range(max(n_starts - 1, 0)):
        eig_perturb = base_eigs * np.exp(rng.normal(0.0, 0.25, size=p))
        a_try = np.poly(eig_perturb).real[1:]
        b_try = np.abs(base_b[:-1] + rng.normal(0.0, 0.25, size=q)) + 0.05
        log_sigma_try = np.log(max(sigma_guess * np.exp(rng.normal(0.0, 0.35)), 1e-6))
        theta_list.append(np.concatenate([a_try, b_try, [log_sigma_try]]))
    return theta_list


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Lévy increment recovery
# ─────────────────────────────────────────────────────────────────────────────

def recover_increments_exact(
    A:       np.ndarray,
    B:       np.ndarray,
    x_filt:  np.ndarray,
    dt_years: float,
) -> np.ndarray:
    """
    Back-recover the driving Lévy increments from Kalman-filtered states.

    The exact discrete-time state equation is
        Z_{k+1} = F Z_k  +  B_load * dL_k
    where  F = exp(A dt)  and  B_load = A^{-1} (F - I) B.

    Inverting:  B_load * dL_k ≈ Z_{k+1} - F Z_k  (exact up to filter noise).
    The least-squares solution  dL_k = (B_load^T B_load)^{-1} B_load^T dZ_k
    is scalar for the standard CARMA case with B = sigma * e_p.

    Parameters
    ----------
    A        : (p, p)  companion matrix (year^{-1} units)
    B        : (p, 1)  noise-loading matrix
    x_filt   : (n, p)  filtered state array from kalman_filter
    dt_years : float   time step in years

    Returns
    -------
    dL : (n-1,) array of recovered (scaled) Lévy increments
    """
    p = A.shape[0]
    F = expm(A * dt_years)

    # B_load = A^{-1} (F - I) B  =  int_0^{dt} exp(As) B ds
    try:
        B_load = np.linalg.solve(A, (F - np.eye(p)) @ B)   # (p, 1)
    except np.linalg.LinAlgError:
        # Near-singular A: fall back to dt * B (Euler approximation)
        B_load = dt_years * B

    # dZ_k = Z_{k+1} - F Z_k
    dZ = x_filt[1:] - (F @ x_filt[:-1].T).T   # (n-1, p)

    # Least-squares recovery: dL_k = B_flat^T dZ_k / ||B_flat||^2
    B_flat = B_load[:, 0]              # (p,)
    b_sq   = float(np.dot(B_flat, B_flat))
    if b_sq < 1e-14:
        raise ValueError("B_load is near-zero; check CARMA parameters.")
    dL = dZ @ B_flat / b_sq            # (n-1,)
    return dL


# ─────────────────────────────────────────────────────────────────────────────
# 7.  NIG distribution helpers  (Python-only, no R required)
# ─────────────────────────────────────────────────────────────────────────────

def nig_logpdf(x: np.ndarray, alpha: float, beta: float,
               mu: float, delta: float) -> np.ndarray:
    """
    Log-PDF of the Normal Inverse Gaussian distribution.

    Parameterisation: NIG(alpha, beta, mu, delta)
        alpha > 0   (tail heaviness;  alpha_bar = alpha * delta)
        |beta| < alpha  (asymmetry)
        mu           (location)
        delta > 0    (scale)

    PDF:
        f(x) = (alpha delta / pi) * K1(alpha*sqrt(delta^2 + (x-mu)^2))
               / sqrt(delta^2 + (x-mu)^2)
               * exp(delta*sqrt(alpha^2 - beta^2) + beta*(x - mu))
    where K1 is the modified Bessel function of the second kind, order 1.
    """
    from scipy.special import k1

    z   = x - mu
    q   = np.sqrt(delta ** 2 + z ** 2)
    gam = np.sqrt(alpha ** 2 - beta ** 2)

    log_f = (
        np.log(alpha * delta / np.pi)
        + np.log(k1(alpha * q))
        - np.log(q)
        + delta * gam
        + beta * z
    )
    return log_f


def fit_nig_mle(data: np.ndarray, verbose: bool = True) -> dict:
    """
    Fit a NIG distribution to data by maximum likelihood.

    Returns
    -------
    dict with keys: alpha, beta, mu, delta, loglik, aic
    """
    data = np.asarray(data, dtype=float)
    n    = len(data)

    def neg_ll(params):
        alpha, beta_raw, mu, log_delta = params
        delta = np.exp(log_delta)
        # Constraint: alpha > |beta|
        beta = np.tanh(beta_raw) * (alpha - 1e-6)   # keeps |beta| < alpha
        try:
            ll = np.sum(nig_logpdf(data, alpha, beta, mu, delta))
            return -ll if np.isfinite(ll) else 1e12
        except Exception:
            return 1e12

    # Starting point from method-of-moments
    m1, m2, m3, m4 = (
        np.mean(data),
        np.var(data),
        float(np.mean((data - np.mean(data)) ** 3) / m2 ** 1.5),   # skew
        float(np.mean((data - np.mean(data)) ** 4) / m2 ** 2),     # kurtosis
    )
    # Approximate MOM starting values for NIG
    kurt_excess = max(m4 - 3.0, 0.1)
    alpha0 = 3.0 / np.sqrt(m2 * kurt_excess + 1e-6)
    theta0 = np.array([max(alpha0, 0.5), 0.0, m1, np.log(max(np.sqrt(m2), 0.01))])

    rng = np.random.default_rng(42)
    best_res = None
    best_ll  = -np.inf
    for _ in range(8):
        try:
            res = minimize(neg_ll, theta0 + rng.normal(0, 0.2, 4),
                           method="Nelder-Mead",
                           options={"maxiter": 20_000, "xatol": 1e-8, "fatol": 1e-8})
            ll = -res.fun
            if ll > best_ll:
                best_ll  = ll
                best_res = res
        except Exception:
            pass
    if best_res is None:
        raise RuntimeError("NIG MLE failed.")

    alpha_opt, beta_raw_opt, mu_opt, log_delta_opt = best_res.x
    delta_opt = np.exp(log_delta_opt)
    beta_opt  = np.tanh(beta_raw_opt) * (alpha_opt - 1e-6)

    result = {
        "alpha":  float(alpha_opt),
        "beta":   float(beta_opt),
        "mu":     float(mu_opt),
        "delta":  float(delta_opt),
        "loglik": float(best_ll),
        "aic":    float(-2 * best_ll + 2 * 4),
        "n_obs":  int(n),
    }
    if verbose:
        print(f"NIG fit:  alpha={alpha_opt:.4f}  beta={beta_opt:.4f}"
              f"  mu={mu_opt:.4f}  delta={delta_opt:.4f}"
              f"  loglik={best_ll:.2f}  AIC={result['aic']:.2f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Simulation (for Monte Carlo pricing in Phase 8)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_carma(
    A: np.ndarray,
    B: np.ndarray,
    c: np.ndarray,
    n_steps: int,
    dt_years: float,
    levy_sampler,
    n_paths: int = 1,
    z0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Simulate CARMA paths using exact matrix-exponential discretisation.

    Parameters
    ----------
    A            : (p, p) companion matrix (year^{-1})
    B            : (p, 1) noise loading
    c            : (p,)   output vector
    n_steps      : number of time steps
    dt_years     : step size in years
    levy_sampler : callable(n_paths, rng) → (n_paths,) increments per step
    n_paths      : Monte Carlo paths
    z0           : (p,) initial state (default: zeros)
    rng          : numpy Generator

    Returns
    -------
    Y : (n_paths, n_steps+1) simulated output values
    Z : (n_paths, n_steps+1, p) simulated states
    """
    if rng is None:
        rng = np.random.default_rng()
    p = A.shape[0]
    F = expm(A * dt_years)

    # B_load = A^{-1} (F - I) B
    try:
        B_load = np.linalg.solve(A, (F - np.eye(p)) @ B)
    except np.linalg.LinAlgError:
        B_load = dt_years * B

    Z = np.zeros((n_paths, n_steps + 1, p))
    if z0 is not None:
        Z[:, 0, :] = z0

    for k in range(n_steps):
        dL          = levy_sampler(n_paths, rng)          # (n_paths,)
        noise       = (B_load[:, 0] * dL[:, None]).T      # broadcasting: (p, n_paths)^T -> (n_paths, p)
        # Correct shape: noise[i, :] = B_load[:, 0] * dL[i]
        noise       = B_load[:, 0][None, :] * dL[:, None] # (n_paths, p)
        Z[:, k+1, :] = (F @ Z[:, k, :].T).T + noise

    Y = (Z @ c)   # (n_paths, n_steps+1)
    return Y, Z


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Pricing and simulation helpers (for notebooks 09 and 10)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_inverse_gaussian(
    mu: float | np.ndarray,
    lam: float | np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from the Inverse Gaussian IG(mu, lambda) distribution.

    Uses the Michael–Johnson–Haas algorithm (1976):
      1. Draw Y ~ N(0,1), set x = mu + mu²y²/(2λ) - mu/(2λ)√(4μλy² + μ²y⁴)
      2. Accept x with prob mu/(mu+x), else return mu²/x.
    """
    y  = rng.standard_normal(size)
    y2 = y ** 2
    mu = np.asarray(mu, dtype=float)
    lam = np.asarray(lam, dtype=float)
    x  = (mu + mu ** 2 * y2 / (2 * lam)
          - mu / (2 * lam) * np.sqrt(4 * mu * lam * y2 + mu ** 2 * y2 ** 2))
    u  = rng.uniform(size=size)
    flip = u >= mu / (mu + x)
    x[flip] = mu ** 2 / x[flip]
    return x


def nig_sample(
    alpha: float,
    beta:  float,
    mu:    float,
    delta: float,
    n:     int,
    rng:   np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw n independent samples from NIG(alpha, beta, mu, delta).

    Uses the normal variance-mean mixture representation:
        X = mu + beta * V + sqrt(V) * Z,
        V ~ InvGauss(delta/gamma, delta^2),  Z ~ N(0,1),
    where gamma = sqrt(alpha^2 - beta^2).
    """
    if rng is None:
        rng = np.random.default_rng()
    gamma = np.sqrt(alpha ** 2 - beta ** 2)
    V = _sample_inverse_gaussian(delta / gamma, delta ** 2, n, rng)
    Z = rng.standard_normal(n)
    return mu + beta * V + np.sqrt(V) * Z


def _nig_params_per_year(nig_params: dict) -> tuple[float, float, float, float]:
    """
    Convert hourly-fitted NIG parameters into year-based Lévy parameters.

    Notebook 07 fits NIG directly to recovered hourly increments. For an NIG
    Lévy process, alpha and beta stay fixed across time scales, while mu and
    delta scale linearly with time.
    """
    return (
        float(nig_params["alpha"]),
        float(nig_params["beta"]),
        float(nig_params["mu"]) * HOURS_PER_YEAR,
        float(nig_params["delta"]) * HOURS_PER_YEAR,
    )


def _nig_params_for_step(
    nig_params: dict,
    dt_years: float,
) -> tuple[float, float, float, float]:
    """Scale hourly-fitted NIG parameters to a step of length dt_years."""
    dt_hours = float(dt_years) * HOURS_PER_YEAR
    return (
        float(nig_params["alpha"]),
        float(nig_params["beta"]),
        float(nig_params["mu"]) * dt_hours,
        float(nig_params["delta"]) * dt_hours,
    )


def nig_char_exponent(
    u:     complex | np.ndarray,
    alpha: float,
    beta:  float,
    mu:    float,
    delta: float,
) -> complex | np.ndarray:
    """
    Characteristic exponent of NIG(alpha, beta, mu, delta):

        psi(u) = log E[exp(i u L_1)]
               = i mu u  +  delta * (gamma - sqrt(alpha^2 - (beta + iu)^2))

    where gamma = sqrt(alpha^2 - beta^2).

    Valid for all real u. The principal branch of sqrt is used (Re(sqrt) >= 0).
    """
    gamma = np.sqrt(alpha ** 2 - beta ** 2)
    return 1j * mu * u + delta * (gamma - np.sqrt(alpha ** 2 - (beta + 1j * u) ** 2))


def nig_cumulant_exponent(
    theta: float | np.ndarray,
    alpha: float,
    beta:  float,
    mu:    float,
    delta: float,
) -> float | np.ndarray:
    """
    Lévy cumulant (Laplace exponent on the real line):

        kappa(theta) = log E[exp(theta L_1)]
                     = mu theta  +  delta * (gamma - sqrt(alpha^2 - (beta+theta)^2))

    Finite for |beta + theta| < alpha.
    """
    gamma = np.sqrt(alpha ** 2 - beta ** 2)
    return mu * theta + delta * (gamma - np.sqrt(alpha ** 2 - (beta + theta) ** 2))


def _kernel_integrals(
    tau:     float,
    A_X:     np.ndarray,
    c_X:     np.ndarray,
    B_X:     np.ndarray,
    A_Y:     np.ndarray,
    c_Y:     np.ndarray,
    B_Y:     np.ndarray,
    Gamma:   np.ndarray,
    n_quad:  int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gauss-Legendre quadrature nodes (t_k) and kernel vectors evaluated at
    s-values in [0, tau], where s = tau - (T - u), i.e. the integration variable.

    Returns
    -------
    s_nodes : (n_quad,)  quadrature points in [0, tau]
    w_nodes : (n_quad,)  quadrature weights
    beta_X  : (n_quad,)  B_X^T exp(A_X^T (tau-s)) c_X
    gamma_X : (n_quad,)  Gamma^T exp(A_X^T (tau-s)) c_X
    beta_Y  : (n_quad,)  B_Y^T exp(A_Y^T (tau-s)) c_Y
    """
    from numpy.polynomial.legendre import leggauss
    xi, w = leggauss(n_quad)              # on [-1, 1]
    s_nodes = 0.5 * tau * (xi + 1)       # map to [0, tau]
    w_nodes = 0.5 * tau * w

    beta_X  = np.array([
        float(B_X.ravel() @ expm(A_X.T * (tau - s)) @ c_X)
        for s in s_nodes
    ])
    gamma_X = np.array([
        float(Gamma.ravel() @ expm(A_X.T * (tau - s)) @ c_X)
        for s in s_nodes
    ])
    beta_Y  = np.array([
        float(B_Y.ravel() @ expm(A_Y.T * (tau - s)) @ c_Y)
        for s in s_nodes
    ])
    return s_nodes, w_nodes, beta_X, gamma_X, beta_Y


def carma_joint_cf(
    u:          float | complex,
    v:          float | complex,
    tau:        float,
    Z_X:        np.ndarray,
    Z_Y:        np.ndarray,
    A_X:        np.ndarray,
    c_X:        np.ndarray,
    B_X:        np.ndarray,
    A_Y:        np.ndarray,
    c_Y:        np.ndarray,
    B_Y:        np.ndarray,
    Gamma:      np.ndarray,
    nig_X:      dict,
    nig_Y:      dict,
    Lambda_X:   float = 0.0,
    Lambda_Y:   float = 0.0,
    n_quad:     int   = 64,
    _kernels:   tuple | None = None,
) -> complex:
    """
    Joint conditional characteristic function of (X_T, Y_T) given F_t.

    Implements equation (3.6) of the paper:

        phi_t(u, v; T) = exp(
            i u X_bar + i v Y_bar
            + int_0^tau psi_X(u * beta_X(s)) ds
            + int_0^tau psi_Y(u * gamma_X(s) + v * beta_Y(s)) ds
        )

    where tau = T - t, X_bar = Lambda_X + c_X^T exp(A_X tau) Z_X,
    Y_bar = Lambda_Y + c_Y^T exp(A_Y tau) Z_Y.

    Parameters
    ----------
    u, v      : transform arguments (complex for damped Fourier)
    tau       : time to maturity in years
    Z_X, Z_Y  : current state vectors (p_X,), (p_Y,)
    nig_X/Y   : dicts with keys alpha, beta, mu, delta
    Lambda_X/Y: deterministic seasonal components at maturity
    _kernels  : pre-computed kernel tuple from _kernel_integrals (for speed)
    """
    if _kernels is None:
        _, w_nodes, bX, gX, bY = _kernel_integrals(
            tau, A_X, c_X, B_X, A_Y, c_Y, B_Y, Gamma, n_quad
        )
    else:
        _, w_nodes, bX, gX, bY = _kernels

    # Deterministic mean components
    X_bar = Lambda_X + float(c_X @ expm(A_X * tau) @ Z_X)
    Y_bar = Lambda_Y + float(c_Y @ expm(A_Y * tau) @ Z_Y)

    # Integrated Lévy exponents (quadrature)
    alpha_X, beta_X_p, mu_X, delta_X = _nig_params_per_year(nig_X)
    alpha_Y, beta_Y_p, mu_Y, delta_Y = _nig_params_per_year(nig_Y)

    psi_X_vals = nig_char_exponent(u * bX, alpha_X, beta_X_p, mu_X, delta_X)
    psi_Y_vals = nig_char_exponent(u * gX + v * bY, alpha_Y, beta_Y_p, mu_Y, delta_Y)

    int_X = np.dot(w_nodes, psi_X_vals)
    int_Y = np.dot(w_nodes, psi_Y_vals)

    log_cf = 1j * u * X_bar + 1j * v * Y_bar + int_X + int_Y
    return complex(np.exp(log_cf))


def compute_forward_price(
    tau:      float,
    Z_X:      np.ndarray,
    A_X:      np.ndarray,
    c_X:      np.ndarray,
    B_X:      np.ndarray,
    A_Y:      np.ndarray,
    c_Y:      np.ndarray,
    B_Y:      np.ndarray,
    Gamma:    np.ndarray,
    nig_X:    dict,
    nig_Y:    dict,
    Lambda_X: float = 0.0,
    n_quad:   int   = 64,
    _kernels: tuple | None = None,
) -> float:
    """
    Forward price F(t, T) = E^Q[S_T | F_t] = exp(X_bar + kappa integrals).

    Implements Corollary 2.4 (eq. 2.14) of the paper:

        F(t,T) = exp( X_bar(T) + int_t^T kappa_X(beta_X) ds
                                + int_t^T kappa_Y(gamma_X) ds )

    Uses u = -i (analytic continuation) in the joint CF formula.
    """
    if _kernels is None:
        _, w_nodes, bX, gX, bY = _kernel_integrals(
            tau, A_X, c_X, B_X, A_Y, c_Y, B_Y, Gamma, n_quad
        )
    else:
        _, w_nodes, bX, gX, bY = _kernels

    X_bar = Lambda_X + float(c_X @ expm(A_X * tau) @ Z_X)

    # kappa = Levy cumulant (real-line version)
    alpha_X, beta_X_p, mu_X, delta_X = _nig_params_per_year(nig_X)
    alpha_Y, beta_Y_p, mu_Y, delta_Y = _nig_params_per_year(nig_Y)

    kappa_X_vals = nig_cumulant_exponent(bX, alpha_X, beta_X_p, mu_X, delta_X)
    kappa_Y_vals = nig_cumulant_exponent(gX, alpha_Y, beta_Y_p, mu_Y, delta_Y)

    log_F = X_bar + np.dot(w_nodes, kappa_X_vals) + np.dot(w_nodes, kappa_Y_vals)
    return float(np.exp(log_F))


def fourier_price_1d(
    tau:      float,
    Z_X:      np.ndarray,
    A_X:      np.ndarray,
    c_X:      np.ndarray,
    B_X:      np.ndarray,
    A_Y:      np.ndarray,
    c_Y:      np.ndarray,
    B_Y:      np.ndarray,
    Gamma:    np.ndarray,
    nig_X:    dict,
    nig_Y:    dict,
    K:        float,
    Lambda_X: float = 0.0,
    alpha:    float = 1.5,
    n_quad:   int   = 64,
    n_fft:    int   = 2048,
) -> float:
    """
    Price of a European call (S_T - K)^+ via 1D Carr-Madan FFT.

    The call price is:
        C = (1/(2pi)) * int_{-inf}^{inf} e^{-i v k} phi_u(v) dv
    where k = log(K) - Lambda_X, phi_u(v) = phi(-i alpha + v; 0) * e^{alpha k}
          / ((alpha + iv)(alpha + iv - 1))

    Parameters
    ----------
    alpha : dampening parameter (alpha > 1 for calls)
    """
    _kernels = _kernel_integrals(tau, A_X, c_X, B_X, A_Y, c_Y, B_Y, Gamma, n_quad)
    k_log = np.log(K) - Lambda_X   # log-strike relative to seasonal component

    # Vectorised over a grid of u values and use trapezoidal integration
    N     = n_fft
    eta   = 0.25                              # frequency spacing
    lam   = 2 * np.pi / (N * eta)            # log-strike spacing
    b     = N * lam / 2                       # b: range [-b, b] in k-space

    # Integration grid
    v_grid = np.arange(N) * eta              # (N,)
    k_grid = -b + lam * np.arange(N)        # log-strike grid

    # Modified CF: phi(v - i*alpha; 0) evaluated at v_grid
    cf_vals = np.array([
        carma_joint_cf(v - 1j * alpha, 0.0, tau, Z_X, Z_Y=np.zeros(A_Y.shape[0]),
                       A_X=A_X, c_X=c_X, B_X=B_X, A_Y=A_Y, c_Y=c_Y, B_Y=B_Y,
                       Gamma=Gamma, nig_X=nig_X, nig_Y=nig_Y,
                       Lambda_X=0.0, Lambda_Y=0.0, _kernels=_kernels)
        for v in v_grid
    ])

    # Carr-Madan formula
    psi_v = np.exp(-1j * v_grid * (k_log + b)) * cf_vals / (
        (alpha + 1j * v_grid) * (alpha + 1j * v_grid - 1.0)
    )
    # Simpson weights
    w_simp = (eta / 3.0) * (3.0 + (-1) ** np.arange(N) - (np.arange(N) == 0).astype(float))
    fft_input = psi_v * w_simp

    fft_output = np.fft.fft(fft_input)
    call_prices = np.real(np.exp(-alpha * k_grid) / np.pi * fft_output)

    # Interpolate to the requested strike
    call = float(np.interp(k_log, k_grid, call_prices))
    return max(call, 0.0)


def fourier_price_indicator_quanto(
    tau:      float,
    Z_X:      np.ndarray,
    Z_Y:      np.ndarray,
    A_X:      np.ndarray,
    c_X:      np.ndarray,
    B_X:      np.ndarray,
    A_Y:      np.ndarray,
    c_Y:      np.ndarray,
    B_Y:      np.ndarray,
    Gamma:    np.ndarray,
    nig_X:    dict,
    nig_Y:    dict,
    K_S:      float,
    K_Y:      float,
    Lambda_X: float = 0.0,
    Lambda_Y: float = 0.0,
    alpha1:   float = 0.75,
    alpha2:   float = 0.75,
    n_quad:   int   = 64,
    n_grid:   int   = 256,
    u_max:    float = 50.0,
) -> float:
    """
    Price of the indicator quanto 1_{S_T > K_S} * 1_{Y_T > K_Y}.

    Uses the 2D Fourier inversion formula (Prop 4.3 of the paper):

        V = (1/(2pi)^2) * int int ĝ_alpha(u,v) * phi(u-i*alpha1, v-i*alpha2) du dv

    The payoff transform factorizes:
        ĝ_alpha(u,v) = exp(-(alpha1+iu)*x*) / (alpha1+iu)
                     * exp(-(alpha2+iv)*y*) / (alpha2+iv)
    where x* = log(K_S) - Lambda_X,  y* = K_Y - Lambda_Y.

    Integration is by 2D Gauss-Legendre on [-u_max, u_max]^2.
    """
    _kernels = _kernel_integrals(tau, A_X, c_X, B_X, A_Y, c_Y, B_Y, Gamma, n_quad)

    x_star = np.log(K_S) - Lambda_X
    y_star = K_Y - Lambda_Y

    from numpy.polynomial.legendre import leggauss
    xi_u, w_u = leggauss(n_grid)
    u_nodes = u_max * xi_u
    v_nodes = u_max * xi_u
    w_scale = (u_max ** 2)

    total = 0.0 + 0.0j
    for i, (u_i, wu_i) in enumerate(zip(u_nodes, w_u)):
        for j, (v_j, wv_j) in enumerate(zip(v_nodes, w_u)):
            u_c = u_i - 1j * alpha1
            v_c = v_j - 1j * alpha2
            phi = carma_joint_cf(
                u_c, v_c, tau, Z_X, Z_Y,
                A_X, c_X, B_X, A_Y, c_Y, B_Y, Gamma,
                nig_X, nig_Y, Lambda_X=0.0, Lambda_Y=0.0,
                _kernels=_kernels,
            )
            g_hat = (np.exp(-(alpha1 + 1j * u_i) * x_star) / (alpha1 + 1j * u_i)
                     * np.exp(-(alpha2 + 1j * v_j) * y_star) / (alpha2 + 1j * v_j))
            total += wu_i * wv_j * g_hat * phi

    price = np.real(w_scale * total / (2 * np.pi) ** 2)
    return float(price)


def simulate_coupled_carma(
    A_X:       np.ndarray,
    B_X:       np.ndarray,
    c_X:       np.ndarray,
    A_Y:       np.ndarray,
    B_Y:       np.ndarray,
    c_Y:       np.ndarray,
    Gamma:     np.ndarray,
    n_steps:   int,
    dt:        float,
    nig_X:     dict,
    nig_Y:     dict,
    n_paths:   int   = 10_000,
    z0_X:      np.ndarray | None = None,
    z0_Y:      np.ndarray | None = None,
    rng:       np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the coupled CARMA system using exact matrix-exponential steps.

    State equations (discrete, exact):
        Z^Y_{k+1} = F_Y Z^Y_k + B_Y_load * dL^Y_k
        Z^X_{k+1} = F_X Z^X_k + B_X_load * dL^X_k + Gamma_load * dL^Y_k

    where F_M = exp(A_M dt),  B_M_load = A_M^{-1}(F_M - I) B_M,
    Gamma_load = A_X^{-1}(F_X - I) Gamma.

    Parameters
    ----------
    nig_X, nig_Y : NIG parameter dicts (alpha, beta, mu, delta)

    Returns
    -------
    X : (n_paths, n_steps+1)  log-price factor output
    Y : (n_paths, n_steps+1)  secondary factor output
    Z_X_paths : (n_paths, n_steps+1, p_X)  price state
    Z_Y_paths : (n_paths, n_steps+1, p_Y)  secondary state
    """
    if rng is None:
        rng = np.random.default_rng(42)

    p_X, p_Y = A_X.shape[0], A_Y.shape[0]
    F_X = expm(A_X * dt)
    F_Y = expm(A_Y * dt)

    def _load(A, F, B):
        try:
            return np.linalg.solve(A, (F - np.eye(A.shape[0])) @ B)
        except np.linalg.LinAlgError:
            return dt * B

    BX_load  = _load(A_X, F_X, B_X)        # (p_X, 1)
    BY_load  = _load(A_Y, F_Y, B_Y)        # (p_Y, 1)
    Gam_load = _load(A_X, F_X, Gamma)      # (p_X, 1)

    Z_X = np.zeros((n_paths, p_X))
    Z_Y = np.zeros((n_paths, p_Y))
    if z0_X is not None:
        Z_X[:] = z0_X
    if z0_Y is not None:
        Z_Y[:] = z0_Y

    X_paths  = np.zeros((n_paths, n_steps + 1))
    Y_paths  = np.zeros((n_paths, n_steps + 1))
    X_paths[:, 0] = Z_X @ c_X
    Y_paths[:, 0] = Z_Y @ c_Y

    alpha_X, beta_X, mu_X, delta_X = _nig_params_for_step(nig_X, dt)
    alpha_Y, beta_Y, mu_Y, delta_Y = _nig_params_for_step(nig_Y, dt)

    for k in range(n_steps):
        dL_Y = nig_sample(alpha_Y, beta_Y, mu_Y, delta_Y, n_paths, rng)
        dL_X = nig_sample(alpha_X, beta_X, mu_X, delta_X, n_paths, rng)

        noise_Y  = BY_load[:, 0][None, :] * dL_Y[:, None]    # (n_paths, p_Y)
        noise_X  = (BX_load[:, 0][None, :] * dL_X[:, None]
                    + Gam_load[:, 0][None, :] * dL_Y[:, None])

        Z_Y = (F_Y @ Z_Y.T).T + noise_Y
        Z_X = (F_X @ Z_X.T).T + noise_X

        X_paths[:, k + 1] = Z_X @ c_X
        Y_paths[:, k + 1] = Z_Y @ c_Y

    return X_paths, Y_paths, None, None   # states not returned to save memory


def compute_hedge_ratio_fd(
    tau:      float,
    Z_X:      np.ndarray,
    Z_Y:      np.ndarray,
    A_X:      np.ndarray,
    c_X:      np.ndarray,
    B_X:      np.ndarray,
    A_Y:      np.ndarray,
    c_Y:      np.ndarray,
    B_Y:      np.ndarray,
    Gamma:    np.ndarray,
    nig_X:    dict,
    nig_Y:    dict,
    K_S:      float,
    K_Y:      float,
    Lambda_X: float = 0.0,
    Lambda_Y: float = 0.0,
    dt:       float = 1.0 / 8760,
    sigma_F:  float | None = None,
) -> float:
    """
    Estimate the GKW hedge ratio xi_t via finite differences on the Fourier price.

    The hedge ratio is:
        xi_t = d<V, F> / d<F, F>

    approximated as:
        xi_t ≈ (V(Z_X + h*B_X_load) - V(Z_X - h*B_X_load)) / (2h * sigma_F^2 * dt)

    where h is a small perturbation in state space and sigma_F^2 is the
    instantaneous variance of log(F).
    """
    _kernels = _kernel_integrals(tau, A_X, c_X, B_X, A_Y, c_Y, B_Y, Gamma)
    F_t = compute_forward_price(tau, Z_X, A_X, c_X, B_X, A_Y, c_Y, B_Y, Gamma,
                                nig_X, nig_Y, Lambda_X=Lambda_X, _kernels=_kernels)

    # Instantaneous variance of log(F): beta_X^T Sigma_X beta_X + gamma_X^T Sigma_Y gamma_X
    # (Using NIG variance: Var(L_1) = delta/gamma^3 * alpha^2 ... for scalar case)
    # Approximate sigma_F^2 * dt using the variance of B_load^T dZ
    F_X = expm(A_X * dt)
    try:
        B_X_load = np.linalg.solve(A_X, (F_X - np.eye(A_X.shape[0])) @ B_X)
    except np.linalg.LinAlgError:
        B_X_load = dt * B_X
    try:
        Gam_load = np.linalg.solve(A_X, (F_X - np.eye(A_X.shape[0])) @ Gamma)
    except np.linalg.LinAlgError:
        Gam_load = dt * Gamma

    # Kernel at t: beta_X(t;T) = B_X^T exp(A_X^T tau) c_X
    beta_X_t  = float(B_X.ravel() @ expm(A_X.T * tau) @ c_X)
    gamma_X_t = float(Gamma.ravel() @ expm(A_X.T * tau) @ c_X)

    alpha_X, beta_X_p, mu_X, delta_X = _nig_params_per_year(nig_X)
    alpha_Y, beta_Y_p, mu_Y, delta_Y = _nig_params_per_year(nig_Y)

    var_L_X = delta_X * alpha_X ** 2 / (alpha_X ** 2 - beta_X_p ** 2) ** 1.5
    var_L_Y = delta_Y * alpha_Y ** 2 / (alpha_Y ** 2 - beta_Y_p ** 2) ** 1.5

    if sigma_F is None:
        sigma_F_sq = beta_X_t ** 2 * var_L_X + gamma_X_t ** 2 * var_L_Y
    else:
        sigma_F_sq = sigma_F ** 2

    if sigma_F_sq < 1e-12:
        return 0.0

    # Finite-difference perturbation in state direction B_X_load
    h = max(np.linalg.norm(B_X_load) * 0.01, 1e-4)
    e = B_X_load[:, 0] / (np.linalg.norm(B_X_load) + 1e-12)

    Z_up   = Z_X + h * e
    Z_down = Z_X - h * e

    V_up   = fourier_price_indicator_quanto(
        tau, Z_up, Z_Y, A_X, c_X, B_X, A_Y, c_Y, B_Y, Gamma,
        nig_X, nig_Y, K_S, K_Y, Lambda_X, Lambda_Y,
    )
    V_down = fourier_price_indicator_quanto(
        tau, Z_down, Z_Y, A_X, c_X, B_X, A_Y, c_Y, B_Y, Gamma,
        nig_X, nig_Y, K_S, K_Y, Lambda_X, Lambda_Y,
    )

    dV_dZ = (V_up - V_down) / (2 * h)        # directional derivative in e direction
    dV_dLogF = dV_dZ / (np.linalg.norm(e) * F_t + 1e-12)

    xi = dV_dLogF / sigma_F_sq
    return float(xi)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Convenience I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_params(params: dict, path: Path | str) -> None:
    """Save a parameter dict to a JSON file (creates parent directories)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved → {path}")


def load_params(path: Path | str) -> dict:
    """Load a parameter dict from a JSON file."""
    with open(path) as f:
        return json.load(f)
