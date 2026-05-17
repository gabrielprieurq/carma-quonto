"""Exact state-space QMLE calibration for a bivariate MCARMA(2,1) model.

The implemented convention is the Dahl / Schlemm / Marquardt-Stelzer form

    P(z) = I_2 z^2 + A1 z + A2,
    Q(z) = B0 z + B1,

with state-space noise loading

    beta = [B0; B1 - A1 @ B0].

Example workflow:

    result_direct = calibrate_mcarma21(
        Y,
        A1_init,
        A2_init,
        Sigma_u=Sigma_u,
        B0_init=B0_init,
        B1_init=B1_init,
        mode="A1_A2_B1",
        maxiter=100,
    )

    result_b1 = calibrate_mcarma21(Y, A1_init, A2_init, B0_init=B0_init, mode="B1_only")
    result_b0b1 = calibrate_mcarma21(
        Y,
        result_b1["A1"],
        result_b1["A2"],
        B0_init=result_b1["B0"],
        B1_init=result_b1["B1"],
        mode="B0_B1",
    )
    result_full = calibrate_mcarma21(
        Y,
        result_b0b1["A1"],
        result_b0b1["A2"],
        B0_init=result_b0b1["B0"],
        B1_init=result_b0b1["B1"],
        mode="full",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import (
    LinAlgError,
    cholesky,
    expm,
    solve,
    solve_discrete_lyapunov,
    solve_triangular,
)
from scipy.optimize import OptimizeResult, minimize

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when numba is absent.
    njit = None
    NUMBA_AVAILABLE = False


FloatArray = NDArray[np.float64]

VALID_CALIBRATION_MODES = {"B1_only", "B0_B1", "A1_A2_B1", "A_B1", "full"}


class MCARMANumericalError(RuntimeError):
    """Controlled exception for invalid numerical points during calibration."""


@dataclass(frozen=True)
class ObjectiveContext:
    """Container for fixed quantities used by the negative log-likelihood."""

    Y_centered: FloatArray
    mode: str
    fixed_matrices: dict[str, FloatArray]
    h: float
    r_min: float
    Sigma_L: FloatArray
    C: FloatArray
    penalty: float
    jitter: float
    ar_margin: float
    psd_tol: float
    covariance_convergence_tol: float | None


def symmetrize(matrix: ArrayLike) -> FloatArray:
    """Return the symmetric part of a square matrix."""
    matrix_array = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix_array + matrix_array.T)


def validate_matrix_shape(name: str, matrix: ArrayLike, shape: tuple[int, int]) -> FloatArray:
    """Convert a matrix to float and validate its shape."""
    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix_array.shape}.")
    return matrix_array.copy()


def enforce_lower_triangular_structure(
    A1: ArrayLike,
    A2: ArrayLike,
    B0: ArrayLike | None = None,
    B1: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray | None, FloatArray | None]:
    """Enforce the no price-to-temperature restrictions and validate shapes."""
    A1_checked = validate_matrix_shape("A1", A1, (2, 2))
    A2_checked = validate_matrix_shape("A2", A2, (2, 2))
    A1_checked[0, 1] = 0.0
    A2_checked[0, 1] = 0.0

    B0_checked = None
    if B0 is not None:
        B0_checked = validate_matrix_shape("B0", B0, (2, 2))
        B0_checked[0, 1] = 0.0

    B1_checked = None
    if B1 is not None:
        B1_checked = validate_matrix_shape("B1", B1, (2, 2))
        B1_checked[0, 1] = 0.0

    return A1_checked, A2_checked, B0_checked, B1_checked


def build_A_comp(A1: ArrayLike, A2: ArrayLike) -> FloatArray:
    """Build the 4x4 continuous-time companion matrix."""
    A1_checked, A2_checked, _, _ = enforce_lower_triangular_structure(A1, A2)
    return np.block(
        [
            [np.zeros((2, 2)), np.eye(2)],
            [-A2_checked, -A1_checked],
        ]
    )


def build_C() -> FloatArray:
    """Build C = [I_2, 0_2]."""
    return np.hstack([np.eye(2), np.zeros((2, 2))])


def build_beta(A1: ArrayLike, B0: ArrayLike, B1: ArrayLike) -> FloatArray:
    """Build beta = [B0; B1 - A1 @ B0] for Q(z) = B0 z + B1."""
    A1_checked = validate_matrix_shape("A1", A1, (2, 2))
    B0_checked = validate_matrix_shape("B0", B0, (2, 2))
    B1_checked = validate_matrix_shape("B1", B1, (2, 2))
    return np.vstack([B0_checked, B1_checked - A1_checked @ B0_checked])


def validate_sigma_L(Sigma_L: ArrayLike | None = None) -> FloatArray:
    """Return the driving-noise covariance, using I_2 by default."""
    if Sigma_L is None:
        return np.eye(2)
    Sigma_L_checked = validate_matrix_shape("Sigma_L", Sigma_L, (2, 2))
    return symmetrize(Sigma_L_checked)


def lower_cholesky_from_sigma_u(Sigma_u: ArrayLike, h: float) -> FloatArray:
    """Build B0 from the lower Cholesky factor of Sigma_u / h."""
    Sigma_u_checked = validate_matrix_shape("Sigma_u", Sigma_u, (2, 2))
    if h <= 0.0:
        raise ValueError("h must be positive.")
    innovation_covariance = symmetrize(Sigma_u_checked / h)
    try:
        return cholesky(innovation_covariance, lower=True, check_finite=True)
    except LinAlgError as exc:
        raise ValueError("Sigma_u / h must be positive definite for Cholesky initialization.") from exc


def _canonical_mode(mode: str) -> str:
    """Normalize mode aliases."""
    if mode == "A_B1":
        return "A1_A2_B1"
    return mode


def unpack_params(
    theta: ArrayLike,
    mode: str,
    fixed_matrices: dict[str, FloatArray],
    r_min: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, dict[str, float]]:
    """Convert an optimizer vector into A1, A2, B0, and B1."""
    mode = _canonical_mode(mode)
    if mode not in VALID_CALIBRATION_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Expected one of {sorted(VALID_CALIBRATION_MODES)}.")
    theta_array = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(theta_array)):
        raise MCARMANumericalError("Non-finite optimizer parameters.")

    if mode == "B1_only":
        if theta_array.size != 3:
            raise ValueError("B1_only mode expects theta with length 3.")
        A1 = fixed_matrices["A1"].copy()
        A2 = fixed_matrices["A2"].copy()
        B0 = fixed_matrices["B0"].copy()
        r_T = r_min + np.exp(theta_array[0])
        r_P = r_min + np.exp(theta_array[1])
        b1_PT = theta_array[2]
    elif mode == "B0_B1":
        if theta_array.size != 6:
            raise ValueError("B0_B1 mode expects theta with length 6.")
        A1 = fixed_matrices["A1"].copy()
        A2 = fixed_matrices["A2"].copy()
        B0 = np.array([[np.exp(theta_array[0]), 0.0], [theta_array[2], np.exp(theta_array[1])]])
        r_T = r_min + np.exp(theta_array[3])
        r_P = r_min + np.exp(theta_array[4])
        b1_PT = theta_array[5]
    elif mode == "A1_A2_B1":
        if theta_array.size != 9:
            raise ValueError("A1_A2_B1 mode expects theta with length 9.")
        A1 = np.array([[theta_array[0], 0.0], [theta_array[1], theta_array[2]]])
        A2 = np.array([[theta_array[3], 0.0], [theta_array[4], theta_array[5]]])
        B0 = fixed_matrices["B0"].copy()
        r_T = r_min + np.exp(theta_array[6])
        r_P = r_min + np.exp(theta_array[7])
        b1_PT = theta_array[8]
    else:
        if theta_array.size != 12:
            raise ValueError("full mode expects theta with length 12.")
        A1 = np.array([[theta_array[0], 0.0], [theta_array[1], theta_array[2]]])
        A2 = np.array([[theta_array[3], 0.0], [theta_array[4], theta_array[5]]])
        B0 = np.array([[np.exp(theta_array[6]), 0.0], [theta_array[8], np.exp(theta_array[7])]])
        r_T = r_min + np.exp(theta_array[9])
        r_P = r_min + np.exp(theta_array[10])
        b1_PT = theta_array[11]

    B1 = np.array([[r_T * B0[0, 0], 0.0], [b1_PT, r_P * B0[1, 1]]])
    A1, A2, B0, B1 = enforce_lower_triangular_structure(A1, A2, B0, B1)
    if B0 is None or B1 is None:
        raise MCARMANumericalError("Internal parameter unpacking failed.")
    return A1, A2, B0, B1, {"r_T": float(r_T), "r_P": float(r_P)}


def initial_theta_from_matrices(
    A1: ArrayLike,
    A2: ArrayLike,
    B0: ArrayLike,
    B1: ArrayLike,
    mode: str,
    r_min: float,
    small_positive: float = 1e-8,
) -> FloatArray:
    """Build a valid starting theta for the requested calibration mode."""
    mode = _canonical_mode(mode)
    A1_checked, A2_checked, B0_checked, B1_checked = enforce_lower_triangular_structure(A1, A2, B0, B1)
    if mode not in VALID_CALIBRATION_MODES:
        raise ValueError(f"Unknown mode {mode!r}.")
    if B0_checked is None or B1_checked is None:
        raise ValueError("B0 and B1 are required to initialize theta.")
    if np.any(np.diag(B0_checked) <= 0.0):
        raise ValueError("B0 must have a positive diagonal for log initialization.")

    r_T = B1_checked[0, 0] / B0_checked[0, 0]
    r_P = B1_checked[1, 1] / B0_checked[1, 1]
    theta_T = np.log(max(r_T - r_min, small_positive))
    theta_P = np.log(max(r_P - r_min, small_positive))

    if mode == "B1_only":
        return np.array([theta_T, theta_P, B1_checked[1, 0]], dtype=float)
    if mode == "B0_B1":
        return np.array(
            [
                np.log(B0_checked[0, 0]),
                np.log(B0_checked[1, 1]),
                B0_checked[1, 0],
                theta_T,
                theta_P,
                B1_checked[1, 0],
            ],
            dtype=float,
        )
    if mode == "A1_A2_B1":
        return np.array(
            [
                A1_checked[0, 0],
                A1_checked[1, 0],
                A1_checked[1, 1],
                A2_checked[0, 0],
                A2_checked[1, 0],
                A2_checked[1, 1],
                theta_T,
                theta_P,
                B1_checked[1, 0],
            ],
            dtype=float,
        )
    return np.array(
        [
            A1_checked[0, 0],
            A1_checked[1, 0],
            A1_checked[1, 1],
            A2_checked[0, 0],
            A2_checked[1, 0],
            A2_checked[1, 1],
            np.log(B0_checked[0, 0]),
            np.log(B0_checked[1, 1]),
            B0_checked[1, 0],
            theta_T,
            theta_P,
            B1_checked[1, 0],
        ],
        dtype=float,
    )


def discretize_exact(
    A_comp: ArrayLike,
    beta: ArrayLike,
    h: float,
    Sigma_L: ArrayLike | None = None,
    method: str = "van_loan",
    quadrature_steps: int = 512,
) -> tuple[FloatArray, FloatArray]:
    """Compute F_h and Q_h by Van Loan, with quadrature available for validation."""
    A_comp_checked = validate_matrix_shape("A_comp", A_comp, (4, 4))
    beta_checked = np.asarray(beta, dtype=float)
    if beta_checked.shape != (4, 2):
        raise ValueError(f"beta must have shape (4, 2), got {beta_checked.shape}.")
    if h <= 0.0:
        raise ValueError("h must be positive.")
    diffusion = beta_checked @ validate_sigma_L(Sigma_L) @ beta_checked.T

    if method == "van_loan":
        block = np.block([[A_comp_checked, diffusion], [np.zeros((4, 4)), -A_comp_checked.T]])
        exponential = expm(block * h)
        F_h = exponential[:4, :4]
        upper_right = exponential[:4, 4:]
        lower_right = exponential[4:, 4:]
        Q_h = solve(lower_right.T, upper_right.T, assume_a="gen").T
    elif method == "quadrature":
        F_h = expm(A_comp_checked * h)
        grid = np.linspace(0.0, h, quadrature_steps + 1)
        weights = np.ones_like(grid)
        weights[0] = 0.5
        weights[-1] = 0.5
        Q_h = np.zeros((4, 4))
        for weight, step in zip(weights, grid):
            transition = expm(A_comp_checked * step)
            Q_h += weight * (transition @ diffusion @ transition.T)
        Q_h *= h / quadrature_steps
    else:
        raise ValueError("method must be 'van_loan' or 'quadrature'.")
    return F_h, symmetrize(Q_h)


def stationary_state_covariance(F_h: ArrayLike, Q_h: ArrayLike) -> FloatArray:
    """Solve Vx = F_h Vx F_h' + Q_h and symmetrize the result."""
    try:
        Vx = solve_discrete_lyapunov(np.asarray(F_h, dtype=float), np.asarray(Q_h, dtype=float))
    except LinAlgError as exc:
        raise MCARMANumericalError("Stationary covariance solve failed.") from exc
    return symmetrize(Vx)


def cholesky_with_progressive_jitter(
    matrix: ArrayLike,
    jitter: float,
    max_tries: int = 8,
) -> tuple[FloatArray, FloatArray, float]:
    """Compute a Cholesky factor, adding numerical jitter only if needed."""
    matrix_checked = symmetrize(matrix)
    if not np.all(np.isfinite(matrix_checked)):
        raise MCARMANumericalError("Innovation covariance contains non-finite values.")
    identity = np.eye(matrix_checked.shape[0])
    base_jitter = max(float(jitter), 0.0)
    for attempt in range(max_tries + 1):
        current_jitter = 0.0 if attempt == 0 else base_jitter * (10.0 ** (attempt - 1))
        if attempt > 0 and current_jitter == 0.0:
            current_jitter = 1e-14 * (10.0 ** (attempt - 1))
        try:
            adjusted = matrix_checked + current_jitter * identity
            factor = cholesky(adjusted, lower=True, check_finite=True)
            return factor, adjusted, current_jitter
        except LinAlgError:
            continue
    raise MCARMANumericalError("Cholesky factorization failed after progressive jitter.")


if NUMBA_AVAILABLE:

    @njit
    def _kalman_loglik_canonical_numba(
        observations: FloatArray,
        F_h: FloatArray,
        Q_h: FloatArray,
        Vx: FloatArray,
        jitter: float,
        convergence_tol: float,
        use_steady_covariance: bool,
    ) -> tuple[int, float]:
        """Numba implementation for the canonical C = [I_2, 0_2] case."""
        n_obs = observations.shape[0]
        predicted_mean = np.zeros(4)
        predicted_covariance = 0.5 * (Vx + Vx.T)
        identity_state = np.eye(4)
        loglik = 0.0

        steady_active = False
        steady_transition = np.zeros((4, 4))
        steady_observation_gain = np.zeros((4, 2))
        steady_l00 = 0.0
        steady_l10 = 0.0
        steady_l11 = 0.0
        steady_s00 = 0.0
        steady_s01 = 0.0
        steady_s11 = 0.0

        for observation_index in range(n_obs):
            innovation0 = observations[observation_index, 0] - predicted_mean[0]
            innovation1 = observations[observation_index, 1] - predicted_mean[1]

            if steady_active:
                l00 = steady_l00
                l10 = steady_l10
                l11 = steady_l11
                s00 = steady_s00
                s01 = steady_s01
                s11 = steady_s11
            else:
                raw_s00 = predicted_covariance[0, 0]
                raw_s01 = 0.5 * (predicted_covariance[0, 1] + predicted_covariance[1, 0])
                raw_s11 = predicted_covariance[1, 1]
                found_cholesky = False
                l00 = 0.0
                l10 = 0.0
                l11 = 0.0
                s00 = 0.0
                s01 = 0.0
                s11 = 0.0

                for attempt in range(9):
                    current_jitter = 0.0
                    if attempt > 0:
                        current_jitter = jitter * (10.0 ** (attempt - 1))
                        if current_jitter == 0.0:
                            current_jitter = 1e-14 * (10.0 ** (attempt - 1))
                    s00 = raw_s00 + current_jitter
                    s01 = raw_s01
                    s11 = raw_s11 + current_jitter
                    if not (np.isfinite(s00) and np.isfinite(s01) and np.isfinite(s11)):
                        return 1, 0.0
                    if s00 <= 0.0:
                        continue
                    l00_candidate = np.sqrt(s00)
                    l10_candidate = s01 / l00_candidate
                    remainder = s11 - l10_candidate * l10_candidate
                    if remainder <= 0.0:
                        continue
                    l00 = l00_candidate
                    l10 = l10_candidate
                    l11 = np.sqrt(remainder)
                    found_cholesky = True
                    break

                if not found_cholesky:
                    return 2, 0.0

            standardized0 = innovation0 / l00
            standardized1 = (innovation1 - l10 * standardized0) / l11
            logdet = 2.0 * (np.log(l00) + np.log(l11))
            quadratic = standardized0 * standardized0 + standardized1 * standardized1
            loglik += -0.5 * (2.0 * np.log(2.0 * np.pi) + logdet + quadratic)

            if steady_active:
                next_mean = np.zeros(4)
                for row in range(4):
                    value = steady_observation_gain[row, 0] * observations[observation_index, 0]
                    value += steady_observation_gain[row, 1] * observations[observation_index, 1]
                    for col in range(4):
                        value += steady_transition[row, col] * predicted_mean[col]
                    next_mean[row] = value
                predicted_mean = next_mean
                continue

            determinant = s00 * s11 - s01 * s01
            if determinant <= 0.0 or not np.isfinite(determinant):
                return 2, 0.0
            inv00 = s11 / determinant
            inv01 = -s01 / determinant
            inv11 = s00 / determinant

            kalman_gain = np.zeros((4, 2))
            for row in range(4):
                p0 = predicted_covariance[row, 0]
                p1 = predicted_covariance[row, 1]
                kalman_gain[row, 0] = p0 * inv00 + p1 * inv01
                kalman_gain[row, 1] = p0 * inv01 + p1 * inv11

            filtered_mean = np.empty(4)
            for row in range(4):
                filtered_mean[row] = predicted_mean[row]
                filtered_mean[row] += kalman_gain[row, 0] * innovation0
                filtered_mean[row] += kalman_gain[row, 1] * innovation1

            filtered_covariance = np.empty((4, 4))
            for row in range(4):
                ks0 = kalman_gain[row, 0] * s00 + kalman_gain[row, 1] * s01
                ks1 = kalman_gain[row, 0] * s01 + kalman_gain[row, 1] * s11
                for col in range(4):
                    value = predicted_covariance[row, col]
                    value -= ks0 * kalman_gain[col, 0] + ks1 * kalman_gain[col, 1]
                    filtered_covariance[row, col] = value
            filtered_covariance = 0.5 * (filtered_covariance + filtered_covariance.T)

            next_mean = np.zeros(4)
            for row in range(4):
                for col in range(4):
                    next_mean[row] += F_h[row, col] * filtered_mean[col]

            temp_covariance = np.zeros((4, 4))
            for row in range(4):
                for col in range(4):
                    value = 0.0
                    for inner in range(4):
                        value += F_h[row, inner] * filtered_covariance[inner, col]
                    temp_covariance[row, col] = value

            next_predicted_covariance = np.empty((4, 4))
            for row in range(4):
                for col in range(4):
                    value = Q_h[row, col]
                    for inner in range(4):
                        value += temp_covariance[row, inner] * F_h[col, inner]
                    next_predicted_covariance[row, col] = value
            next_predicted_covariance = 0.5 * (next_predicted_covariance + next_predicted_covariance.T)

            if use_steady_covariance and observation_index >= 2:
                numerator = 0.0
                denominator = 0.0
                for row in range(4):
                    for col in range(4):
                        diff = next_predicted_covariance[row, col] - predicted_covariance[row, col]
                        numerator += diff * diff
                        denominator += predicted_covariance[row, col] * predicted_covariance[row, col]
                covariance_scale = max(np.sqrt(denominator), 1.0)
                covariance_change = np.sqrt(numerator) / covariance_scale
                if covariance_change < convergence_tol:
                    steady_active = True
                    steady_l00 = l00
                    steady_l10 = l10
                    steady_l11 = l11
                    steady_s00 = s00
                    steady_s01 = s01
                    steady_s11 = s11

                    F_times_gain = np.zeros((4, 2))
                    for row in range(4):
                        for gain_col in range(2):
                            value = 0.0
                            for inner in range(4):
                                value += F_h[row, inner] * kalman_gain[inner, gain_col]
                            F_times_gain[row, gain_col] = value
                            steady_observation_gain[row, gain_col] = value
                    for row in range(4):
                        for col in range(4):
                            steady_transition[row, col] = F_h[row, col]
                        steady_transition[row, 0] -= F_times_gain[row, 0]
                        steady_transition[row, 1] -= F_times_gain[row, 1]

            predicted_mean = next_mean
            predicted_covariance = next_predicted_covariance

        if not np.isfinite(loglik):
            return 3, 0.0
        return 0, loglik


def _can_use_numba_kalman(C: FloatArray, return_details: bool) -> bool:
    """Return True when the numba Kalman implementation is applicable."""
    if not NUMBA_AVAILABLE or return_details:
        return False
    canonical_C = build_C()
    return bool(np.allclose(C, canonical_C, atol=0.0, rtol=0.0))


def kalman_loglik(
    Y_centered: ArrayLike,
    F_h: ArrayLike,
    Q_h: ArrayLike,
    C: ArrayLike,
    Vx: ArrayLike,
    jitter: float,
    return_details: bool = False,
    covariance_convergence_tol: float | None = 1e-12,
) -> float | tuple[float, FloatArray, FloatArray]:
    """Evaluate the Gaussian Kalman log-likelihood with R = 0."""
    observations = np.asarray(Y_centered, dtype=float)
    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError(f"Y_centered must have shape (n_obs, 2), got {observations.shape}.")
    F_h_checked = validate_matrix_shape("F_h", F_h, (4, 4))
    Q_h_checked = validate_matrix_shape("Q_h", Q_h, (4, 4))
    C_checked = np.asarray(C, dtype=float)
    if C_checked.shape != (2, 4):
        raise ValueError(f"C must have shape (2, 4), got {C_checked.shape}.")
    if _can_use_numba_kalman(C_checked, return_details):
        use_steady_covariance = covariance_convergence_tol is not None
        convergence_tol = 0.0 if covariance_convergence_tol is None else float(covariance_convergence_tol)
        status, loglik = _kalman_loglik_canonical_numba(
            np.ascontiguousarray(observations, dtype=np.float64),
            np.ascontiguousarray(F_h_checked, dtype=np.float64),
            np.ascontiguousarray(Q_h_checked, dtype=np.float64),
            np.ascontiguousarray(validate_matrix_shape("Vx", Vx, (4, 4)), dtype=np.float64),
            float(jitter),
            convergence_tol,
            use_steady_covariance,
        )
        if status == 0:
            return float(loglik)
        if status == 1:
            raise MCARMANumericalError("Innovation covariance contains non-finite values.")
        if status == 2:
            raise MCARMANumericalError("Cholesky factorization failed after progressive jitter.")
        raise MCARMANumericalError("Kalman log-likelihood is not finite.")

    predicted_mean = np.zeros(4)
    predicted_covariance = symmetrize(Vx)
    identity_state = np.eye(4)
    loglik = 0.0

    innovations = [] if return_details else None
    innovation_covariances = [] if return_details else None
    steady_transition = None
    steady_observation_gain = None
    steady_chol = None
    steady_covariance_for_solve = None

    for observation_index, observation in enumerate(observations):
        innovation = observation - C_checked @ predicted_mean

        if steady_transition is None:
            innovation_covariance = symmetrize(C_checked @ predicted_covariance @ C_checked.T)
            chol_factor, covariance_for_solve, _ = cholesky_with_progressive_jitter(
                innovation_covariance,
                jitter=jitter,
            )
            covariance_cross = predicted_covariance @ C_checked.T
            gain_transpose = solve_triangular(
                chol_factor.T,
                solve_triangular(chol_factor, covariance_cross.T, lower=True, check_finite=False),
                lower=False,
                check_finite=False,
            )
            kalman_gain = gain_transpose.T
            filtered_covariance = symmetrize(
                predicted_covariance - kalman_gain @ covariance_for_solve @ kalman_gain.T
            )
            next_predicted_covariance = symmetrize(
                F_h_checked @ filtered_covariance @ F_h_checked.T + Q_h_checked
            )

            if covariance_convergence_tol is not None:
                covariance_scale = max(np.linalg.norm(predicted_covariance, ord="fro"), 1.0)
                covariance_change = (
                    np.linalg.norm(next_predicted_covariance - predicted_covariance, ord="fro") / covariance_scale
                )
                if observation_index >= 2 and covariance_change < covariance_convergence_tol:
                    steady_transition = F_h_checked @ (identity_state - kalman_gain @ C_checked)
                    steady_observation_gain = F_h_checked @ kalman_gain
                    steady_chol = chol_factor
                    steady_covariance_for_solve = covariance_for_solve
        else:
            chol_factor = steady_chol
            covariance_for_solve = steady_covariance_for_solve
            innovation_covariance = covariance_for_solve

        if chol_factor is None or covariance_for_solve is None:
            raise MCARMANumericalError("Internal Kalman factorization state is invalid.")
        standardized = solve_triangular(chol_factor, innovation, lower=True, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(chol_factor)))
        quadratic = float(standardized @ standardized)
        loglik += -0.5 * (2.0 * np.log(2.0 * np.pi) + logdet + quadratic)

        if steady_transition is None:
            filtered_mean = predicted_mean + kalman_gain @ innovation
            predicted_mean = F_h_checked @ filtered_mean
            predicted_covariance = next_predicted_covariance
        else:
            predicted_mean = steady_transition @ predicted_mean + steady_observation_gain @ observation

        if return_details:
            innovations.append(innovation)
            innovation_covariances.append(innovation_covariance)

    if not np.isfinite(loglik):
        raise MCARMANumericalError("Kalman log-likelihood is not finite.")
    if return_details:
        if innovations is None or innovation_covariances is None:
            raise MCARMANumericalError("Internal Kalman detail storage is invalid.")
        return float(loglik), np.asarray(innovations), np.asarray(innovation_covariances)
    return float(loglik)


def controllability_rank(A_comp: ArrayLike, beta: ArrayLike, tol: float = 1e-9) -> int:
    """Compute rank([beta, A beta, A^2 beta, A^3 beta])."""
    A_comp_checked = validate_matrix_shape("A_comp", A_comp, (4, 4))
    current_block = np.asarray(beta, dtype=float)
    if current_block.shape != (4, 2):
        raise ValueError(f"beta must have shape (4, 2), got {current_block.shape}.")
    blocks = []
    for _ in range(4):
        blocks.append(current_block)
        current_block = A_comp_checked @ current_block
    return int(np.linalg.matrix_rank(np.hstack(blocks), tol=tol))


def transfer_function_test(
    A1: ArrayLike,
    A2: ArrayLike,
    B0: ArrayLike,
    B1: ArrayLike,
    A_comp: ArrayLike,
    beta: ArrayLike,
    C: ArrayLike,
    z_values: list[complex] | None = None,
) -> float:
    """Return the maximum relative error in the MCARMA transfer-function identity."""
    if z_values is None:
        z_values = [1.0 + 0.5j, 1.5 - 0.25j, -1.0 + 0.75j, 2.0 + 0.0j, -2.5 + 1.0j]
    A1_checked, A2_checked, B0_checked, B1_checked = enforce_lower_triangular_structure(A1, A2, B0, B1)
    if B0_checked is None or B1_checked is None:
        raise ValueError("B0 and B1 are required.")
    A_comp_checked = validate_matrix_shape("A_comp", A_comp, (4, 4))
    beta_checked = np.asarray(beta, dtype=float)
    C_checked = np.asarray(C, dtype=float)
    max_relative_error = 0.0
    I4 = np.eye(4, dtype=complex)
    I2_complex = np.eye(2, dtype=complex)
    for z_value in z_values:
        z_value = complex(z_value)
        left = C_checked @ solve(z_value * I4 - A_comp_checked, beta_checked, assume_a="gen")
        P_z = I2_complex * (z_value**2) + A1_checked * z_value + A2_checked
        Q_z = B0_checked * z_value + B1_checked
        right = solve(P_z, Q_z, assume_a="gen")
        denominator = max(np.linalg.norm(right, ord="fro"), 1e-12)
        relative_error = np.linalg.norm(left - right, ord="fro") / denominator
        max_relative_error = max(max_relative_error, float(relative_error))
    return max_relative_error


def diagnostics(
    A1: ArrayLike,
    A2: ArrayLike,
    B0: ArrayLike,
    B1: ArrayLike,
    beta: ArrayLike,
    A_comp: ArrayLike,
    F_h: ArrayLike,
    Q_h: ArrayLike,
    Vx: ArrayLike,
    r_min: float,
    ar_margin: float,
    psd_tol: float,
) -> dict[str, Any]:
    """Return numerical and structural diagnostics for a calibrated MCARMA(2,1) model."""
    A1_checked, A2_checked, B0_checked, B1_checked = enforce_lower_triangular_structure(A1, A2, B0, B1)
    if B0_checked is None or B1_checked is None:
        raise ValueError("B0 and B1 are required.")
    beta_checked = np.asarray(beta, dtype=float)
    A_comp_checked = validate_matrix_shape("A_comp", A_comp, (4, 4))
    Q_h_checked = validate_matrix_shape("Q_h", Q_h, (4, 4))
    Vx_checked = validate_matrix_shape("Vx", Vx, (4, 4))
    det_B0 = float(np.linalg.det(B0_checked))
    B0_invertible = abs(det_B0) > psd_tol
    if B0_invertible:
        ma_zeros = np.linalg.eigvals(-solve(B0_checked, B1_checked, assume_a="gen"))
        max_real_ma_zero = float(np.max(np.real(ma_zeros)))
    else:
        ma_zeros = np.array([np.nan, np.nan])
        max_real_ma_zero = np.inf

    A_eigenvalues = np.linalg.eigvals(A_comp_checked)
    Qh_eigenvalues = np.linalg.eigvalsh(symmetrize(Q_h_checked))
    Vx_eigenvalues = np.linalg.eigvalsh(symmetrize(Vx_checked))
    rank_value = controllability_rank(A_comp_checked, beta_checked)
    transfer_error = transfer_function_test(
        A1_checked,
        A2_checked,
        B0_checked,
        B1_checked,
        A_comp_checked,
        beta_checked,
        build_C(),
    )

    return {
        "B0_shape_ok": B0_checked.shape == (2, 2),
        "B1_shape_ok": B1_checked.shape == (2, 2),
        "beta_shape_ok": beta_checked.shape == (4, 2),
        "no_price_to_temperature_AR": bool(abs(A1_checked[0, 1]) <= psd_tol and abs(A2_checked[0, 1]) <= psd_tol),
        "no_price_to_temperature_MA": bool(abs(B0_checked[0, 1]) <= psd_tol and abs(B1_checked[0, 1]) <= psd_tol),
        "B0_positive_diagonal": bool(np.all(np.diag(B0_checked) > 0.0)),
        "B1_positive_diagonal": bool(np.all(np.diag(B1_checked) > 0.0)),
        "det_B0": det_B0,
        "B0_invertible": bool(B0_invertible),
        "ma_zeros": ma_zeros,
        "MA_invertible": bool(max_real_ma_zero <= -r_min + psd_tol),
        "A_eigenvalues": A_eigenvalues,
        "AR_stable": bool(np.max(np.real(A_eigenvalues)) < -ar_margin),
        "Qh_eigenvalues": Qh_eigenvalues,
        "Qh_positive_semidefinite": bool(np.min(Qh_eigenvalues) >= -psd_tol),
        "Vx_eigenvalues": Vx_eigenvalues,
        "Vx_positive_semidefinite": bool(np.min(Vx_eigenvalues) >= -psd_tol),
        "controllability_rank": rank_value,
        "state_controllable": bool(rank_value == 4),
        "transfer_function_test_max_error": transfer_error,
        "transfer_function_test_ok": bool(transfer_error < 1e-8),
    }


def evaluate_mcarma_theta(theta: ArrayLike, context: ObjectiveContext) -> dict[str, Any]:
    """Evaluate a parameter vector and return likelihood objects for a valid point."""
    A1, A2, B0, B1, derived = unpack_params(
        theta=theta,
        mode=context.mode,
        fixed_matrices=context.fixed_matrices,
        r_min=context.r_min,
    )
    if not all(np.all(np.isfinite(matrix)) for matrix in [A1, A2, B0, B1]):
        raise MCARMANumericalError("Non-finite matrices generated from theta.")

    A_comp = build_A_comp(A1, A2)
    if np.max(np.real(np.linalg.eigvals(A_comp))) >= -context.ar_margin:
        raise MCARMANumericalError("AR stability margin violated.")

    beta = build_beta(A1, B0, B1)
    F_h, Q_h = discretize_exact(A_comp, beta, context.h, Sigma_L=context.Sigma_L)
    if np.min(np.linalg.eigvalsh(Q_h)) < -context.psd_tol:
        raise MCARMANumericalError("Q_h is not numerically positive semidefinite.")

    Vx = stationary_state_covariance(F_h, Q_h)
    if np.min(np.linalg.eigvalsh(Vx)) < -context.psd_tol:
        raise MCARMANumericalError("Stationary covariance is not numerically positive semidefinite.")

    loglik = kalman_loglik(
        context.Y_centered,
        F_h=F_h,
        Q_h=Q_h,
        C=context.C,
        Vx=Vx,
        jitter=context.jitter,
        return_details=False,
        covariance_convergence_tol=context.covariance_convergence_tol,
    )
    if not isinstance(loglik, float):
        raise MCARMANumericalError("Unexpected Kalman likelihood return type.")
    negative_loglik = -loglik
    if not np.isfinite(negative_loglik):
        raise MCARMANumericalError("Negative log-likelihood is not finite.")

    return {
        "loglik": loglik,
        "negative_loglik": float(negative_loglik),
        "A1": A1,
        "A2": A2,
        "B0": B0,
        "B1": B1,
        "beta": beta,
        "A_comp": A_comp,
        "F_h": F_h,
        "Q_h": Q_h,
        "Vx": Vx,
        "derived": derived,
    }


def negative_loglik_objective(theta: ArrayLike, context: ObjectiveContext) -> float:
    """Objective passed to scipy.optimize.minimize."""
    try:
        return float(evaluate_mcarma_theta(theta, context)["negative_loglik"])
    except (ValueError, LinAlgError, FloatingPointError, MCARMANumericalError, OverflowError):
        return float(context.penalty)


def _prepare_initial_matrices(
    A1_init: ArrayLike,
    A2_init: ArrayLike,
    Sigma_u: ArrayLike | None,
    B0_init: ArrayLike | None,
    B1_init: ArrayLike | None,
    h: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Validate and construct initial A1, A2, B0, and B1."""
    A1, A2, _, _ = enforce_lower_triangular_structure(A1_init, A2_init)
    if B0_init is None:
        if Sigma_u is None:
            raise ValueError("Provide either B0_init or Sigma_u to initialize B0.")
        B0 = lower_cholesky_from_sigma_u(Sigma_u, h)
    else:
        _, _, B0, _ = enforce_lower_triangular_structure(A1, A2, B0=B0_init)
        if B0 is None:
            raise ValueError("B0 initialization failed.")
    if np.any(np.diag(B0) <= 0.0):
        raise ValueError("B0_init must have a positive diagonal after imposing the structural zero.")

    if B1_init is None:
        B1 = np.array([[B0[0, 0], 0.0], [0.0, B0[1, 1]]])
    else:
        _, _, _, B1 = enforce_lower_triangular_structure(A1, A2, B1=B1_init)
        if B1 is None:
            raise ValueError("B1 initialization failed.")
    if np.any(np.diag(B1) <= 0.0):
        raise ValueError("B1_init must have a positive diagonal after imposing the structural zero.")
    return A1, A2, B0, B1


def calibrate_mcarma21(
    Y: ArrayLike,
    A1_init: ArrayLike,
    A2_init: ArrayLike,
    Sigma_u: ArrayLike | None = None,
    B0_init: ArrayLike | None = None,
    B1_init: ArrayLike | None = None,
    h: float = 1.0,
    r_min: float = 0.01,
    mode: str = "B1_only",
    demean: bool = True,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 1000,
    penalty: float = 1e12,
    jitter: float = 1e-10,
    ar_margin: float = 1e-6,
    psd_tol: float = 1e-10,
    Sigma_L: ArrayLike | None = None,
    optimizer_options: dict[str, Any] | None = None,
    covariance_convergence_tol: float | None = 1e-12,
    verbose: bool = False,
) -> dict[str, Any]:
    """Calibrate a bivariate MCARMA(2,1) model by exact discrete-time QMLE."""
    mode = _canonical_mode(mode)
    if mode not in VALID_CALIBRATION_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Expected one of {sorted(VALID_CALIBRATION_MODES)}.")
    observations = np.asarray(Y, dtype=float)
    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError(f"Y must be a NumPy-compatible array with shape (n_obs, 2), got {observations.shape}.")
    if observations.shape[0] < 2:
        raise ValueError("Y must contain at least two observations.")
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if r_min <= 0.0:
        raise ValueError("r_min must be positive.")

    A1, A2, B0, B1 = _prepare_initial_matrices(A1_init, A2_init, Sigma_u, B0_init, B1_init, h)
    theta_start = initial_theta_from_matrices(A1, A2, B0, B1, mode=mode, r_min=r_min)
    mean_used = observations.mean(axis=0) if demean else np.zeros(2)
    observations_centered = observations - mean_used

    fixed_matrices = {"A1": A1, "A2": A2, "B0": B0}
    context = ObjectiveContext(
        Y_centered=observations_centered,
        mode=mode,
        fixed_matrices=fixed_matrices,
        h=h,
        r_min=r_min,
        Sigma_L=validate_sigma_L(Sigma_L),
        C=build_C(),
        penalty=float(penalty),
        jitter=float(jitter),
        ar_margin=float(ar_margin),
        psd_tol=float(psd_tol),
        covariance_convergence_tol=covariance_convergence_tol,
    )

    options = {"maxiter": int(maxiter)}
    if optimizer_options is not None:
        options.update(dict(optimizer_options))

    if verbose:
        start_value = negative_loglik_objective(theta_start, context)
        print(f"Initial negative log-likelihood ({mode}): {start_value:.6f}")

    optimizer_result: OptimizeResult = minimize(
        fun=negative_loglik_objective,
        x0=theta_start,
        args=(context,),
        method=optimizer,
        options=options,
    )

    try:
        final_evaluation = evaluate_mcarma_theta(optimizer_result.x, context)
        final_message = str(optimizer_result.message)
        final_success = bool(optimizer_result.success)
    except (ValueError, LinAlgError, FloatingPointError, MCARMANumericalError, OverflowError) as exc:
        final_evaluation = evaluate_mcarma_theta(theta_start, context)
        final_message = f"Optimizer ended at an invalid point; returned start evaluation. Reason: {exc}"
        final_success = False

    final_diagnostics = diagnostics(
        A1=final_evaluation["A1"],
        A2=final_evaluation["A2"],
        B0=final_evaluation["B0"],
        B1=final_evaluation["B1"],
        beta=final_evaluation["beta"],
        A_comp=final_evaluation["A_comp"],
        F_h=final_evaluation["F_h"],
        Q_h=final_evaluation["Q_h"],
        Vx=final_evaluation["Vx"],
        r_min=r_min,
        ar_margin=ar_margin,
        psd_tol=psd_tol,
    )

    final_loglik = float(final_evaluation["loglik"])
    final_negative_loglik = float(final_evaluation["negative_loglik"])
    if verbose:
        print(f"Final negative log-likelihood ({mode}): {final_negative_loglik:.6f}")
        print(f"Optimizer success ({mode}): {final_success}; {final_message}")

    return {
        "success": final_success,
        "message": final_message,
        "n_iter": getattr(optimizer_result, "nit", None),
        "final_loglik": final_loglik,
        "final_negative_loglik": final_negative_loglik,
        "average_loglik_per_observation": final_loglik / observations.shape[0],
        "theta_opt": np.asarray(optimizer_result.x, dtype=float),
        "A1": final_evaluation["A1"],
        "A2": final_evaluation["A2"],
        "B0": final_evaluation["B0"],
        "B1": final_evaluation["B1"],
        "beta": final_evaluation["beta"],
        "A_comp": final_evaluation["A_comp"],
        "C": context.C,
        "F_h": final_evaluation["F_h"],
        "Q_h": final_evaluation["Q_h"],
        "Vx": final_evaluation["Vx"],
        "mean_used": mean_used,
        "diagnostics": final_diagnostics,
        "optimizer_result": optimizer_result,
    }
