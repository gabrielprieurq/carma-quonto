import numpy as np
from scipy.linalg import solve

from mcarma_calibration import (
    build_A_comp,
    build_C,
    build_beta,
    calibrate_mcarma21,
    controllability_rank,
    diagnostics,
    discretize_exact,
    enforce_lower_triangular_structure,
    kalman_loglik,
    stationary_state_covariance,
    transfer_function_test,
)


def stable_example():
    A1 = np.array([[1.00, 0.0], [0.08, 1.20]])
    A2 = np.array([[0.20, 0.0], [0.04, 0.32]])
    B0 = np.array([[0.70, 0.0], [0.10, 0.35]])
    B1 = np.array([[0.45 * B0[0, 0], 0.0], [0.08, 0.65 * B0[1, 1]]])
    A1, A2, B0, B1 = enforce_lower_triangular_structure(A1, A2, B0, B1)
    return A1, A2, B0, B1


def simulate_observations(F_h, Q_h, C, Vx, n_obs=80, seed=123):
    rng = np.random.default_rng(seed)
    state = rng.multivariate_normal(mean=np.zeros(4), cov=0.5 * (Vx + Vx.T))
    observations = np.empty((n_obs, 2))
    for index in range(n_obs):
        observations[index] = C @ state
        innovation = rng.multivariate_normal(mean=np.zeros(4), cov=0.5 * (Q_h + Q_h.T))
        state = F_h @ state + innovation
    return observations


def state_space_objects():
    A1, A2, B0, B1 = stable_example()
    A_comp = build_A_comp(A1, A2)
    C = build_C()
    beta = build_beta(A1, B0, B1)
    F_h, Q_h = discretize_exact(A_comp, beta, h=1.0)
    Vx = stationary_state_covariance(F_h, Q_h)
    return A1, A2, B0, B1, beta, A_comp, C, F_h, Q_h, Vx


def test_matrix_shapes():
    A1, A2, B0, B1, beta, A_comp, C, F_h, Q_h, _ = state_space_objects()
    assert A1.shape == (2, 2)
    assert A2.shape == (2, 2)
    assert B0.shape == (2, 2)
    assert B1.shape == (2, 2)
    assert beta.shape == (4, 2)
    assert A_comp.shape == (4, 4)
    assert C.shape == (2, 4)
    assert F_h.shape == (4, 4)
    assert Q_h.shape == (4, 4)


def test_structural_zeros():
    A1, A2, B0, B1 = stable_example()
    assert A1[0, 1] == 0.0
    assert A2[0, 1] == 0.0
    assert B0[0, 1] == 0.0
    assert B1[0, 1] == 0.0


def test_positive_diagonals():
    _, _, B0, B1 = stable_example()
    assert np.all(np.diag(B0) > 0.0)
    assert np.all(np.diag(B1) > 0.0)


def test_ma_invertibility_margin():
    _, _, B0, B1 = stable_example()
    r_min = 0.01
    ma_zeros = np.linalg.eigvals(-solve(B0, B1, assume_a="gen"))
    assert np.max(np.real(ma_zeros)) <= -r_min


def test_ar_stability():
    _, _, _, _, _, A_comp, _, _, _, _ = state_space_objects()
    assert np.max(np.real(np.linalg.eigvals(A_comp))) < 0.0


def test_Q_h_positive_semidefinite():
    *_, Q_h, _ = state_space_objects()
    assert np.min(np.linalg.eigvalsh(Q_h)) >= -1e-10


def test_stationary_covariance_equation():
    *_, F_h, Q_h, Vx = state_space_objects()
    assert np.allclose(Vx, F_h @ Vx @ F_h.T + Q_h, atol=1e-8, rtol=1e-8)


def test_transfer_function_identity():
    A1, A2, B0, B1, beta, A_comp, C, *_ = state_space_objects()
    max_error = transfer_function_test(A1, A2, B0, B1, A_comp, beta, C)
    assert max_error < 1e-8


def test_van_loan_matches_quadrature():
    A1, A2, B0, B1 = stable_example()
    A_comp = build_A_comp(A1, A2)
    beta = build_beta(A1, B0, B1)
    _, Q_van_loan = discretize_exact(A_comp, beta, h=1.0, method="van_loan")
    _, Q_quad = discretize_exact(A_comp, beta, h=1.0, method="quadrature", quadrature_steps=2048)
    assert np.allclose(Q_van_loan, Q_quad, atol=5e-5, rtol=5e-5)


def test_kalman_likelihood_finite():
    _, _, _, _, _, _, C, F_h, Q_h, Vx = state_space_objects()
    Y = simulate_observations(F_h, Q_h, C, Vx, n_obs=50, seed=321)
    loglik = kalman_loglik(Y - Y.mean(axis=0), F_h, Q_h, C, Vx, jitter=1e-10)
    assert np.isfinite(loglik)
    assert np.isfinite(-loglik)


def test_fast_kalman_matches_full_recursion():
    _, _, _, _, _, _, C, F_h, Q_h, Vx = state_space_objects()
    Y = simulate_observations(F_h, Q_h, C, Vx, n_obs=60, seed=987)
    centered = Y - Y.mean(axis=0)
    fast_loglik = kalman_loglik(centered, F_h, Q_h, C, Vx, jitter=1e-10)
    full_loglik = kalman_loglik(
        centered,
        F_h,
        Q_h,
        C,
        Vx,
        jitter=1e-10,
        covariance_convergence_tol=None,
    )
    assert abs(fast_loglik - full_loglik) <= 1e-8 * max(abs(full_loglik), 1.0)


def test_controllability_rank():
    _, _, _, _, beta, A_comp, *_ = state_space_objects()
    assert controllability_rank(A_comp, beta) == 4


def test_diagnostics_contains_required_keys():
    A1, A2, B0, B1, beta, A_comp, _, F_h, Q_h, Vx = state_space_objects()
    diag = diagnostics(A1, A2, B0, B1, beta, A_comp, F_h, Q_h, Vx, 0.01, 1e-6, 1e-10)
    required_keys = {
        "B0_shape_ok",
        "B1_shape_ok",
        "beta_shape_ok",
        "no_price_to_temperature_AR",
        "no_price_to_temperature_MA",
        "B0_positive_diagonal",
        "B1_positive_diagonal",
        "det_B0",
        "B0_invertible",
        "ma_zeros",
        "MA_invertible",
        "A_eigenvalues",
        "AR_stable",
        "Qh_eigenvalues",
        "Qh_positive_semidefinite",
        "Vx_eigenvalues",
        "Vx_positive_semidefinite",
        "controllability_rank",
        "state_controllable",
        "transfer_function_test_max_error",
        "transfer_function_test_ok",
    }
    assert required_keys <= set(diag)


def test_direct_A1_A2_B1_optimization_smoke():
    A1, A2, B0, B1, _, _, C, F_h, Q_h, Vx = state_space_objects()
    Y = simulate_observations(F_h, Q_h, C, Vx, n_obs=45, seed=555)
    result = calibrate_mcarma21(
        Y,
        A1,
        A2,
        B0_init=B0,
        B1_init=B1,
        mode="A1_A2_B1",
        maxiter=2,
        demean=True,
    )
    assert isinstance(result, dict)
    assert "diagnostics" in result
    assert result["A1"][0, 1] == 0.0
    assert result["A2"][0, 1] == 0.0
    assert result["B0"][0, 1] == 0.0
    assert result["B1"][0, 1] == 0.0
    assert result["B0"].shape == (2, 2)
    assert result["B1"].shape == (2, 2)
