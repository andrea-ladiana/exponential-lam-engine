import numpy as np
import pytest

import lam_engine
import lam_engine_torch
from config import DEVICE


def test_saddle_point():
    with pytest.raises(ValueError, match="LAM requires L >= 2."):
        lam_engine.saddle_point(1)
    
    res = lam_engine.saddle_point(2)
    assert "x_star" in res
    assert "m_star" in res
    assert "phi_L" in res
    assert "rho_L" in res
    assert "r_c" in res


def test_theory_functions():
    L, N = 2, 32
    mu1 = lam_engine.mu1_signal(L)
    mu1_fin = lam_engine.mu1_signal_finiteN(L, N)
    assert mu1 > 0
    assert mu1_fin > 0
    
    K_L = lam_engine.K_L_prefactor(L)
    assert K_L > 0
    
    noise = lam_engine.noise_var_per_pattern_annealed(N, L)
    assert noise > 0


def test_gamma_eff_and_P():
    L, N = 2, 32
    # Scalar
    g_scalar = lam_engine.gamma_eff(10, N, L)
    assert isinstance(g_scalar, float)
    # Array
    g_arr = lam_engine.gamma_eff([10, 20], N, L)
    assert g_arr.shape == (2,)
    
    # Invert P
    P_inv = lam_engine.P_for_gamma(g_scalar, N, L)
    assert isinstance(P_inv, int)
    
    # Check max and min bounds
    P_min = lam_engine.P_for_gamma(-10.0, N, L, P_min=5)
    assert P_min == 5
    P_max = lam_engine.P_for_gamma(100.0, N, L, P_max=50)
    assert P_max == 50


def test_m1_theory_gamma():
    m1 = lam_engine.m1_theory_gamma(0.5)
    assert isinstance(m1, float)
    m1_arr = lam_engine.m1_theory_gamma([0.5, 1.0])
    assert m1_arr.shape == (2,)


def test_N_for_log_capacity():
    N = lam_engine.N_for_log_capacity(2, 5.0, N_min=10)
    assert N >= 10


def test_noise_var_per_pattern_typical():
    noise = lam_engine.noise_var_per_pattern_typical(L=2, N=8, n_calib=10, seed=42)
    assert noise > 0


def test_dynamics_and_overlaps():
    rng = np.random.default_rng(42)
    P, L, N = 3, 2, 8
    xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)
    
    # initial_state
    sig0_full = lam_engine.initial_state(xi, mu0=0, r=1.0, rng=rng)
    assert np.array_equal(sig0_full, xi[0])
    sig0_noisy = lam_engine.initial_state(xi, mu0=0, r=0.5, rng=rng)
    assert sig0_noisy.shape == (L, N)
    
    # shape mismatch local field
    with pytest.raises(ValueError):
        lam_engine.compute_local_field(xi, np.zeros((L, N+1)))
        
    H = lam_engine.compute_local_field(xi, sig0_noisy)
    assert H.shape == (L, N)
    
    sig1 = lam_engine.one_step_parallel(xi, sig0_noisy)
    assert sig1.shape == (L, N)
    assert set(np.unique(sig1)).issubset({-1, 1})
    
    # overlaps
    m_full = lam_engine.mattis_overlap(xi, sig1, mu0=0)
    assert isinstance(m_full, float)
    m_layer = lam_engine.mattis_overlap(xi, sig1, mu0=0, layer=0)
    assert isinstance(m_layer, float)


def test_mc_one_step():
    mean_val, sem_val = lam_engine.run_mc_one_step(8, 2, 3, r=1.0, n_trials=2, seed=42)
    assert isinstance(mean_val, float)
    assert isinstance(sem_val, float)


def test_theory_predictions():
    L, P, N = 2, 3, 8
    m1 = lam_engine.theory_m1_one_step(L, P, N, r=1.0)
    assert 0.0 <= m1 <= 1.0
    
    m1_no_noise = lam_engine.theory_m1_one_step(L, 1, N, r=1.0)
    assert m1_no_noise == 1.0
    
    noise_std = lam_engine.empirical_noise_std_mc(L, N, P, n_trials=2, seed=42)
    assert noise_std >= 0.0
    
    basin = lam_engine.theory_basin_finiteN(L, P, N, r=1.0, sigma2_per_pattern=0.1)
    assert 0.0 <= basin <= 1.0
    
    # Edge case: L=1
    basin_L1 = lam_engine.theory_basin_finiteN(1, P, N, r=1.0, sigma2_per_pattern=0.1)
    assert 0.0 <= basin_L1 <= 1.0


def test_torch_helpers():
    avail = lam_engine_torch.torch_available()
    assert isinstance(avail, bool)
    if avail:
        xi = np.ones((2, 2, 4), dtype=np.int8)
        t = lam_engine_torch.to_xi_tensor(xi, device="cpu")
        assert t.shape == (2, 2, 4)


def test_torch_dynamics():
    P, L, N, B = 2, 2, 8, 3
    rng = np.random.default_rng(42)
    xi = rng.choice([-1, 1], size=(P, L, N)).astype(np.int8)
    sig0 = rng.choice([-1, 1], size=(B, L, N)).astype(np.int8)
    
    # Auto loop (CPU fallback / Torch CPU depending on environment)
    sig1 = lam_engine_torch.iterate_parallel_auto(xi, sig0, n_steps=1, device="cpu")
    assert sig1.shape == (B, L, N)
    
    # Clamping
    sig2 = lam_engine_torch.iterate_parallel_auto(xi, sig0, n_steps=2, clamp_layers=[0], device="cpu")
    assert np.array_equal(sig2[:, 0, :], sig0[:, 0, :])
    
    # Cue seed
    cue_sig = lam_engine_torch.cue_seed_auto(xi, sig0, known_layers=[0], device="cpu")
    assert np.array_equal(cue_sig[:, 0, :], sig0[:, 0, :])
    
    # Test numpy explicit path
    sig_np = lam_engine_torch.iterate_parallel_batched_numpy(xi, sig0, n_steps=1, clamp_layers=[0])
    assert np.array_equal(sig_np[:, 0, :], sig0[:, 0, :])
    
    cue_np = lam_engine_torch.cue_seed_numpy(xi, sig0, known_layers=[0])
    assert np.array_equal(cue_np[:, 0, :], sig0[:, 0, :])

def test_theory_basin_finiteN_empty_K():
    L, P, N = 2, 3, 8
    # prob_threshold > 1.0 will cause all probabilities to be filtered out
    basin = lam_engine.theory_basin_finiteN(L, P, N, r=1.0, prob_threshold=2.0)
    assert basin == 0.0

def test_no_torch(monkeypatch):
    monkeypatch.setattr(lam_engine_torch, '_HAS_TORCH', False)
    
    with pytest.raises(RuntimeError):
        lam_engine_torch.to_xi_tensor(np.ones((2, 2, 4), dtype=np.int8))
        
    rng = np.random.default_rng(42)
    xi = rng.choice([-1, 1], size=(2, 2, 8)).astype(np.int8)
    sig0 = rng.choice([-1, 1], size=(3, 2, 8)).astype(np.int8)
    
    # Should fallback to numpy
    sig1 = lam_engine_torch.iterate_parallel_auto(xi, sig0, n_steps=1)
    assert sig1.shape == (3, 2, 8)
    
    cue = lam_engine_torch.cue_seed_auto(xi, sig0, known_layers=[0])
    assert cue.shape == (3, 2, 8)
