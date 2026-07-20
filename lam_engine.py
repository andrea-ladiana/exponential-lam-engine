"""
lam_engine.py
=============
Reference (NumPy) LAM engine + closed-form theory, valid for any L >= 2.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import brentq
from scipy.special import comb as scomb
from scipy.special import erf as sp_erf


# ===========================================================================
# Closed-form theory
# ===========================================================================
def saddle_point(L: int) -> dict:
    """Annealed saddle point for L layers. Requires L >= 2."""
    if L < 2:
        raise ValueError("LAM requires L >= 2.")
    a = 2.0 * (L - 1)
    x_star = brentq(lambda x: np.tanh(x) - x / a, 1e-6, 50.0)
    m_star = np.tanh(x_star)
    phi_L = -x_star ** 2 / (4.0 * (L - 1)) + np.log(np.cosh(x_star))
    rho_L = L * ((L - 1) - phi_L)
    r_c = np.sqrt(max(1.0 - rho_L / (L * (L - 1)), 0.0))
    return dict(x_star=x_star, m_star=m_star, phi_L=phi_L, rho_L=rho_L, r_c=r_c)


def mu1_signal(L: int) -> float:
    """Asymptotic one-step signal strength."""
    return float(np.exp(-(L - 1)) * np.sinh(L - 1))


def mu1_signal_finiteN(L: int, N: int) -> float:
    """Finite-N corrected one-step signal strength."""
    arg_exp = -(L - 1) - (L - 1) * (L - 2) / (2.0 * N)
    arg_sinh = (L - 1) * (N - 1) / N
    return float(np.exp(arg_exp) * np.sinh(arg_sinh))


def K_L_prefactor(L: int) -> float:
    sp = saddle_point(L)
    return float(np.sinh((L - 1) * sp["m_star"]) ** 2
                 * np.cosh(2.0 * (L - 1) * sp["m_star"]) ** (L - 1))


def noise_var_per_pattern_annealed(N: int, L: int) -> float:
    sp = saddle_point(L)
    return float(K_L_prefactor(L) * np.exp(-N * sp["rho_L"]))


# ===========================================================================
# Reduced-load (gamma) parameterisation
# ===========================================================================
def gamma_eff(P, N: int, L: int) -> float:
    """Dimensionless noise-to-signal ratio governing the asymptotic one-step transition."""
    sp = saddle_point(L)
    K_L = K_L_prefactor(L)
    mu1_sq = mu1_signal(L) ** 2
    P = np.atleast_1d(np.asarray(P, dtype=np.float64))
    g = np.maximum(P - 1.0, 0.0) * K_L * np.exp(-N * sp["rho_L"]) / mu1_sq
    return float(g) if g.size == 1 else g


def P_for_gamma(gamma: float, N: int, L: int, P_min: int = 2,
                P_max: float | None = None) -> int:
    """Invert gamma_eff: the (rounded) P such that gamma_eff(P, N, L) == gamma."""
    sp = saddle_point(L)
    K_L = K_L_prefactor(L)
    mu1_sq = mu1_signal(L) ** 2
    denom = K_L * np.exp(-N * sp["rho_L"])
    P = 1.0 + gamma * mu1_sq / max(denom, 1e-300)
    if P_max is not None:
        P = min(P, P_max)
    return max(P_min, int(round(P)))


def m1_theory_gamma(gamma) -> float:
    """Asymptotic one-step overlap as a pure function of the reduced load."""
    g = np.maximum(np.atleast_1d(np.asarray(gamma, dtype=np.float64)), 1e-300)
    m = sp_erf(1.0 / np.sqrt(2.0 * g))
    return float(m) if m.size == 1 else m


def N_for_log_capacity(L: int, log_P_target: float, N_min: int = 6) -> int:
    """Smallest N (floored at N_min) such that ln P_c(N,L) = N rho_L is approx log_P_target."""
    sp = saddle_point(L)
    N = int(round(log_P_target / sp["rho_L"]))
    return max(N_min, N)


def noise_var_per_pattern_typical(L: int, N: int, n_calib: int = 20000, seed: int = 42) -> float:
    """Monte-Carlo estimate of the typical per-pattern noise variance."""
    rng = np.random.default_rng(seed)
    xi1 = rng.choice([-1, 1], size=(L, N)).astype(np.int8)
    sigma = xi1.copy()
    binomL2 = L * (L - 1) // 2
    xi_mu = rng.choice([-1, 1], size=(n_calib, L, N)).astype(np.int8)
    a = 0
    mhat = (xi_mu[:, :, 1:] * sigma[None, :, 1:]).sum(axis=2) / N
    sum_m = mhat.sum(axis=1)
    sum_m2 = (mhat ** 2).sum(axis=1)
    Ehat = 0.5 * N * (sum_m ** 2 - sum_m2) - N * binomL2
    F = mhat.sum(axis=1, keepdims=True) - mhat
    S = (xi_mu[:, :, 0] * sigma[None, :, 0] * F).sum(axis=1)
    logPhi = S - xi_mu[:, a, 0] * sigma[a, 0] * F[:, a]
    sh = np.sinh(xi_mu[:, a, 0] * F[:, a])
    X = xi1[a, 0] * xi_mu[:, a, 0] * np.exp(Ehat + logPhi) * sh
    return float((X ** 2).mean())


# ===========================================================================
# Reference dynamics (single state)
# ===========================================================================
def compute_local_field(xi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Exact one-step local field H (shape (L,N)) for a single state sigma."""
    P, L, N = xi.shape
    if sigma.shape != (L, N):
        raise ValueError(f"sigma shape {sigma.shape} != (L,N)=({L},{N})")
    binomL2 = L * (L - 1) // 2
    xf = xi.astype(np.float64)
    sf = sigma.astype(np.float64)
    m = (xf * sf[None, :, :]).sum(axis=2) / N
    H = np.zeros((L, N), dtype=np.float64)
    for i in range(N):
        xi_i = xf[:, :, i]
        sig_i = sf[:, i]
        mhat = m - xi_i * sig_i[None, :] / N
        rowsum = mhat.sum(axis=1, keepdims=True)
        F = rowsum - mhat
        sum_m = mhat.sum(axis=1)
        sum_m2 = (mhat ** 2).sum(axis=1)
        Ehat = 0.5 * N * (sum_m ** 2 - sum_m2) - N * binomL2
        S = (xi_i * sig_i[None, :] * F).sum(axis=1)
        for a in range(L):
            logPhi_a = S - xi_i[:, a] * sig_i[a] * F[:, a]
            arg = Ehat + logPhi_a
            mx = arg.max()
            eA = np.exp(arg - mx)
            sh = np.sinh(xi_i[:, a] * F[:, a])
            H[a, i] = np.exp(mx) * (eA * sh).sum()
    return H


def one_step_parallel(xi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Computes one synchronous update of the spins."""
    H = compute_local_field(xi, sigma)
    sn = np.sign(H).astype(np.int8)
    sn[sn == 0] = 1
    return sn


def initial_state(xi, mu0=0, r=1.0, rng=None) -> np.ndarray:
    """Generates an initial state with a specified overlap r to pattern mu0."""
    rng = rng or np.random.default_rng()
    L, N = xi.shape[1], xi.shape[2]
    sigma = xi[mu0].copy().astype(np.int8)
    if r >= 1.0 - 1e-12:
        return sigma
    p_flip = 0.5 * (1.0 - r)
    flip = rng.random(size=(L, N)) < p_flip
    return np.where(flip, -sigma, sigma).astype(np.int8)


def mattis_overlap(xi, sigma, mu0=0, layer=None) -> float:
    """Mattis overlap m^mu_a = <xi^mu_a, sigma_a> / N."""
    L, N = sigma.shape
    if layer is not None:
        xi_l = xi[mu0, layer].astype(np.int32)
        sig_l = np.asarray(sigma[layer], dtype=np.int32)
        return float((xi_l * sig_l).sum() / N)
    xi_m = xi[mu0].astype(np.int32)
    sig_m = np.asarray(sigma, dtype=np.int32)
    return float((xi_m * sig_m).sum() / (L * N))


def run_mc_one_step(N, L, P, r=1.0, n_trials=30, seed=None):
    """MC mean +/- SEM of the one-step overlap on synthetic i.i.d. Rademacher memory."""
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n_trials):
        xi = rng.choice(np.array([-1, 1], dtype=np.int8), size=(P, L, N))
        sigma0 = initial_state(xi, mu0=0, r=r, rng=rng)
        sigma1 = one_step_parallel(xi, sigma0)
        res.append(mattis_overlap(xi, sigma1, mu0=0))
    a = np.array(res)
    return float(a.mean()), float(a.std() / max(np.sqrt(len(a)), 1))


# ===========================================================================
# Finite-N predictions
# ===========================================================================
def theory_m1_one_step(L, P, N, r=1.0, sigma2_per_pattern=None,
                       use_finiteN_signal=True) -> float:
    mu1 = mu1_signal_finiteN(L, N) if use_finiteN_signal else mu1_signal(L)
    if sigma2_per_pattern is None:
        sigma2_per_pattern = noise_var_per_pattern_typical(L, N)
    sig2 = max(P - 1, 0) * sigma2_per_pattern
    if sig2 <= 0:
        return 1.0
    return float(sp_erf(mu1 / np.sqrt(2.0 * sig2)))


def empirical_noise_std_mc(L, N, P, r=1.0, n_trials=30, seed=42) -> float:
    rng_master = np.random.default_rng(seed)
    samples = []
    for trial in range(n_trials):
        rng_t = np.random.default_rng(seed + trial * 1000)
        xi = rng_t.choice([-1, 1], size=(P, L, N)).astype(np.int8)
        sigma0 = initial_state(xi, mu0=0, r=r, rng=rng_t)
        H_noise = compute_local_field(xi[1:], sigma0) if P > 1 else np.zeros((L, N))
        samples.append((xi[0] * H_noise).flatten())
    return float(np.concatenate(samples).std())


def theory_basin_finiteN(L, P, N, r, sigma2_per_pattern=None,
                         prob_threshold=1e-12) -> float:
    """Vectorised finite-N basin theory (probability of correct retrieval)."""
    Nm1 = N - 1
    p_succ = (1.0 + r) / 2.0
    ks = np.arange(Nm1 + 1)
    binom_probs = (scomb(Nm1, ks, exact=False)
                   * (p_succ ** ks) * ((1 - p_succ) ** (Nm1 - ks)))
    mhat_vals = (2 * ks - Nm1) / float(N)

    if sigma2_per_pattern is None:
        sigma_noise = empirical_noise_std_mc(L, N, P, r=r, n_trials=1000)
    else:
        sigma_noise = np.sqrt(max((P - 1) * sigma2_per_pattern, 1e-300))
    binomL2 = L * (L - 1) / 2.0
    sqrt2 = np.sqrt(2.0)

    eps = max(prob_threshold ** (1.0 / L), 1e-8)
    keep = np.where(binom_probs >= eps)[0]
    if len(keep) == 0:
        keep = np.array([int(np.argmax(binom_probs))])
    bk = binom_probs[keep]
    mk = mhat_vals[keep]
    Nk = len(keep)

    grids = np.meshgrid(*[np.arange(Nk)] * L, indexing="ij")
    K = np.stack([g.reshape(-1) for g in grids], axis=1)
    prob_k = np.prod(bk[K], axis=1)
    mask = prob_k >= prob_threshold
    K, prob_k = K[mask], prob_k[mask]
    if len(K) == 0:
        return 0.0
    mhat = mk[K]
    rowsum = mhat.sum(axis=1, keepdims=True)
    F = rowsum - mhat
    sum_m = mhat.sum(axis=1)
    sum_m2 = (mhat ** 2).sum(axis=1)
    Ehat = 0.5 * N * (sum_m ** 2 - sum_m2) - N * binomL2

    if L == 1:
        S = np.zeros((1, 0))
    else:
        S = np.array(list(itertools.product([-1, 1], repeat=L - 1)), dtype=np.float64)
    pp, pn = (1.0 + r) / 2.0, (1.0 - r) / 2.0
    p_s = (pp ** (S == 1).sum(axis=1)) * (pn ** (S == -1).sum(axis=1))
    F_other = F[:, 1:] if L > 1 else np.zeros((F.shape[0], 0))
    arg = S @ F_other.T if L > 1 else np.zeros((1, F.shape[0]))
    X_sig = np.sinh(F[:, 0])[None, :] * np.exp(Ehat[None, :] + arg)
    local_m = (sp_erf(X_sig / (sqrt2 * sigma_noise))
               if sigma_noise > 1e-300 else np.sign(X_sig))
    weighted = (p_s[:, None] * local_m).sum(axis=0)
    return float((prob_k * weighted).sum())


if __name__ == "__main__":  # pragma: no cover
    print("=== LAM engine (reference) self-test ===")
    for L in (2, 3, 4):
        sp = saddle_point(L)
        print("L=%d  x*=%.4f  rho_L=%.4f  r_c=%.4f  mu1=%.6f  mu1_finN(128)=%.6f"
              % (L, sp["x_star"], sp["rho_L"], sp["r_c"],
                 mu1_signal(L), mu1_signal_finiteN(L, 128)))
