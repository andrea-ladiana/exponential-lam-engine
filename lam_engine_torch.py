"""
lam_engine_torch.py
====================
GPU-batched LAM dynamics (PyTorch) with a transparent NumPy fallback.

Runs unmodified on CPU-only machines (the NumPy fallback path, `*_numpy` functions)
and on a CUDA GPU when available.
"""
from __future__ import annotations

import logging

import numpy as np

from config import DEVICE, SITE_CHUNK

_DEFAULT_CHUNK = SITE_CHUNK if SITE_CHUNK is not None else 16

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
def torch_available() -> bool:
    return _HAS_TORCH


def to_xi_tensor(xi: np.ndarray, device: str = DEVICE):
    """Move a (P,L,N) int8 memory tensor to the device as float32 (+-1)."""
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not available.")
    return torch.as_tensor(xi, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Batched one-step parallel dynamics (GPU)
# ---------------------------------------------------------------------------
def one_step_parallel_batched(xi_t, sigma_t, site_chunk: int = _DEFAULT_CHUNK):
    """One synchronous parallel update of B states. Returns (B,L,N) float32 +-1."""
    P, L, N = xi_t.shape
    B = sigma_t.shape[0]
    binomL2 = L * (L - 1) / 2.0

    # Full overlaps m[b,p,a] = <xi[p,a], sigma[b,a]> / N
    m = torch.einsum("pln,bln->bpl", xi_t, sigma_t) / N      # (B,P,L)

    new_sigma = torch.empty_like(sigma_t)
    for s in range(0, N, site_chunk):
        e = min(s + site_chunk, N)
        xi_c = xi_t[:, :, s:e]                                # (P,L,C)
        sig_c = sigma_t[:, :, s:e]                            # (B,L,C)
        prod = xi_c.unsqueeze(0) * sig_c.unsqueeze(1)         # (B,P,L,C)
        mhat = m.unsqueeze(-1) - prod / N                     # (B,P,L,C)
        rowsum = mhat.sum(dim=2, keepdim=True)                # (B,P,1,C)
        F = rowsum - mhat                                     # (B,P,L,C)
        sum_m = rowsum[:, :, 0, :]                            # (B,P,C)
        sum_m2 = (mhat * mhat).sum(dim=2)                     # (B,P,C)
        Ehat = 0.5 * N * (sum_m ** 2 - sum_m2) - N * binomL2  # (B,P,C)
        S = (prod * F).sum(dim=2)                             # (B,P,C)
        for a in range(L):
            Fa = F[:, :, a, :]                                # (B,P,C)
            prod_a = prod[:, :, a, :]                         # (B,P,C)
            arg = Ehat + (S - prod_a * Fa)                    # (B,P,C)
            mx = arg.amax(dim=1, keepdim=True)                # (B,1,C)
            w = torch.exp(arg - mx) * torch.sinh(xi_c[:, a, :].unsqueeze(0) * Fa)
            Hs = w.sum(dim=1)                                 # (B,C)
            sgn = torch.sign(Hs)
            sgn[sgn == 0] = 1.0
            new_sigma[:, a, s:e] = sgn
    return new_sigma


# ---------------------------------------------------------------------------
# Cue-driven seed of the unknown layers
# ---------------------------------------------------------------------------
def cue_seed_batched(xi_t, cue_t, known_layers):
    """Cue-driven seed for the UNKNOWN layers (batched, GPU)."""
    P, L, N = xi_t.shape
    binomL2 = L * (L - 1) / 2.0
    known = list(known_layers)
    unknown = [a for a in range(L) if a not in known]

    m = torch.einsum("pln,bln->bpl", xi_t, cue_t) / N      # (B,P,L)
    mask = torch.zeros(L, dtype=m.dtype, device=m.device)
    mask[known] = 1.0
    m = m * mask.view(1, 1, L)                             # zero unknown overlaps
    sum_m = m.sum(dim=2)                                   # (B,P)
    sum_m2 = (m * m).sum(dim=2)
    Ehat = 0.5 * N * (sum_m ** 2 - sum_m2) - N * binomL2   # (B,P)
    F = m.sum(dim=2, keepdim=True) - m                     # (B,P,L)
    mx = Ehat.amax(dim=1, keepdim=True)                    # (B,1)
    w = torch.exp(Ehat - mx)                               # (B,P)

    out = cue_t.clone()
    for a in unknown:
        coeff = w * torch.sinh(F[:, :, a])                 # (B,P)
        H = torch.einsum("bp,pn->bn", coeff, xi_t[:, a, :])  # (B,N)
        s = torch.sign(H)
        s = torch.where(s == 0, torch.ones_like(s), s)
        out[:, a, :] = s
    return out


def cue_seed_numpy(xi: np.ndarray, cue: np.ndarray, known_layers):
    """NumPy version of cue_seed_batched."""
    P, L, N = xi.shape
    binomL2 = L * (L - 1) / 2.0
    known = list(known_layers)
    unknown = [a for a in range(L) if a not in known]
    xf = xi.astype(np.float64)
    m = np.einsum("pln,bln->bpl", xf, cue.astype(np.float64)) / N
    mask = np.zeros(L)
    mask[known] = 1.0
    m = m * mask.reshape(1, 1, L)
    sum_m = m.sum(2)
    sum_m2 = (m * m).sum(2)
    Ehat = 0.5 * N * (sum_m ** 2 - sum_m2) - N * binomL2
    F = m.sum(2, keepdims=True) - m
    w = np.exp(Ehat - Ehat.max(axis=1, keepdims=True))
    out = cue.copy().astype(np.int8)
    for a in unknown:
        coeff = w * np.sinh(F[:, :, a])
        H = coeff @ xf[:, a, :]
        s = np.sign(H).astype(np.int8)
        s[s == 0] = 1
        out[:, a, :] = s
    return out


# ---------------------------------------------------------------------------
# Iterated parallel dynamics (with cue clamping)
# ---------------------------------------------------------------------------
def iterate_parallel_batched(xi_t, sigma_t, n_steps: int, site_chunk: int = _DEFAULT_CHUNK,
                             clamp_layers=None):
    """Apply n_steps batched parallel updates."""
    s = sigma_t
    cue = None
    if clamp_layers:
        clamp_layers = list(clamp_layers)
        cue = sigma_t[:, clamp_layers, :].clone()
    for _ in range(n_steps):
        s = one_step_parallel_batched(xi_t, s, site_chunk=site_chunk)
        if clamp_layers:
            s[:, clamp_layers, :] = cue
    return s


# ---------------------------------------------------------------------------
# NumPy fallback (loops states through the reference field)
# ---------------------------------------------------------------------------
def iterate_parallel_batched_numpy(xi: np.ndarray, sigma_batch: np.ndarray,
                                   n_steps: int, clamp_layers=None) -> np.ndarray:
    """CPU fallback: (B,L,N) int8 -> (B,L,N) int8 after n_steps parallel steps."""
    from lam_engine import one_step_parallel

    clamp_layers = list(clamp_layers) if clamp_layers else None
    out = np.empty_like(sigma_batch)
    for b in range(sigma_batch.shape[0]):
        s = sigma_batch[b].astype(np.int8)
        cue = s[clamp_layers].copy() if clamp_layers else None
        for _ in range(n_steps):
            s = one_step_parallel(xi, s)
            if clamp_layers:
                s[clamp_layers] = cue
        out[b] = s
    return out


# ---------------------------------------------------------------------------
# Unified entry point: dispatches to GPU or NumPy transparently
# ---------------------------------------------------------------------------
def iterate_parallel_auto(xi: np.ndarray, sigma_batch: np.ndarray, n_steps: int,
                          clamp_layers=None, device: str = DEVICE,
                          site_chunk: int = _DEFAULT_CHUNK) -> np.ndarray:
    """Run the batched dynamics on device if torch+CUDA are available,
    otherwise fall back to the pure-NumPy loop. Always returns a NumPy array.
    """
    if device == "cuda" and _HAS_TORCH and torch.cuda.is_available():
        xi_t = to_xi_tensor(xi, device=device)
        sig_t = torch.as_tensor(sigma_batch, dtype=torch.float32, device=device)
        out_t = iterate_parallel_batched(xi_t, sig_t, n_steps, site_chunk=site_chunk,
                                         clamp_layers=clamp_layers)
        return out_t.cpu().numpy().astype(np.int8)
    if _HAS_TORCH:
        xi_t = to_xi_tensor(xi, device="cpu")
        sig_t = torch.as_tensor(sigma_batch, dtype=torch.float32, device="cpu")
        out_t = iterate_parallel_batched(xi_t, sig_t, n_steps, site_chunk=site_chunk,
                                         clamp_layers=clamp_layers)
        return out_t.cpu().numpy().astype(np.int8)
    return iterate_parallel_batched_numpy(xi, sigma_batch, n_steps, clamp_layers=clamp_layers)


def cue_seed_auto(xi: np.ndarray, cue: np.ndarray, known_layers, device: str = DEVICE) -> np.ndarray:
    """Dispatches cue_seed_batched (GPU) or cue_seed_numpy (CPU)."""
    if device == "cuda" and _HAS_TORCH and torch.cuda.is_available():
        xi_t = to_xi_tensor(xi, device=device)
        cue_t = torch.as_tensor(cue, dtype=torch.float32, device=device)
        return cue_seed_batched(xi_t, cue_t, known_layers).cpu().numpy().astype(np.int8)
    if _HAS_TORCH:
        xi_t = to_xi_tensor(xi, device="cpu")
        cue_t = torch.as_tensor(cue, dtype=torch.float32, device="cpu")
        return cue_seed_batched(xi_t, cue_t, known_layers).cpu().numpy().astype(np.int8)
    return cue_seed_numpy(xi, cue, known_layers)


# ---------------------------------------------------------------------------
# Unit test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from lam_engine import one_step_parallel

    rng = np.random.default_rng(0)
    for (P, L, N) in [(20, 2, 32), (50, 2, 64), (30, 3, 48), (25, 4, 40)]:
        xi = rng.choice(np.array([-1, 1], dtype=np.int8), size=(P, L, N))
        B = 7
        sig0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=(B, L, N))
        ref = np.stack([one_step_parallel(xi, sig0[b]) for b in range(B)])

        if _HAS_TORCH:
            xi_t = to_xi_tensor(xi, device="cpu")
            sig_t = torch.as_tensor(sig0, dtype=torch.float32)
            got = one_step_parallel_batched(xi_t, sig_t, site_chunk=8).cpu().numpy().astype(np.int8)
            ok = np.array_equal(got, ref)
            print(f"[torch ] P={P} L={L} N={N}: match={ok}")
            assert ok, "GPU batched dynamics disagree with reference!"
        got_np = iterate_parallel_batched_numpy(xi, sig0, n_steps=1)
        ok_np = np.array_equal(got_np, ref)
        print(f"[numpy ] P={P} L={L} N={N}: match={ok_np}")
        assert ok_np
    print("All dynamics-equivalence checks passed (CPU + GPU-if-available).")
