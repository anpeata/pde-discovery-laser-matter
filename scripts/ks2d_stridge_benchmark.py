"""Benchmark: recover 2D Kuramoto–Sivashinsky (KS) PDE via STRidge.

Matches the KS generator used in notebooks/sind_without_pysindy_on_2d.ipynb:
  u_t = -∇²u - ∇⁴u - 0.5 |∇u|²

This script:
- Simulates 2D KS with explicit Euler and periodic BCs
- Builds a PDE-FIND dictionary (true terms + decoys)
- Fits coefficients via STRidge (sequential thresholded ridge)
- Reports coefficient errors vs ground truth + basic rollout RMSE

Run:
  python scripts/ks2d_stridge_benchmark.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-18))


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, scale) for per-column standardization."""
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return mean, scale


def standardize_transform(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - mean) / scale


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression with no intercept: argmin ||Xb-y||^2 + alpha||b||^2."""
    XtX = X.T @ X
    Xty = X.T @ y
    p = XtX.shape[0]
    return np.linalg.solve(XtX + alpha * np.eye(p), Xty)


def laplacian(f2d: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return (
        (np.roll(f2d, -1, axis=0) - 2 * f2d + np.roll(f2d, 1, axis=0)) / (dx**2)
        + (np.roll(f2d, -1, axis=1) - 2 * f2d + np.roll(f2d, 1, axis=1)) / (dy**2)
    )


def gradients(f2d: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    gx = (np.roll(f2d, -1, axis=0) - np.roll(f2d, 1, axis=0)) / (2 * dx)
    gy = (np.roll(f2d, -1, axis=1) - np.roll(f2d, 1, axis=1)) / (2 * dy)
    return gx, gy


def _spectral_grids(nx: int, ny: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (KX, KY) wavenumber grids in rad / physical-unit."""
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return KX, KY


def _spectral_mask(KX: np.ndarray, KY: np.ndarray, cutoff_frac: float) -> np.ndarray:
    """Radial low-pass mask, cutoff_frac in (0, 1]."""
    cutoff_frac = float(cutoff_frac)
    if cutoff_frac >= 1.0:
        return np.ones_like(KX, dtype=np.float64)
    if cutoff_frac <= 0.0:
        raise ValueError("spectral cutoff must be > 0")
    k_mag = np.sqrt(KX**2 + KY**2)
    k_max = float(np.max(k_mag))
    return (k_mag <= cutoff_frac * k_max).astype(np.float64)


def gradients_spectral(f2d: np.ndarray, dx: float, dy: float, *, cutoff_frac: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Spectral (FFT) gradients with optional radial low-pass cutoff."""
    nx, ny = f2d.shape
    KX, KY = _spectral_grids(nx, ny, dx=dx, dy=dy)
    mask = _spectral_mask(KX, KY, cutoff_frac=cutoff_frac)
    F = np.fft.fft2(f2d.astype(np.float64, copy=False)) * mask
    gx = np.fft.ifft2(1j * KX * F).real
    gy = np.fft.ifft2(1j * KY * F).real
    return gx.astype(np.float64, copy=False), gy.astype(np.float64, copy=False)


def laplacian_spectral(f2d: np.ndarray, dx: float, dy: float, *, cutoff_frac: float = 1.0) -> np.ndarray:
    """Spectral (FFT) Laplacian with optional radial low-pass cutoff."""
    nx, ny = f2d.shape
    KX, KY = _spectral_grids(nx, ny, dx=dx, dy=dy)
    mask = _spectral_mask(KX, KY, cutoff_frac=cutoff_frac)
    K2 = KX**2 + KY**2
    F = np.fft.fft2(f2d.astype(np.float64, copy=False)) * mask
    out = np.fft.ifft2(-K2 * F).real
    return out.astype(np.float64, copy=False)


def ks_rhs(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    lap = laplacian(u, dx=dx, dy=dy)
    bih = laplacian(lap, dx=dx, dy=dy)
    ux, uy = gradients(u, dx=dx, dy=dy)
    return -lap - bih - 0.5 * (ux**2 + uy**2)


def gaussian_smooth_periodic_2d(frame: np.ndarray, sigma_px: float) -> np.ndarray:
    """Periodic Gaussian low-pass filter using FFT.

    sigma_px is the Gaussian std in pixel units.
    """
    sigma_px = float(sigma_px)
    if sigma_px <= 0:
        return frame.astype(np.float64, copy=True)

    nx, ny = frame.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    H = np.exp(-0.5 * (sigma_px**2) * (KX**2 + KY**2))

    F = np.fft.fft2(frame.astype(np.float64, copy=False))
    out = np.fft.ifft2(F * H).real
    return out.astype(np.float64, copy=False)


def time_smooth_moving_average(U: np.ndarray, window: int) -> np.ndarray:
    """Time-domain moving average (reflect-padded) along axis 0.

    Returns an array with the same shape as U.
    """
    window = int(window)
    if window <= 1:
        return U.astype(np.float64, copy=True)
    if window % 2 == 0:
        raise ValueError("time smoothing window must be odd")

    pad = window // 2
    U_pad = np.pad(U.astype(np.float64, copy=False), ((pad, pad), (0, 0), (0, 0)), mode="reflect")
    zero = np.zeros_like(U_pad[:1])
    cs = np.concatenate([zero, np.cumsum(U_pad, axis=0)], axis=0)
    out = (cs[window:] - cs[:-window]) / float(window)
    return out


def _fourier_test_functions(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    *,
    max_k: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Build a small real Fourier test-function basis (cos/sin) on a periodic grid.

    Returns (phis, k2, k4), where each phi is (nx, ny) and k2/k4 are arrays aligned with phis.
    We exclude the constant (0,0) mode.
    """
    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    phis: list[np.ndarray] = []
    k2_list: list[float] = []
    k4_list: list[float] = []

    for m in range(0, int(max_k) + 1):
        for n in range(0, int(max_k) + 1):
            if m == 0 and n == 0:
                continue
            kx = 2.0 * np.pi * m / float(lx)
            ky = 2.0 * np.pi * n / float(ly)
            k2 = float(kx**2 + ky**2)
            k4 = float(k2**2)

            phase = kx * X + ky * Y
            phis.append(np.cos(phase).astype(np.float64))
            k2_list.append(k2)
            k4_list.append(k4)
            phis.append(np.sin(phase).astype(np.float64))
            k2_list.append(k2)
            k4_list.append(k4)

    return phis, np.asarray(k2_list, dtype=np.float64), np.asarray(k4_list, dtype=np.float64)


def _gaussian_test_functions(
    nx: int,
    ny: int,
    *,
    n_phi: int,
    sigma_px: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Localized Gaussian test functions on a periodic grid (wrap-around distance)."""
    sigma_px = float(sigma_px)
    if sigma_px <= 0:
        raise ValueError("weak sigma_px must be > 0")
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")

    phis: list[np.ndarray] = []
    for _ in range(int(n_phi)):
        cx = float(rng.uniform(0, nx))
        cy = float(rng.uniform(0, ny))

        dxp = np.minimum(np.abs(X - cx), nx - np.abs(X - cx))
        dyp = np.minimum(np.abs(Y - cy), ny - np.abs(Y - cy))
        r2 = dxp**2 + dyp**2
        phi = np.exp(-0.5 * r2 / (sigma_px**2))
        # Normalize to unit L2 to reduce scaling issues across test functions.
        norm = float(np.sqrt(np.sum(phi**2)))
        if norm > 0:
            phi = phi / norm
        phis.append(phi.astype(np.float64, copy=False))
    return phis


def build_weakform_dataset(
    U: np.ndarray,
    *,
    dx: float,
    dy: float,
    dt_frame: float,
    lx: float,
    ly: float,
    max_k: int,
    basis: str,
    n_phi: int,
    sigma_px: float,
    grad_cutoff: float,
    motion_correct: bool = False,
    motion_est_sigma_px: float = 0.0,
    motion_smooth_window: int = 1,
    motion_clip_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Weak-form dataset for KS terms using Fourier test functions.

    Target uses time-difference of integrated coefficients:
      y = (⟨phi, u_{k+1}⟩ - ⟨phi, u_k⟩) / DT
    Linear terms use integration-by-parts (derivatives on phi):
      ⟨phi, ∇²u⟩ = ⟨u, ∇²phi⟩ = -k^2 ⟨phi, u⟩
      ⟨phi, ∇⁴u⟩ = ⟨u, ∇⁴phi⟩ = +k^4 ⟨phi, u⟩
    Nonlinear term still needs ∇u; we compute it spectrally with a low-pass cutoff.
    """
    if U.ndim != 3:
        raise ValueError("U must be (T, Nx, Ny)")
    t_len, nx, ny = U.shape
    if t_len < 2:
        raise ValueError("Need at least 2 frames")

    basis = str(basis)
    if basis == "fourier":
        phis, k2, k4 = _fourier_test_functions(nx, ny, lx=lx, ly=ly, max_k=max_k)
    elif basis == "gaussian":
        rng_phi = np.random.default_rng(123)
        phis = _gaussian_test_functions(nx, ny, n_phi=int(n_phi), sigma_px=float(sigma_px), rng=rng_phi)
        k2 = np.zeros(len(phis), dtype=np.float64)
        k4 = np.zeros(len(phis), dtype=np.float64)
    else:
        raise ValueError("weak basis must be 'fourier' or 'gaussian'")
    n_phi_eff = len(phis)
    area = float(dx * dy)

    # Vectorize inner products: S[t,j] = <phi_j, u_t>.
    phi_stack = np.stack([p.astype(np.float64, copy=False) for p in phis], axis=0)  # (P,Nx,Ny)
    phi_flat_T = phi_stack.reshape(n_phi_eff, -1).T  # (Nx*Ny, P)
    U_flat = U.astype(np.float64, copy=False).reshape(t_len, -1)  # (T,Nx*Ny)
    S = area * (U_flat @ phi_flat_T)  # (T,P)

    # y[t,j] = (S[t+1,j]-S[t,j])/DT  approximates <phi, u_t>.
    y = (S[1:] - S[:-1]) / float(dt_frame)  # (T-1,P)

    # Optional motion-aware correction for global translation jitter.
    # If observed frames include camera motion modeled as advection
    #   u_t + v(t)·∇u = RHS(u)
    # then the weak-form identity gives (periodic BCs):
    #   <phi, u_t> - <u, v·∇phi> = <phi, RHS(u)>
    # This avoids putting derivatives on noisy u in the motion term.
    if bool(motion_correct):
        sx_px, sy_px = estimate_interframe_shifts(U, estimate_sigma_px=float(motion_est_sigma_px))
        sx_px = smooth_1d(sx_px, window=int(motion_smooth_window))
        sy_px = smooth_1d(sy_px, window=int(motion_smooth_window))

        clip_px = motion_clip_px
        if clip_px is not None:
            clip_px = float(clip_px)
            if clip_px > 0:
                sx_px = np.clip(sx_px, -clip_px, clip_px)
                sy_px = np.clip(sy_px, -clip_px, clip_px)

        vx = (-sx_px * float(dx)) / float(dt_frame)
        vy = (-sy_px * float(dy)) / float(dt_frame)

        phi_x = np.empty_like(phi_stack)
        phi_y = np.empty_like(phi_stack)
        for j in range(n_phi_eff):
            phi_x[j], phi_y[j] = gradients_spectral(phi_stack[j], dx=dx, dy=dy, cutoff_frac=1.0)

        phi_x_flat_T = phi_x.reshape(n_phi_eff, -1).T
        phi_y_flat_T = phi_y.reshape(n_phi_eff, -1).T
        U_flat_k = U_flat[:-1]  # (T-1,N)
        U_phi_x = area * (U_flat_k @ phi_x_flat_T)  # (T-1,P)
        U_phi_y = area * (U_flat_k @ phi_y_flat_T)  # (T-1,P)

        y = y - (vx[:, None] * U_phi_x + vy[:, None] * U_phi_y)

    # Features at time t: lap and bih.
    if basis == "fourier":
        X_lap_m = -S[:-1] * k2[None, :]
        X_bih_m = S[:-1] * k4[None, :]
    else:
        # For localized basis, compute lap(phi) and bih(phi) spectrally once.
        lap_phi = np.empty_like(phi_stack)
        bih_phi = np.empty_like(phi_stack)
        for j in range(n_phi_eff):
            lap_phi[j] = laplacian_spectral(phi_stack[j], dx=dx, dy=dy, cutoff_frac=1.0)
            bih_phi[j] = laplacian_spectral(lap_phi[j], dx=dx, dy=dy, cutoff_frac=1.0)
        lap_flat_T = lap_phi.reshape(n_phi_eff, -1).T
        bih_flat_T = bih_phi.reshape(n_phi_eff, -1).T
        U_flat_k = U_flat[:-1]
        X_lap_m = area * (U_flat_k @ lap_flat_T)
        X_bih_m = area * (U_flat_k @ bih_flat_T)

    # Nonlinear feature: <phi, |∇u|^2>.
    X_gsq_m = np.empty((t_len - 1, n_phi_eff), dtype=np.float64)
    for k in range(t_len - 1):
        u = U[k].astype(np.float64, copy=False)
        ux, uy = gradients_spectral(u, dx=dx, dy=dy, cutoff_frac=float(grad_cutoff))
        gs = (ux**2 + uy**2).reshape(-1)
        X_gsq_m[k] = area * (gs @ phi_flat_T)

    X = np.column_stack([X_lap_m.reshape(-1), X_bih_m.reshape(-1), X_gsq_m.reshape(-1)])
    y_flat = y.reshape(-1)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y_flat)
    return X[valid], y_flat[valid]


def build_blockwise_dataset(
    Ut: np.ndarray,
    terms: dict[str, np.ndarray],
    names: list[str],
    *,
    block_t: int,
    block_x: int,
    block_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Block-integrated regression dataset (bridge between pointwise and weak-form).

    We average u_t and each library term over spatiotemporal blocks.
    This reduces variance from noise/registration without changing the dictionary.
    """
    if Ut.ndim != 3:
        raise ValueError("Ut must be (T, Nx, Ny)")
    T, nx, ny = Ut.shape
    bt = int(block_t)
    bx = int(block_x)
    by = int(block_y)
    if bt <= 0 or bx <= 0 or by <= 0:
        raise ValueError("block sizes must be > 0")

    rows: list[np.ndarray] = []
    ys: list[float] = []
    for t0 in range(0, T, bt):
        t1 = min(T, t0 + bt)
        for x0 in range(0, nx, bx):
            x1 = min(nx, x0 + bx)
            for y0 in range(0, ny, by):
                y1 = min(ny, y0 + by)
                y_block = float(np.mean(Ut[t0:t1, x0:x1, y0:y1]))
                x_block = np.array(
                    [float(np.mean(terms[n][t0:t1, x0:x1, y0:y1])) for n in names],
                    dtype=np.float64,
                )
                if not (np.isfinite(y_block) and np.isfinite(x_block).all()):
                    continue
                ys.append(y_block)
                rows.append(x_block)

    if not rows:
        return np.zeros((0, len(names)), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.stack(rows, axis=0), np.asarray(ys, dtype=np.float64)


def stridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1e-3,
    threshold: float = 1e-6,
    max_iter: int = 25,
) -> np.ndarray:
    mean, scale = standardize_fit(X)
    Xs = standardize_transform(X, mean, scale)

    coeffs = ridge_fit(Xs, y, alpha=alpha).copy()

    for _ in range(max_iter):
        small = np.abs(coeffs) < threshold
        if small.all():
            coeffs[:] = 0.0
            break
        coeffs[small] = 0.0
        big = ~small
        coeffs_big = ridge_fit(Xs[:, big], y, alpha=alpha).copy()
        coeffs = np.zeros_like(coeffs)
        coeffs[big] = coeffs_big

    return coeffs / (scale + 1e-12)


# =============================================================================
# ROBUST REGRESSION METHODS (new)
# =============================================================================


def huber_weight(r: np.ndarray, delta: float = 1.35) -> np.ndarray:
    """Huber weight function: w = min(1, delta / |r|)."""
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 1.0, delta / (abs_r + 1e-12))


def irls_huber_fit(
    X: np.ndarray, y: np.ndarray, *, alpha: float = 1e-3, delta: float = 1.35, max_iter: int = 50, tol: float = 1e-6
) -> np.ndarray:
    """Iteratively Reweighted Least Squares with Huber loss.

    Solves: argmin sum(rho_huber(Xb - y)) + alpha * ||b||^2
    where rho_huber(r) = 0.5*r^2 if |r| <= delta, else delta*(|r| - delta/2).
    """
    n, p = X.shape
    beta = ridge_fit(X, y, alpha=alpha)

    for _ in range(max_iter):
        r = y - X @ beta
        sigma = float(np.median(np.abs(r)) * 1.4826 + 1e-12)  # robust scale (MAD)
        r_scaled = r / sigma
        w = huber_weight(r_scaled, delta=delta)

        # Weighted ridge: (X^T W X + alpha I) beta = X^T W y
        W = np.diag(w)
        XtWX = X.T @ W @ X
        XtWy = X.T @ (w * y)
        beta_new = np.linalg.solve(XtWX + alpha * np.eye(p), XtWy)

        if np.max(np.abs(beta_new - beta)) < tol:
            break
        beta = beta_new

    return beta


def stridge_huber(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1e-3,
    threshold: float = 1e-6,
    max_iter: int = 25,
    huber_delta: float = 1.35,
    huber_iter: int = 50,
) -> np.ndarray:
    """STRidge with Huber robust regression for inner solve.

    Drop-in replacement for stridge() that is robust to outliers.
    """
    mean, scale = standardize_fit(X)
    Xs = standardize_transform(X, mean, scale)

    coeffs = irls_huber_fit(Xs, y, alpha=alpha, delta=huber_delta, max_iter=huber_iter).copy()

    for _ in range(max_iter):
        small = np.abs(coeffs) < threshold
        if small.all():
            coeffs[:] = 0.0
            break
        coeffs[small] = 0.0
        big = ~small
        coeffs_big = irls_huber_fit(Xs[:, big], y, alpha=alpha, delta=huber_delta, max_iter=huber_iter).copy()
        coeffs = np.zeros_like(coeffs)
        coeffs[big] = coeffs_big

    return coeffs / (scale + 1e-12)


def trimmed_stridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1e-3,
    threshold: float = 1e-6,
    max_iter: int = 25,
    trim_frac: float = 0.1,
) -> np.ndarray:
    """STRidge with trimmed (outlier-robust) least squares.

    First fits ridge, then trims top trim_frac of residuals, and refits.
    """
    mean, scale = standardize_fit(X)
    Xs = standardize_transform(X, mean, scale)

    # Initial fit
    coeffs = ridge_fit(Xs, y, alpha=alpha).copy()

    # Trim outliers based on residuals
    n = len(y)
    n_trim = int(n * trim_frac)
    if n_trim > 0:
        resid = np.abs(y - Xs @ coeffs)
        keep_idx = np.argsort(resid)[: n - n_trim]
        Xs_trim = Xs[keep_idx]
        y_trim = y[keep_idx]
    else:
        Xs_trim = Xs
        y_trim = y

    coeffs = ridge_fit(Xs_trim, y_trim, alpha=alpha).copy()

    for _ in range(max_iter):
        small = np.abs(coeffs) < threshold
        if small.all():
            coeffs[:] = 0.0
            break
        coeffs[small] = 0.0
        big = ~small
        coeffs_big = ridge_fit(Xs_trim[:, big], y_trim, alpha=alpha).copy()
        coeffs = np.zeros_like(coeffs)
        coeffs[big] = coeffs_big

    return coeffs / (scale + 1e-12)


def stridge_sign_constrained(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1e-3,
    threshold: float = 1e-6,
    max_iter: int = 25,
    signs: list[int] | None = None,
) -> np.ndarray:
    """STRidge with physics-informed sign constraints.

    For KS equation, we know all coefficients should be negative: signs=[-1,-1,-1].
    If a coefficient has the wrong sign, it's forced to zero (treated as inactive).
    """
    mean, scale = standardize_fit(X)
    Xs = standardize_transform(X, mean, scale)
    p = X.shape[1]

    if signs is None:
        signs = [0] * p  # no constraint

    coeffs = ridge_fit(Xs, y, alpha=alpha).copy()

    for _ in range(max_iter):
        # Enforce sign constraints: if sign[j] != 0 and coeff[j] has wrong sign, zero it
        for j in range(p):
            if signs[j] == -1 and coeffs[j] > 0:
                coeffs[j] = 0.0
            elif signs[j] == 1 and coeffs[j] < 0:
                coeffs[j] = 0.0

        small = np.abs(coeffs) < threshold
        if small.all():
            coeffs[:] = 0.0
            break
        coeffs[small] = 0.0
        big = ~small
        coeffs_big = ridge_fit(Xs[:, big], y, alpha=alpha).copy()
        coeffs = np.zeros_like(coeffs)
        coeffs[big] = coeffs_big

        # Re-enforce sign constraints after refit
        for j in range(p):
            if signs[j] == -1 and coeffs[j] > 0:
                coeffs[j] = 0.0
            elif signs[j] == 1 and coeffs[j] < 0:
                coeffs[j] = 0.0

    return coeffs / (scale + 1e-12)


def ensemble_stridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1e-3,
    threshold: float = 1e-6,
    max_iter: int = 25,
    n_bootstrap: int = 50,
    subsample_frac: float = 0.7,
    seed: int = 0,
    use_huber: bool = False,
    huber_delta: float = 1.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble STRidge with bootstrap for robust coefficient estimation.

    Returns (mean_coeffs, std_coeffs) from n_bootstrap subsampled fits.
    Provides both robustness and uncertainty quantification.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    n_sub = max(int(n * subsample_frac), 1)
    p = X.shape[1]

    all_coeffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n_sub, replace=True)
        X_sub = X[idx]
        y_sub = y[idx]

        if use_huber:
            c = stridge_huber(X_sub, y_sub, alpha=alpha, threshold=threshold, max_iter=max_iter, huber_delta=huber_delta)
        else:
            c = stridge(X_sub, y_sub, alpha=alpha, threshold=threshold, max_iter=max_iter)
        all_coeffs.append(c)

    all_coeffs = np.stack(all_coeffs, axis=0)  # (n_bootstrap, p)

    # Use median instead of mean for robustness
    mean_coeffs = np.median(all_coeffs, axis=0)
    std_coeffs = np.std(all_coeffs, axis=0)

    return mean_coeffs, std_coeffs


def robust_stridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1e-3,
    threshold: float = 1e-6,
    max_iter: int = 25,
    use_huber: bool = True,
    huber_delta: float = 1.35,
    trim_frac: float = 0.05,
    n_bootstrap: int = 30,
    signs: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Full robust STRidge pipeline combining multiple robustness strategies.

    Combines:
    1. Trimmed least squares (remove worst outliers)
    2. Huber regression (downweight remaining outliers)
    3. Bootstrap ensemble (reduce variance, provide uncertainty)
    4. Sign constraints (physics-informed for KS: all negative)

    Returns:
        coeffs: Final robust coefficient estimates
        info: Dict with uncertainty and diagnostics
    """
    mean, scale = standardize_fit(X)
    Xs = standardize_transform(X, mean, scale)
    n, p = Xs.shape

    # Step 1: Initial fit to identify outliers
    c_init = ridge_fit(Xs, y, alpha=alpha)
    resid = np.abs(y - Xs @ c_init)

    # Step 2: Trim worst outliers
    n_trim = int(n * trim_frac)
    if n_trim > 0:
        keep_idx = np.argsort(resid)[: n - n_trim]
        Xs_clean = Xs[keep_idx]
        y_clean = y[keep_idx]
    else:
        Xs_clean = Xs
        y_clean = y

    # Step 3: Bootstrap ensemble with optional Huber
    rng = np.random.default_rng(42)
    n_clean = len(y_clean)
    all_coeffs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_clean, size=int(n_clean * 0.8), replace=True)
        X_sub = Xs_clean[idx]
        y_sub = y_clean[idx]

        if use_huber:
            c = irls_huber_fit(X_sub, y_sub, alpha=alpha, delta=huber_delta)
        else:
            c = ridge_fit(X_sub, y_sub, alpha=alpha)

        # STRidge thresholding
        for _ in range(max_iter):
            small = np.abs(c) < threshold
            if small.all():
                c = np.zeros(p)
                break
            c[small] = 0.0
            big = ~small
            if use_huber:
                c_big = irls_huber_fit(X_sub[:, big], y_sub, alpha=alpha, delta=huber_delta)
            else:
                c_big = ridge_fit(X_sub[:, big], y_sub, alpha=alpha)
            c = np.zeros(p)
            c[big] = c_big

        # Apply sign constraints
        if signs is not None:
            for j in range(p):
                if signs[j] == -1 and c[j] > 0:
                    c[j] = 0.0
                elif signs[j] == 1 and c[j] < 0:
                    c[j] = 0.0

        all_coeffs.append(c)

    all_coeffs = np.stack(all_coeffs, axis=0)

    # Robust aggregation: median
    coeffs = np.median(all_coeffs, axis=0) / (scale + 1e-12)
    std_coeffs = np.std(all_coeffs, axis=0) / (scale + 1e-12)

    # Compute confidence intervals (percentile bootstrap)
    ci_low = np.percentile(all_coeffs, 2.5, axis=0) / (scale + 1e-12)
    ci_high = np.percentile(all_coeffs, 97.5, axis=0) / (scale + 1e-12)

    info = {
        "std": std_coeffs,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "n_trimmed": n_trim,
        "n_bootstrap": n_bootstrap,
    }

    return coeffs, info


@dataclass(frozen=True)
class SimConfig:
    Lx: float = 50.0
    Ly: float = 50.0
    Nx: int = 100
    Ny: int = 100
    dt: float = 1e-3
    n_seconds: float = 2.0
    save_every: int = 1
    seed: int = 42


def simulate(cfg: SimConfig) -> tuple[np.ndarray, float, float, float]:
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny
    total_steps = int(cfg.n_seconds / cfg.dt)
    n_frames = total_steps // cfg.save_every
    DT = cfg.dt * cfg.save_every

    rng = np.random.default_rng(cfg.seed)
    u = rng.uniform(-0.1, 0.1, size=(cfg.Nx, cfg.Ny)).astype(np.float64)

    U = np.zeros((n_frames, cfg.Nx, cfg.Ny), dtype=np.float64)
    frame = 0
    for step in range(total_steps):
        u = u + cfg.dt * ks_rhs(u, dx=dx, dy=dy)
        u = np.nan_to_num(u)
        if step % cfg.save_every == 0:
            U[frame] = u
            frame += 1

    return U, dx, dy, DT


def _shift_frame_wrap(frame: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    """Subpixel shift with periodic wrapping (OpenCV)."""
    if cv2 is None:
        raise RuntimeError("cv2 is required for N1 shifts / N3 blur perturbations.")
    # OpenCV uses (x, y) = (col, row)
    M = np.array([[1.0, 0.0, float(shift_y)], [0.0, 1.0, float(shift_x)]], dtype=np.float32)
    out = cv2.warpAffine(
        frame.astype(np.float32),
        M,
        dsize=(frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return out.astype(np.float64)


def _blur_frame_wrap(frame: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with periodic wrapping (OpenCV)."""
    if cv2 is None:
        raise RuntimeError("cv2 is required for N3 blur perturbations.")
    # NOTE: OpenCV does not support BORDER_WRAP for GaussianBlur.
    # Emulate periodic BCs by wrap-padding, blurring, then cropping.
    pad = int(np.ceil(3.0 * float(sigma)))
    if pad <= 0:
        return frame.astype(np.float64, copy=True)

    padded = np.pad(frame, pad_width=((pad, pad), (pad, pad)), mode="wrap").astype(
        np.float32
    )
    blurred = cv2.GaussianBlur(
        padded,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_CONSTANT,
    )
    out = blurred[pad:-pad, pad:-pad]
    return out.astype(np.float64, copy=False)


def apply_perturbation_suite(
    U_clean: np.ndarray,
    *,
    perturbation: str,
    rng: np.random.Generator,
    noise_rel: float,
    shift_max_px: float,
    shift_mode: str,
    blur_sigma: float,
    drift_per_frame: float,
) -> np.ndarray:
    """Apply requested perturbations to observed frames (measurement corruptions)."""
    U = U_clean.astype(np.float64, copy=True)
    T = U.shape[0]

    def add_noise(arr: np.ndarray) -> np.ndarray:
        if noise_rel <= 0:
            return arr
        sigma0 = float(np.std(arr))
        sigma = float(noise_rel) * sigma0
        return arr + rng.normal(0.0, sigma, size=arr.shape)

    def add_shifts(arr: np.ndarray) -> np.ndarray:
        if shift_max_px <= 0:
            return arr
        out = np.empty_like(arr)
        mode = str(shift_mode)
        if mode not in {"constant", "jitter"}:
            raise ValueError("shift_mode must be 'constant' or 'jitter'")

        if mode == "constant":
            # A single global translation should not change the PDE, but may add small
            # interpolation error. This is the right model for camera drift.
            sx = float(rng.uniform(-shift_max_px, shift_max_px))
            sy = float(rng.uniform(-shift_max_px, shift_max_px))
            for t in range(T):
                out[t] = _shift_frame_wrap(arr[t], shift_x=sx, shift_y=sy)
            return out

        # Frame-to-frame jitter (strong corruption): dominates finite-difference u_t (shift/DT)
        # and is a good stress-test for robustness.
        for t in range(T):
            sx = float(rng.uniform(-shift_max_px, shift_max_px))
            sy = float(rng.uniform(-shift_max_px, shift_max_px))
            out[t] = _shift_frame_wrap(arr[t], shift_x=sx, shift_y=sy)
        return out

    def add_blur(arr: np.ndarray) -> np.ndarray:
        if blur_sigma <= 0:
            return arr
        out = np.empty_like(arr)
        for t in range(T):
            out[t] = _blur_frame_wrap(arr[t], sigma=blur_sigma)
        return out

    def add_drift(arr: np.ndarray) -> np.ndarray:
        if drift_per_frame <= 0:
            return arr
        # Intensity decay: multiply each frame by (1 - drift)^t
        factors = (1.0 - float(drift_per_frame)) ** np.arange(T, dtype=np.float64)
        return arr * factors[:, None, None]

    # Map to the requested suite
    if perturbation == "none":
        return U
    if perturbation == "N1_shifts":
        return add_shifts(U)
    if perturbation == "N2_noise":
        return add_noise(U)
    if perturbation == "N3_blur":
        return add_blur(U)
    if perturbation == "N4_drift":
        return add_drift(U)
    if perturbation == "N5_shifts_noise":
        return add_noise(add_shifts(U))
    if perturbation == "N6_blur_noise":
        return add_noise(add_blur(U))
    if perturbation == "N7_all":
        return add_noise(add_blur(add_drift(add_shifts(U))))
    raise ValueError(f"Unknown perturbation: {perturbation}")


def estimate_shift_phasecorr(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float]:
    """Estimate subpixel shift aligning mov to ref via phase correlation.

    Returns shifts in array-axis coordinates: (shift_x along axis=0, shift_y along axis=1)
    such that applying that translation to mov best matches ref.
    """
    if cv2 is not None:
        ref32 = ref.astype(np.float32, copy=False)
        mov32 = mov.astype(np.float32, copy=False)
        # OpenCV returns (dx, dy) in (col, row) coordinates.
        (dx, dy), _ = cv2.phaseCorrelate(ref32, mov32)
        # NOTE: phaseCorrelate returns the shift between images; to align mov to ref,
        # we must apply the *negative* shift to mov.
        return -float(dy), -float(dx)

    # Fallback: integer peak from FFT phase correlation.
    ref_f = np.fft.fft2(ref.astype(np.float64, copy=False))
    mov_f = np.fft.fft2(mov.astype(np.float64, copy=False))
    R = ref_f * np.conj(mov_f)
    denom = np.abs(R)
    R = np.where(denom > 0, R / denom, 0.0)
    cc = np.fft.ifft2(R).real
    peak = np.unravel_index(int(np.argmax(cc)), cc.shape)
    sx, sy = int(peak[0]), int(peak[1])
    nx, ny = cc.shape
    if sx > nx // 2:
        sx -= nx
    if sy > ny // 2:
        sy -= ny
    # As above: apply the negative shift to align mov to ref.
    return -float(sx), -float(sy)


def stabilize_translation_sequence(
    U: np.ndarray,
    *,
    mode: str = "to_first",
    estimate_sigma_px: float = 0.0,
) -> np.ndarray:
    """Align frames by estimating and undoing translations.

    mode:
      - to_first: align each frame to frame 0
      - to_prev:  align each frame to previous aligned frame (cumulative)
    """
    mode = str(mode)
    if mode not in {"to_first", "to_prev"}:
        raise ValueError("stabilize mode must be 'to_first' or 'to_prev'")
    U = U.astype(np.float64, copy=False)
    out = np.empty_like(U)
    out[0] = U[0]
    ref = out[0]
    sigma = float(estimate_sigma_px)
    ref_est = gaussian_smooth_periodic_2d(ref, sigma_px=sigma) if sigma > 0 else ref
    for t in range(1, U.shape[0]):
        mov = U[t]
        mov_est = gaussian_smooth_periodic_2d(mov, sigma_px=sigma) if sigma > 0 else mov
        sx, sy = estimate_shift_phasecorr(ref_est, mov_est)
        # Apply subpixel shift with wrapping.
        if cv2 is not None:
            out[t] = _shift_frame_wrap(mov, shift_x=sx, shift_y=sy)
        else:
            out[t] = np.roll(np.roll(mov, int(round(sx)), axis=0), int(round(sy)), axis=1)
        if mode == "to_prev":
            ref = out[t]
            ref_est = gaussian_smooth_periodic_2d(ref, sigma_px=sigma) if sigma > 0 else ref
    return out


def estimate_interframe_shifts(
    U: np.ndarray,
    *,
    estimate_sigma_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate frame-to-frame shifts (t -> t+1) in pixel units.

    Returns (sx, sy) arrays of length T-1 where (sx[t], sy[t]) is the shift that,
    when applied to frame t+1, best aligns it to frame t.
    """
    U = U.astype(np.float64, copy=False)
    T = U.shape[0]
    if T < 2:
        raise ValueError("Need at least 2 frames to estimate interframe shifts")
    sigma = float(estimate_sigma_px)
    sx = np.zeros(T - 1, dtype=np.float64)
    sy = np.zeros(T - 1, dtype=np.float64)
    for t in range(T - 1):
        ref = U[t]
        mov = U[t + 1]
        if sigma > 0:
            ref = gaussian_smooth_periodic_2d(ref, sigma_px=sigma)
            mov = gaussian_smooth_periodic_2d(mov, sigma_px=sigma)
        sx[t], sy[t] = estimate_shift_phasecorr(ref, mov)
    return sx, sy


def smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Simple centered moving-average smoothing (odd window)."""
    x = np.asarray(x, dtype=np.float64)
    w = int(window)
    if w <= 1:
        return x
    if w % 2 == 0:
        w += 1
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(xp, k, mode="valid")


def build_dictionary(
    U_mid: np.ndarray,
    dx: float,
    dy: float,
    *,
    deriv: str = "finite",
    spectral_cutoff: float = 1.0,
) -> tuple[list[str], dict[str, np.ndarray]]:
    Tmid = U_mid.shape[0]

    lap = np.empty_like(U_mid)
    bih = np.empty_like(U_mid)
    ux = np.empty_like(U_mid)
    uy = np.empty_like(U_mid)
    for k in range(Tmid):
        u = U_mid[k]
        if deriv == "spectral":
            ux_k, uy_k = gradients_spectral(u, dx=dx, dy=dy, cutoff_frac=spectral_cutoff)
            lap_k = laplacian_spectral(u, dx=dx, dy=dy, cutoff_frac=spectral_cutoff)
            bih_k = laplacian_spectral(lap_k, dx=dx, dy=dy, cutoff_frac=spectral_cutoff)
        else:
            ux_k, uy_k = gradients(u, dx=dx, dy=dy)
            lap_k = laplacian(u, dx=dx, dy=dy)
            bih_k = laplacian(lap_k, dx=dx, dy=dy)
        ux[k] = ux_k
        uy[k] = uy_k
        lap[k] = lap_k
        bih[k] = bih_k

    grad_sq = ux**2 + uy**2

    terms: dict[str, np.ndarray] = {
        "1": np.ones_like(U_mid),
        "u": U_mid,
        "u^2": U_mid**2,
        "u_x": ux,
        "u_y": uy,
        "∇²u": lap,
        "∇⁴u": bih,
        "|∇u|²": grad_sq,
        "u·∇²u": U_mid * lap,
    }
    names = list(terms.keys())
    return names, terms


def build_dictionary_true(
    U_frames: np.ndarray,
    dx: float,
    dy: float,
    *,
    deriv: str = "finite",
    spectral_cutoff: float = 1.0,
    include_advection: bool = False,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Dictionary matching the simulated KS RHS exactly."""
    T = U_frames.shape[0]
    lap = np.empty_like(U_frames)
    bih = np.empty_like(U_frames)
    ux = np.empty_like(U_frames)
    uy = np.empty_like(U_frames)

    for k in range(T):
        u = U_frames[k]
        if deriv == "spectral":
            ux_k, uy_k = gradients_spectral(u, dx=dx, dy=dy, cutoff_frac=spectral_cutoff)
            lap_k = laplacian_spectral(u, dx=dx, dy=dy, cutoff_frac=spectral_cutoff)
            bih_k = laplacian_spectral(lap_k, dx=dx, dy=dy, cutoff_frac=spectral_cutoff)
        else:
            ux_k, uy_k = gradients(u, dx=dx, dy=dy)
            lap_k = laplacian(u, dx=dx, dy=dy)
            bih_k = laplacian(lap_k, dx=dx, dy=dy)
        ux[k] = ux_k
        uy[k] = uy_k
        lap[k] = lap_k
        bih[k] = bih_k

    grad_sq = ux**2 + uy**2
    terms: dict[str, np.ndarray] = {
        "∇²u": lap,
        "∇⁴u": bih,
        "|∇u|²": grad_sq,
    }
    if bool(include_advection):
        terms["u_x"] = ux
        terms["u_y"] = uy
    names = list(terms.keys())
    return names, terms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nx", type=int, default=100)
    parser.add_argument("--Ny", type=int, default=100)
    parser.add_argument("--n-seconds", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save every N Euler steps (1 gives the cleanest derivative target).",
    )
    parser.add_argument(
        "--method",
        choices=["pointwise", "blockwise", "weakform"],
        default="pointwise",
        help="Regression target: pointwise u_t (default) or weak-form integrated identity.",
    )
    parser.add_argument(
        "--noise-rel",
        type=float,
        default=0.0,
        help="Add i.i.d. Gaussian noise to frames: sigma = noise_rel * std(U_clean).",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=999,
        help="RNG seed for measurement noise.",
    )

    parser.add_argument(
        "--include-advection",
        action="store_true",
        help="Include u_x and u_y terms in the regression library (useful when frames have residual jitter/transport).",
    )
    parser.add_argument(
        "--enforce-no-advection",
        action="store_true",
        help="Force-remove first-order advection terms (u_x/u_y) even if present; recommended for isotropic KS unless explicitly modeling motion.",
    )

    parser.add_argument(
        "--perturbation",
        choices=[
            "none",
            "N1_shifts",
            "N2_noise",
            "N3_blur",
            "N4_drift",
            "N5_shifts_noise",
            "N6_blur_noise",
            "N7_all",
        ],
        default="none",
        help="Apply measurement corruptions to frames before identification.",
    )
    parser.add_argument("--shift-max", type=float, default=1.5, help="Max abs random shift in pixels (N1/N5/N7).")
    parser.add_argument(
        "--shift-mode",
        choices=["constant", "jitter"],
        default="constant",
        help="Shift model: constant translation (camera drift) or per-frame jitter (strong corruption).",
    )
    parser.add_argument(
        "--stabilize-shifts",
        action="store_true",
        help="Estimate and undo frame-to-frame translations via phase correlation (helps against jitter).",
    )
    parser.add_argument(
        "--correct-shift-ut",
        action="store_true",
        help="Correct u_t for estimated inter-frame translations (treat jitter as advection).",
    )
    parser.add_argument(
        "--ut-shift-smooth",
        type=int,
        default=7,
        help="Odd window for smoothing estimated inter-frame shifts used in --correct-shift-ut.",
    )
    parser.add_argument(
        "--ut-adv-deriv",
        choices=["finite", "spectral"],
        default="spectral",
        help="Derivative backend for advection correction gradients (spectral is more robust under noise).",
    )
    parser.add_argument(
        "--ut-adv-cutoff",
        type=float,
        default=0.5,
        help="Spectral low-pass cutoff for advection gradients (only if --ut-adv-deriv spectral).",
    )
    parser.add_argument(
        "--stabilize-mode",
        choices=["to_first", "to_prev"],
        default="to_first",
        help="Shift stabilization reference: align to first frame or recursively to previous aligned frame.",
    )
    parser.add_argument(
        "--stabilize-est-sigma",
        type=float,
        default=0.0,
        help="Optional Gaussian sigma (px) for smoothing frames *only for shift estimation* (helps under noise).",
    )
    parser.add_argument("--blur-sigma", type=float, default=1.5, help="Gaussian blur sigma in pixels (N3/N6/N7).")
    parser.add_argument("--drift", type=float, default=0.02, help="Intensity decay per frame (N4/N7).")
    parser.add_argument("--n-sample", type=int, default=50_000)
    parser.add_argument("--rollout-steps", type=int, default=50)
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="If set, search over alpha/threshold grid (slower).",
    )
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--threshold", type=float, default=1e-10)
    parser.add_argument(
        "--dictionary",
        choices=["true", "rich"],
        default="true",
        help="Use 'true' (KS terms only) or 'rich' (true + decoys).",
    )
    parser.add_argument(
        "--derivatives",
        choices=["finite", "spectral"],
        default="finite",
        help="How to compute spatial derivatives for dictionary terms.",
    )
    parser.add_argument(
        "--spectral-cutoff",
        type=float,
        default=1.0,
        help="Radial low-pass cutoff fraction in (0,1] for spectral derivatives.",
    )
    parser.add_argument(
        "--weak-max-k",
        type=int,
        default=3,
        help="Max Fourier mode index (m,n<=K) for weak-form test functions.",
    )
    parser.add_argument(
        "--weak-basis",
        choices=["gaussian", "fourier"],
        default="gaussian",
        help="Weak-form test-function family: localized gaussians (default) or global fourier modes.",
    )
    parser.add_argument(
        "--weak-n-phi",
        type=int,
        default=64,
        help="Number of Gaussian test functions (if --weak-basis gaussian).",
    )
    parser.add_argument(
        "--weak-sigma-px",
        type=float,
        default=6.0,
        help="Gaussian sigma in pixels for weak-form test functions.",
    )
    parser.add_argument(
        "--weak-grad-cutoff",
        type=float,
        default=0.65,
        help="Low-pass cutoff for spectral gradients used in weak-form nonlinear term.",
    )
    parser.add_argument(
        "--weak-motion-correct",
        action="store_true",
        help="Weak-form only: correct the target for estimated global translation velocity (u_t + v·∇u = RHS).",
    )
    parser.add_argument(
        "--weak-motion-est-sigma",
        type=float,
        default=0.0,
        help="Weak-form only: optional Gaussian sigma (px) for smoothing frames before estimating shifts.",
    )
    parser.add_argument(
        "--weak-motion-smooth",
        type=int,
        default=7,
        help="Weak-form only: odd window for smoothing estimated inter-frame shifts (velocity).",
    )
    parser.add_argument(
        "--weak-motion-clip-px",
        type=float,
        default=-1.0,
        help="Weak-form only: clip estimated inter-frame shifts to +/- this many pixels (use -1 to auto).",
    )

    parser.add_argument("--block-t", type=int, default=3, help="Blockwise only: block size in time (frames).")
    parser.add_argument("--block-x", type=int, default=8, help="Blockwise only: block size along x (axis=0).")
    parser.add_argument("--block-y", type=int, default=8, help="Blockwise only: block size along y (axis=1).")

    # Robust regression options
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use full robust pipeline: trimmed + Huber + bootstrap + sign constraints.",
    )
    parser.add_argument(
        "--regression",
        choices=["standard", "huber", "trimmed", "sign_constrained", "ensemble"],
        default="standard",
        help="Regression method: standard STRidge (default), huber (outlier-robust), trimmed (remove outliers), sign_constrained (physics-informed), ensemble (bootstrap).",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.35,
        help="Huber loss threshold (delta). Smaller = more robust, larger = closer to L2.",
    )
    parser.add_argument(
        "--trim-frac",
        type=float,
        default=0.05,
        help="Fraction of samples to trim (remove) based on residual magnitude.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=30,
        help="Number of bootstrap samples for ensemble/robust regression.",
    )
    parser.add_argument(
        "--sign-constraints",
        type=str,
        default="",
        help="Comma-separated signs for KS coefficients: e.g., '-1,-1,-1' means all must be negative. Empty = no constraint.",
    )

    parser.add_argument(
        "--denoise-time-window",
        type=int,
        default=1,
        help="Odd moving-average window (in frames) applied before computing u_t (reduces noise).",
    )
    parser.add_argument(
        "--denoise-space-sigma",
        type=float,
        default=0.0,
        help="Periodic Gaussian smoothing sigma (pixels) applied before computing spatial-derivative features.",
    )
    parser.add_argument(
        "--denoise-space-on",
        choices=["features", "all"],
        default="features",
        help="Apply spatial smoothing to 'features' only (default) or to 'all' (also affects u_t).",
    )
    args = parser.parse_args()

    cfg = SimConfig()
    cfg = SimConfig(
        Nx=int(args.Nx),
        Ny=int(args.Ny),
        dt=float(args.dt),
        n_seconds=float(args.n_seconds),
        save_every=int(args.save_every),
    )
    print(
        "Config:",
        {
            "Nx": cfg.Nx,
            "Ny": cfg.Ny,
            "dt": cfg.dt,
            "n_seconds": cfg.n_seconds,
            "save_every": cfg.save_every,
            "dictionary": args.dictionary,
            "noise_rel": float(args.noise_rel),
            "noise_seed": int(args.noise_seed),
            "perturbation": str(args.perturbation),
            "shift_max": float(args.shift_max),
            "shift_mode": str(args.shift_mode),
            "stabilize_shifts": bool(args.stabilize_shifts),
            "stabilize_mode": str(args.stabilize_mode),
            "blur_sigma": float(args.blur_sigma),
            "drift": float(args.drift),
            "n_sample": int(args.n_sample),
            "grid_search": bool(args.grid_search),
            "alpha": float(args.alpha),
            "threshold": float(args.threshold),
            "denoise_time_window": int(args.denoise_time_window),
            "denoise_space_sigma": float(args.denoise_space_sigma),
            "denoise_space_on": str(args.denoise_space_on),
            "derivatives": str(args.derivatives),
            "spectral_cutoff": float(args.spectral_cutoff),
            "method": str(args.method),
            "weak_max_k": int(args.weak_max_k),
            "weak_basis": str(args.weak_basis),
            "weak_n_phi": int(args.weak_n_phi),
            "weak_sigma_px": float(args.weak_sigma_px),
            "weak_grad_cutoff": float(args.weak_grad_cutoff),
            "weak_motion_correct": bool(args.weak_motion_correct),
            "weak_motion_est_sigma": float(args.weak_motion_est_sigma),
            "weak_motion_smooth": int(args.weak_motion_smooth),
            "weak_motion_clip_px": float(args.weak_motion_clip_px),
            "block_t": int(args.block_t),
            "block_x": int(args.block_x),
            "block_y": int(args.block_y),
            "enforce_no_advection": bool(args.enforce_no_advection),
        },
    )
    print("Simulating 2D KS...")
    U_clean, dx, dy, DT = simulate(cfg)

    rng_obs = np.random.default_rng(int(args.noise_seed))
    # By default, N2/N5/N6/N7 use noise_rel=0.03 if not specified.
    noise_rel = float(args.noise_rel)
    if args.perturbation in {"N2_noise", "N5_shifts_noise", "N6_blur_noise", "N7_all"} and noise_rel == 0.0:
        noise_rel = 0.03

    U = apply_perturbation_suite(
        U_clean,
        perturbation=str(args.perturbation),
        rng=rng_obs,
        noise_rel=noise_rel,
        shift_max_px=float(args.shift_max),
        shift_mode=str(args.shift_mode),
        blur_sigma=float(args.blur_sigma),
        drift_per_frame=float(args.drift),
    )

    if bool(args.stabilize_shifts):
        U = stabilize_translation_sequence(
            U,
            mode=str(args.stabilize_mode),
            estimate_sigma_px=float(args.stabilize_est_sigma),
        )
        print(f"Applied shift stabilization: mode={str(args.stabilize_mode)}")
    if str(args.perturbation) != "none":
        print(
            "Applied perturbation:",
            {
                "perturbation": str(args.perturbation),
                "noise_rel": noise_rel,
                "shift_max": float(args.shift_max),
                "shift_mode": str(args.shift_mode),
                "blur_sigma": float(args.blur_sigma),
                "drift": float(args.drift),
            },
        )
    print(f"U={U.shape}, dx={dx:.4g}, dy={dy:.4g}, DT={DT:.4g}")

    # Optional denoising to stabilize derivative-based terms (especially |∇u|²) under noise.
    U_for_ut = U
    if int(args.denoise_time_window) > 1:
        U_for_ut = time_smooth_moving_average(U_for_ut, window=int(args.denoise_time_window))
        print(f"Applied time smoothing: window={int(args.denoise_time_window)}")

    U_for_features = U_for_ut
    if float(args.denoise_space_sigma) > 0.0:
        if str(args.denoise_space_on) == "all":
            U_for_ut = np.stack(
                [gaussian_smooth_periodic_2d(U_for_ut[t], sigma_px=float(args.denoise_space_sigma)) for t in range(U_for_ut.shape[0])],
                axis=0,
            )
            U_for_features = U_for_ut
        else:
            U_for_features = np.stack(
                [gaussian_smooth_periodic_2d(U_for_features[t], sigma_px=float(args.denoise_space_sigma)) for t in range(U_for_features.shape[0])],
                axis=0,
            )
        print(
            f"Applied space smoothing: sigma_px={float(args.denoise_space_sigma):.3g} on {str(args.denoise_space_on)}"
        )

    rng = np.random.default_rng(0)

    if str(args.method) == "weakform":
        if args.dictionary != "true":
            raise ValueError("weakform currently supports --dictionary true only")
        names = ["∇²u", "∇⁴u", "|∇u|²"]
        print(f"Dictionary (weakform) terms ({len(names)}): {names}")

        clip_px = float(args.weak_motion_clip_px)
        if clip_px <= 0:
            # Auto: if shifts are part of the perturbation, use that bound.
            clip_px = float(args.shift_max) if str(args.perturbation) in {"N1_shifts", "N5_shifts_noise", "N7_all"} else 0.0
        motion_clip_px = clip_px if clip_px > 0 else None

        X_all, y_all = build_weakform_dataset(
            U_for_ut,
            dx=dx,
            dy=dy,
            dt_frame=DT,
            lx=float(cfg.Lx),
            ly=float(cfg.Ly),
            max_k=int(args.weak_max_k),
            basis=str(args.weak_basis),
            n_phi=int(args.weak_n_phi),
            sigma_px=float(args.weak_sigma_px),
            grad_cutoff=float(args.weak_grad_cutoff),
            motion_correct=bool(args.weak_motion_correct),
            motion_est_sigma_px=float(args.weak_motion_est_sigma),
            motion_smooth_window=int(args.weak_motion_smooth),
            motion_clip_px=motion_clip_px,
        )
        # Optional subsample of (time,basis) pairs
        n_total = y_all.size
        n_sample = int(min(args.n_sample, n_total))
        idx = rng.choice(n_total, size=n_sample, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]
        print(f"Sampled dataset (weakform): X={X_all.shape}, y={y_all.shape}")
    elif str(args.method) == "blockwise":
        # Forward-difference temporal derivative consistent with Euler update.
        U_frames = U_for_features[:-1]
        Ut = (U_for_ut[1:] - U_for_ut[:-1]) / DT

        include_adv = bool(args.include_advection) and not bool(args.enforce_no_advection)
        if bool(args.enforce_no_advection) and bool(args.include_advection):
            print("NOTE: --enforce-no-advection overrides --include-advection")

        if args.dictionary == "true":
            names, terms = build_dictionary_true(
                U_frames,
                dx=dx,
                dy=dy,
                deriv=str(args.derivatives),
                spectral_cutoff=float(args.spectral_cutoff),
                include_advection=include_adv,
            )
        else:
            names, terms = build_dictionary(
                U_frames,
                dx=dx,
                dy=dy,
                deriv=str(args.derivatives),
                spectral_cutoff=float(args.spectral_cutoff),
            )

            # If using a rich dictionary, drop obvious advection-like terms when requested.
            if bool(args.enforce_no_advection):
                drop = {"u_x", "u_y"}
                names = [n for n in names if n not in drop]
                terms = {k: v for k, v in terms.items() if k in set(names)}

        print(f"Dictionary ({args.dictionary}) terms ({len(names)}): {names}")
        X_all, y_all = build_blockwise_dataset(
            Ut,
            terms,
            names,
            block_t=int(args.block_t),
            block_x=int(args.block_x),
            block_y=int(args.block_y),
        )
        print(f"Blockwise dataset: X={X_all.shape}, y={y_all.shape}")
    else:
        # Use a forward-difference temporal derivative consistent with Euler update:
        #   (u_{k+1} - u_k)/DT ≈ RHS(u_k)
        U_frames = U_for_features[:-1]
        Ut = (U_for_ut[1:] - U_for_ut[:-1]) / DT

        if bool(args.correct_shift_ut):
            # Treat inter-frame translations as an apparent advection term in u_t and remove it:
            # u_t,obs \approx u_t,true - v·∇u  with  v = (Δx/Δt, Δy/Δt) in physical units.
            # We estimate the per-step shift (in pixels) and convert to physical velocity.
            sx_px, sy_px = estimate_interframe_shifts(
                U_for_ut,
                estimate_sigma_px=float(args.stabilize_est_sigma),
            )
            sx_px = smooth_1d(sx_px, window=int(args.ut_shift_smooth))
            sy_px = smooth_1d(sy_px, window=int(args.ut_shift_smooth))
            U_adv = U_for_ut[:-1]
            ux_adv = np.empty_like(U_adv)
            uy_adv = np.empty_like(U_adv)
            for t in range(U_adv.shape[0]):
                if str(args.ut_adv_deriv) == "spectral":
                    ux_adv[t], uy_adv[t] = gradients_spectral(
                        U_adv[t],
                        dx=dx,
                        dy=dy,
                        cutoff_frac=float(args.ut_adv_cutoff),
                    )
                else:
                    ux_adv[t], uy_adv[t] = gradients(U_adv[t], dx=dx, dy=dy)
            vx = (-sx_px * dx) / DT
            vy = (-sy_px * dy) / DT
            Ut = Ut + vx[:, None, None] * ux_adv + vy[:, None, None] * uy_adv
            print(
                "Applied u_t shift correction (advection):",
                {
                    "estimate_sigma_px": float(args.stabilize_est_sigma),
                    "shift_smooth": int(args.ut_shift_smooth),
                    "adv_deriv": str(args.ut_adv_deriv),
                    "adv_cutoff": float(args.ut_adv_cutoff),
                    "vx_rms": float(np.sqrt(np.mean(vx**2))),
                    "vy_rms": float(np.sqrt(np.mean(vy**2))),
                },
            )

        include_adv = bool(args.include_advection) and not bool(args.enforce_no_advection)
        if bool(args.enforce_no_advection) and bool(args.include_advection):
            print("NOTE: --enforce-no-advection overrides --include-advection")

        if args.dictionary == "true":
            names, terms = build_dictionary_true(
                U_frames,
                dx=dx,
                dy=dy,
                deriv=str(args.derivatives),
                spectral_cutoff=float(args.spectral_cutoff),
                include_advection=include_adv,
            )
        else:
            # Rich dictionary computed on the same frames as the forward-diff target
            names, terms = build_dictionary(
                U_frames,
                dx=dx,
                dy=dy,
                deriv=str(args.derivatives),
                spectral_cutoff=float(args.spectral_cutoff),
            )

            if bool(args.enforce_no_advection):
                drop = {"u_x", "u_y"}
                names = [n for n in names if n not in drop]
                terms = {k: v for k, v in terms.items() if k in set(names)}

        print(f"Dictionary ({args.dictionary}) terms ({len(names)}): {names}")

        # Sample pointwise dataset
        n_total = Ut.size
        n_sample = int(min(args.n_sample, n_total))
        flat_idx = rng.choice(n_total, size=n_sample, replace=False)

        y_all = Ut.reshape(-1)[flat_idx]
        X_all = np.column_stack([terms[n].reshape(-1)[flat_idx] for n in names])

        valid = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all)
        X_all = X_all[valid]
        y_all = y_all[valid]
        print(f"Sampled dataset: X={X_all.shape}, y={y_all.shape}")

    # Train/test split
    perm = rng.permutation(len(y_all))
    split = int(0.7 * len(y_all))
    tr, te = perm[:split], perm[split:]
    X_tr, y_tr = X_all[tr], y_all[tr]
    X_te, y_te = X_all[te], y_all[te]

    # Feature scaling improves conditioning, especially when mixing derivative terms
    # (which can have very different magnitudes under noise/blur).
    # We scale by per-column RMS on the training split and unscale coefficients after.
    eps = 1e-12
    scale = np.sqrt(np.mean(X_tr**2, axis=0)) + eps
    # Do not scale the constant column if present.
    for j, name in enumerate(names):
        if name == "1":
            scale[j] = 1.0
    X_tr_s = X_tr / scale
    X_te_s = X_te / scale

    # Parse sign constraints if provided
    sign_constraints = None
    if args.sign_constraints:
        sign_constraints = [int(s.strip()) for s in args.sign_constraints.split(",")]
        if len(sign_constraints) != X_tr.shape[1]:
            print(f"Warning: sign_constraints has {len(sign_constraints)} entries but {X_tr.shape[1]} features. Ignoring.")
            sign_constraints = None

    # Helper function to select regression method
    def do_regression(X_s, y, alpha, threshold, max_iter=25):
        """Dispatch to selected regression method."""
        if args.robust:
            # Full robust pipeline
            c_s, info = robust_stridge(
                X_s, y,
                alpha=alpha,
                threshold=threshold,
                max_iter=max_iter,
                use_huber=True,
                huber_delta=float(args.huber_delta),
                trim_frac=float(args.trim_frac),
                n_bootstrap=int(args.n_bootstrap),
                signs=sign_constraints,
            )
            return c_s, info
        elif args.regression == "huber":
            return stridge_huber(
                X_s, y,
                alpha=alpha,
                threshold=threshold,
                max_iter=max_iter,
                huber_delta=float(args.huber_delta),
            ), None
        elif args.regression == "trimmed":
            return trimmed_stridge(
                X_s, y,
                alpha=alpha,
                threshold=threshold,
                max_iter=max_iter,
                trim_frac=float(args.trim_frac),
            ), None
        elif args.regression == "sign_constrained":
            return stridge_sign_constrained(
                X_s, y,
                alpha=alpha,
                threshold=threshold,
                max_iter=max_iter,
                signs=sign_constraints,
            ), None
        elif args.regression == "ensemble":
            mean_c, std_c = ensemble_stridge(
                X_s, y,
                alpha=alpha,
                threshold=threshold,
                max_iter=max_iter,
                n_bootstrap=int(args.n_bootstrap),
                use_huber=True,
                huber_delta=float(args.huber_delta),
            )
            return mean_c, {"std": std_c}
        else:
            return stridge(X_s, y, alpha=alpha, threshold=threshold, max_iter=max_iter), None

    if args.grid_search:
        alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        thresholds = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

        best: dict | None = None
        for a in alphas:
            for thr in thresholds:
                c_s, _ = do_regression(X_tr_s, y_tr, alpha=a, threshold=thr, max_iter=25)
                c = c_s / scale
                y_pred = X_te @ c
                r2 = r2_score(y_te, y_pred)
                err = rmse(y_te, y_pred)
                n_active = int(np.sum(np.abs(c) > 0))
                key = (r2, -n_active, -err)
                if best is None or key > best["key"]:
                    best = {
                        "key": key,
                        "alpha": a,
                        "threshold": thr,
                        "coeffs": c,
                        "r2_test": float(r2),
                        "rmse_test": float(err),
                        "n_active": n_active,
                    }

        assert best is not None
        c_best = best["coeffs"]
        robust_info = None
        print("\nBest STRidge hyperparams:")
        print({k: v for k, v in best.items() if k != "coeffs"})
    else:
        c_best_s, robust_info = do_regression(
            X_tr_s,
            y_tr,
            alpha=float(args.alpha),
            threshold=float(args.threshold),
            max_iter=25,
        )
        c_best = c_best_s / scale
        y_pred = X_te @ c_best
        r2 = r2_score(y_te, y_pred)
        err = rmse(y_te, y_pred)
        n_active = int(np.sum(np.abs(c_best) > 0))
        reg_method = "robust" if args.robust else args.regression
        print(f"\n{reg_method.upper()} STRidge hyperparams:")
        print(
            {
                "alpha": float(args.alpha),
                "threshold": float(args.threshold),
                "r2_test": float(r2),
                "rmse_test": float(err),
                "n_active": n_active,
                "regression": reg_method,
            }
        )
        if robust_info is not None and "std" in robust_info:
            print("\nCoefficient uncertainty (std):")
            for name, std_val in zip(names, robust_info["std"] / scale):
                print(f"  {name:8s}: ±{std_val:.6f}")

    print("\nDiscovered PDE (|c| > 1e-8):")
    for name, c in sorted(zip(names, c_best), key=lambda p: -abs(p[1])):
        if abs(c) > 1e-8:
            print(f"  {name:8s}: {c:+.6f}")

    # Ground truth
    gt = {"∇²u": -1.0, "∇⁴u": -1.0, "|∇u|²": -0.5}
    print("\nGround-truth comparison (relative error):")
    for k, v in gt.items():
        est = float(c_best[names.index(k)])
        rel = abs(est - v) / (abs(v) + 1e-12) * 100.0
        print(f"  {k:8s}: gt={v:+.6f}, est={est:+.6f}, rel_err={rel:.3f}%")

    # Final metrics
    y_pred_tr = X_tr @ c_best
    y_pred_te = X_te @ c_best
    print("\nFit quality:")
    print(
        f"  Train R2={r2_score(y_tr, y_pred_tr):.6f}, RMSE={rmse(y_tr, y_pred_tr):.6e}"
    )
    print(
        f"  Test  R2={r2_score(y_te, y_pred_te):.6f}, RMSE={rmse(y_te, y_pred_te):.6e}"
    )

    # Rollout check
    def rhs_from_coeffs(u2d: np.ndarray) -> np.ndarray:
        ux2, uy2 = gradients(u2d, dx=dx, dy=dy)
        lap2 = laplacian(u2d, dx=dx, dy=dy)
        bih2 = laplacian(lap2, dx=dx, dy=dy)
        grad_sq2 = ux2**2 + uy2**2
        vals: dict[str, np.ndarray | float] = {
            "1": 1.0,
            "u": u2d,
            "u^2": u2d**2,
            "u_x": ux2,
            "u_y": uy2,
            "∇²u": lap2,
            "∇⁴u": bih2,
            "|∇u|²": grad_sq2,
            "u·∇²u": u2d * lap2,
        }
        out = np.zeros_like(u2d, dtype=np.float64)
        for name, c in zip(names, c_best):
            if abs(c) < 1e-12:
                continue
            v = vals[name]
            out += c * (v if isinstance(v, np.ndarray) else float(v))
        return out

    n_roll = int(min(args.rollout_steps, U.shape[0] - 1))
    u_hat = U[0].copy()
    errs = []
    for k in range(n_roll):
        u_hat = u_hat + DT * rhs_from_coeffs(u_hat)
        errs.append(rmse(U[k + 1].ravel(), u_hat.ravel()))

    print(
        f"\nRollout RMSE over {n_roll} steps: first={errs[0]:.3e}, last={errs[-1]:.3e}, mean={float(np.mean(errs)):.3e}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
