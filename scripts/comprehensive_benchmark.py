"""
=============================================================================
COMPREHENSIVE PDE DISCOVERY BENCHMARK
=============================================================================

Tests ALL methods across MULTIPLE noise types to find the best general approach.

NOISE TYPES (mimicking real experimental data):
  N1: Spatial shifts only (unregistered images)
  N2: Gaussian noise only (sensor noise)
  N3: Blur only (optical blur/defocus)
  N4: Intensity drift (photobleaching)
  N5: Shifts + Noise (common in microscopy)
  N6: Blur + Noise (common in imaging)
  N7: ALL combined (worst case)

METHODS (Ideas 1-17, 19):
  M1:  Baseline SINDy (no preprocessing)
  M2:  Gaussian denoising
  M3:  Median filtering
  M4:  Total Variation denoising
  M5:  Bilateral filtering
  M6:  Temporal smoothing
  M7:  Robust regression (Huber)
  M8:  Robust regression (RANSAC-like)
  M9:  Fourier derivatives
  M10: Weak form SINDy
  M11: Ensemble averaging
  M12: Standard DMD
  M13: Optimized DMD (rank tuning)
  M14: DMD + Fourier
  M15: Sparse DMD
  M16: Multi-scale DMD (best on shifts)
  M17: Physics-constrained DMD
  M18: Intensity detrending

Ground Truth PDE: ∂u/∂t = -∇²u - ∇⁴u - 0.5|∇u|²
True coefficients: [-1, -1, -0.5]

Author: MLDM Research 2025
=============================================================================
"""

import numpy as np
import json
import time
from scipy.ndimage import shift as ndshift, gaussian_filter, median_filter, uniform_filter
from scipy.linalg import svd, lstsq
from sklearn.linear_model import Ridge, HuberRegressor, RANSACRegressor, LinearRegression
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
TRUE_COEF = np.array([-1.0, -1.0, -0.5])  # [∇²u, ∇⁴u, |∇u|²]

# Simulation parameters
Lx, Ly = 100, 100
Nx, Ny = 64, 64  # Smaller for faster benchmarking
dx = Lx / Nx
dy = Ly / Ny
dt = 0.0001
n_seconds = 1.0
save_every = 50
total_steps = int(n_seconds / dt)
n_frames = total_steps // save_every
dt_frame = dt * save_every

print("="*70)
print("COMPREHENSIVE PDE DISCOVERY BENCHMARK")
print("="*70)
print(f"\nGrid: {Nx}x{Ny}, Frames: {n_frames}, dt_frame: {dt_frame}")
print(f"True PDE: ∂u/∂t = {TRUE_COEF[0]}·∇²u + {TRUE_COEF[1]}·∇⁴u + {TRUE_COEF[2]}·|∇u|²")

# =============================================================================
# GENERATE CLEAN DATA
# =============================================================================
print("\n[1] Generating clean KS data...")

def get_laplacian(f, dx):
    return (np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) +
            np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) - 4*f) / (dx**2)

def get_gradients(f, dx, dy):
    gx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dx)
    gy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dy)
    return gx, gy

def compute_rhs(u, dx, dy):
    lap = get_laplacian(u, dx)
    lap_lap = get_laplacian(lap, dx)
    gx, gy = get_gradients(u, dx, dy)
    return -lap - lap_lap - 0.5*(gx**2 + gy**2)

np.random.seed(42)
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)
u0 = 0.1 * (np.sin(2*np.pi*X/Lx) * np.cos(2*np.pi*Y/Ly) + 
            0.5 * np.sin(4*np.pi*X/Lx) * np.sin(4*np.pi*Y/Ly))

u_clean = np.zeros((n_frames, Nx, Ny))
current_u = u0.copy()
frame_idx = 0

for step in range(total_steps):
    rhs = compute_rhs(current_u, dx, dy)
    current_u = current_u + dt * rhs
    current_u = np.clip(current_u, -10, 10)
    if step % save_every == 0 and frame_idx < n_frames:
        u_clean[frame_idx] = current_u.copy()
        frame_idx += 1

print(f"   ✓ Clean data: {u_clean.shape}")

# =============================================================================
# NOISE FUNCTIONS
# =============================================================================
def add_shifts(data, max_shift=1.5):
    """Add random spatial shifts (unregistered images)"""
    result = np.zeros_like(data)
    np.random.seed(123)
    for i in range(len(data)):
        shift = np.random.uniform(-max_shift, max_shift, 2) if i > 0 else [0, 0]
        result[i] = ndshift(data[i], shift, mode='wrap')
    return result

def add_noise(data, std=0.02):
    """Add Gaussian noise"""
    np.random.seed(456)
    return data + std * np.std(data) * np.random.randn(*data.shape)

def add_blur(data, sigma=1.5):
    """Add Gaussian blur"""
    result = np.zeros_like(data)
    for i in range(len(data)):
        result[i] = gaussian_filter(data[i], sigma=sigma)
    return result

def add_intensity_drift(data, decay_rate=0.02):
    """Add intensity drift (photobleaching)"""
    t = np.arange(len(data))
    decay = np.exp(-decay_rate * t)
    return data * decay[:, None, None]

# Create all noise combinations
print("\n[2] Creating noisy datasets...")
NOISE_CONFIGS = {
    'N1_shifts': lambda d: add_shifts(d, max_shift=1.5),
    'N2_noise': lambda d: add_noise(d, std=0.03),
    'N3_blur': lambda d: add_blur(d, sigma=1.5),
    'N4_drift': lambda d: add_intensity_drift(d, decay_rate=0.02),
    'N5_shifts_noise': lambda d: add_noise(add_shifts(d, 1.5), 0.02),
    'N6_blur_noise': lambda d: add_noise(add_blur(d, 1.0), 0.02),
    'N7_all': lambda d: add_noise(add_blur(add_intensity_drift(add_shifts(d, 1.0), 0.015), 1.0), 0.02),
}

noisy_datasets = {}
for name, noise_fn in NOISE_CONFIGS.items():
    noisy_datasets[name] = noise_fn(u_clean.copy())
    print(f"   ✓ {name}")

# =============================================================================
# PREPROCESSING METHODS
# =============================================================================

def preprocess_none(data):
    """No preprocessing"""
    return data

def preprocess_gaussian(data, sigma=1.0):
    """Gaussian smoothing"""
    result = np.zeros_like(data)
    for i in range(len(data)):
        result[i] = gaussian_filter(data[i], sigma=sigma)
    return result

def preprocess_median(data, size=3):
    """Median filtering"""
    result = np.zeros_like(data)
    for i in range(len(data)):
        result[i] = median_filter(data[i], size=size)
    return result

def preprocess_tv(data, weight=0.1, n_iter=50):
    """Total Variation denoising (simplified)"""
    result = data.copy()
    for i in range(len(data)):
        u = data[i].copy()
        for _ in range(n_iter):
            gx = np.roll(u, -1, 0) - u
            gy = np.roll(u, -1, 1) - u
            norm = np.sqrt(gx**2 + gy**2 + 1e-8)
            div = (gx - np.roll(gx, 1, 0))/norm + (gy - np.roll(gy, 1, 1))/norm
            u = data[i] + weight * div
        result[i] = u
    return result

def preprocess_bilateral(data, sigma_s=2, sigma_r=0.1):
    """Bilateral-like filtering (approximation)"""
    result = np.zeros_like(data)
    for i in range(len(data)):
        smoothed = gaussian_filter(data[i], sigma=sigma_s)
        diff = data[i] - smoothed
        weights = np.exp(-diff**2 / (2*sigma_r**2))
        result[i] = weights * data[i] + (1-weights) * smoothed
    return result

def preprocess_temporal(data, window=3):
    """Temporal smoothing"""
    result = np.zeros_like(data)
    half = window // 2
    for i in range(len(data)):
        i_start = max(0, i - half)
        i_end = min(len(data), i + half + 1)
        result[i] = data[i_start:i_end].mean(axis=0)
    return result

def preprocess_detrend(data):
    """Remove intensity drift - properly normalize each frame"""
    result = np.zeros_like(data)
    for i in range(len(data)):
        frame = data[i]
        result[i] = (frame - frame.mean()) / (frame.std() + 1e-10)
    return result

def preprocess_detrend_global(data):
    """Remove global exponential intensity drift"""
    means = data.mean(axis=(1, 2))
    t = np.arange(len(means))
    # Fit linear in log space
    log_means = np.log(np.abs(means) + 1e-10)
    poly = np.polyfit(t, log_means, 1)
    trend = np.exp(poly[0] * t + poly[1])
    # Divide out trend
    result = data / (trend[:, None, None] + 1e-10)
    # Then normalize
    return (result - result.mean()) / (result.std() + 1e-10)

# =============================================================================
# DMD METHODS
# =============================================================================

def standard_dmd(data, rank=None):
    """Standard DMD reconstruction"""
    n_frames = data.shape[0]
    X = data.reshape(n_frames, -1).T
    X1, X2 = X[:, :-1], X[:, 1:]
    
    U, s, Vh = svd(X1, full_matrices=False)
    r = min(rank or len(s), len(s), n_frames-2)
    U, s, Vh = U[:, :r], s[:r], Vh[:r, :]
    
    s_reg = np.where(s > 1e-10, s, 1e-10)
    Atilde = U.T @ X2 @ Vh.T @ np.diag(1/s_reg)
    eigenvalues, W = np.linalg.eig(Atilde)
    Phi = X2 @ Vh.T @ np.diag(1/s_reg) @ W
    
    b = lstsq(Phi, X[:, 0], cond=1e-10)[0]
    
    X_recon = np.zeros((X.shape[0], n_frames), dtype=complex)
    for t in range(n_frames):
        X_recon[:, t] = Phi @ (b * (eigenvalues ** t))
    
    return np.real(X_recon.T.reshape(data.shape))

def multiscale_dmd(data, low_rank=5, high_rank=15, cutoff=0.1):
    """Multi-scale DMD: separate low/high frequency"""
    n_frames, Ny, Nx = data.shape
    
    # Frequency separation
    u_low = np.zeros_like(data)
    u_high = np.zeros_like(data)
    
    kx = np.fft.fftfreq(Ny)
    ky = np.fft.fftfreq(Nx)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    low_mask = K < cutoff
    
    for t in range(n_frames):
        u_hat = np.fft.fft2(data[t])
        u_low[t] = np.real(np.fft.ifft2(u_hat * low_mask))
        u_high[t] = np.real(np.fft.ifft2(u_hat * ~low_mask))
    
    # DMD on each scale
    u_low_dmd = standard_dmd(u_low, rank=low_rank)
    u_high_dmd = standard_dmd(u_high, rank=high_rank)
    
    return u_low_dmd + u_high_dmd

def sparse_dmd(data, rank=20, sparsity=0.1):
    """Sparse DMD with mode selection"""
    n_frames = data.shape[0]
    X = data.reshape(n_frames, -1).T
    X1, X2 = X[:, :-1], X[:, 1:]
    
    U, s, Vh = svd(X1, full_matrices=False)
    r = min(rank, len(s))
    U, s, Vh = U[:, :r], s[:r], Vh[:r, :]
    
    Atilde = U.T @ X2 @ Vh.T @ np.diag(1/s)
    eigenvalues, W = np.linalg.eig(Atilde)
    Phi = X2 @ Vh.T @ np.diag(1/s) @ W
    
    # Sparse mode selection
    mode_energies = np.abs(Phi).sum(axis=0)
    threshold = np.percentile(mode_energies, 100*(1-sparsity))
    keep = mode_energies >= threshold
    
    Phi_sparse = Phi[:, keep]
    eigenvalues_sparse = eigenvalues[keep]
    
    b = lstsq(Phi_sparse, X[:, 0], cond=1e-10)[0]
    
    X_recon = np.zeros((X.shape[0], n_frames), dtype=complex)
    for t in range(n_frames):
        X_recon[:, t] = Phi_sparse @ (b * (eigenvalues_sparse ** t))
    
    return np.real(X_recon.T.reshape(data.shape))

# =============================================================================
# PDE DISCOVERY (SINDy)
# =============================================================================

def fourier_derivatives(u, dx, dy):
    """Compute derivatives using FFT"""
    Ny, Nx = u.shape
    kx = np.fft.fftfreq(Ny, dx) * 2 * np.pi
    ky = np.fft.fftfreq(Nx, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k2 = KX**2 + KY**2
    
    u_hat = np.fft.fft2(u)
    ux = np.real(np.fft.ifft2(1j * KX * u_hat))
    uy = np.real(np.fft.ifft2(1j * KY * u_hat))
    lap = np.real(np.fft.ifft2(-k2 * u_hat))
    biharm = np.real(np.fft.ifft2(k2**2 * u_hat))
    
    return ux, uy, lap, biharm

def fd_derivatives(u, dx, dy):
    """Finite difference derivatives"""
    ux = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2*dx)
    uy = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2*dy)
    lap = (np.roll(u, -1, 0) + np.roll(u, 1, 0) +
           np.roll(u, -1, 1) + np.roll(u, 1, 1) - 4*u) / dx**2
    biharm = get_laplacian(lap, dx)
    return ux, uy, lap, biharm

def sindy_discover(data, dx, dy, dt_frame, use_fourier=True, 
                   regressor='ridge', margin=3):
    """Discover PDE coefficients"""
    lib_list, tgt_list = [], []
    inner = (slice(margin, -margin), slice(margin, -margin))
    
    for i in range(len(data) - 1):
        u = data[i]
        dudt = (data[i + 1] - u) / dt_frame
        
        if use_fourier:
            ux, uy, lap, biharm = fourier_derivatives(u, dx, dy)
        else:
            ux, uy, lap, biharm = fd_derivatives(u, dx, dy)
        
        grad_sq = ux**2 + uy**2
        
        # Library: [∇²u, ∇⁴u, |∇u|²]
        lib_list.append(np.column_stack([
            lap[inner].flatten(),
            biharm[inner].flatten(),
            grad_sq[inner].flatten()
        ]))
        tgt_list.append(dudt[inner].flatten())
    
    lib = np.vstack(lib_list)
    tgt = np.concatenate(tgt_list)
    
    # Normalize
    scales = np.std(lib, axis=0)
    scales[scales < 1e-10] = 1.0
    lib_norm = lib / scales
    
    # Subsample
    n_samples = min(20000, len(tgt))
    idx = np.random.choice(len(tgt), n_samples, replace=False)
    
    # Fit
    if regressor == 'ridge':
        model = Ridge(alpha=1e-4)
        model.fit(lib_norm[idx], tgt[idx])
        coef = model.coef_ / scales
    elif regressor == 'huber':
        model = HuberRegressor(epsilon=1.35, max_iter=200)
        model.fit(lib_norm[idx], tgt[idx])
        coef = model.coef_ / scales
    elif regressor == 'ransac':
        model = RANSACRegressor(LinearRegression(), max_trials=100)
        model.fit(lib_norm[idx], tgt[idx])
        coef = model.estimator_.coef_ / scales
    else:
        model = Ridge(alpha=1e-4)
        model.fit(lib_norm[idx], tgt[idx])
        coef = model.coef_ / scales
    
    return coef

def weak_form_sindy(data, dx, dy, dt_frame, test_width=5, margin=5):
    """Weak form SINDy using test functions"""
    # Gaussian test function
    x = np.arange(-test_width, test_width+1)
    y = np.arange(-test_width, test_width+1)
    X, Y = np.meshgrid(x, y)
    sigma = test_width / 2
    phi = np.exp(-(X**2 + Y**2) / (2*sigma**2))
    phi /= phi.sum()
    
    lib_list, tgt_list = [], []
    
    for i in range(len(data) - 1):
        u = data[i]
        dudt = (data[i + 1] - u) / dt_frame
        
        ux, uy, lap, biharm = fd_derivatives(u, dx, dy)
        grad_sq = ux**2 + uy**2
        
        # Convolve with test function
        from scipy.ndimage import convolve
        dudt_w = convolve(dudt, phi, mode='wrap')
        lap_w = convolve(lap, phi, mode='wrap')
        biharm_w = convolve(biharm, phi, mode='wrap')
        grad_sq_w = convolve(grad_sq, phi, mode='wrap')
        
        inner = (slice(margin, -margin), slice(margin, -margin))
        lib_list.append(np.column_stack([
            lap_w[inner].flatten(),
            biharm_w[inner].flatten(),
            grad_sq_w[inner].flatten()
        ]))
        tgt_list.append(dudt_w[inner].flatten())
    
    lib = np.vstack(lib_list)
    tgt = np.concatenate(tgt_list)
    
    scales = np.std(lib, axis=0)
    scales[scales < 1e-10] = 1.0
    
    n_samples = min(20000, len(tgt))
    idx = np.random.choice(len(tgt), n_samples, replace=False)
    
    model = Ridge(alpha=1e-4)
    model.fit(lib / scales, tgt)
    return model.coef_ / scales

def ensemble_sindy(data, dx, dy, dt_frame, n_ensemble=5):
    """Ensemble SINDy with bootstrap"""
    coefs = []
    n_frames = len(data)
    
    for _ in range(n_ensemble):
        # Bootstrap sample of frames
        idx = np.random.choice(n_frames, n_frames, replace=True)
        idx = np.sort(np.unique(idx))
        if len(idx) < 3:
            continue
        data_boot = data[idx]
        coef = sindy_discover(data_boot, dx, dy, dt_frame)
        coefs.append(coef)
    
    return np.median(coefs, axis=0)

# =============================================================================
# METHOD DEFINITIONS
# =============================================================================

METHODS = {
    'M01_baseline': lambda d: sindy_discover(d, dx, dy, dt_frame),
    'M02_gaussian': lambda d: sindy_discover(preprocess_gaussian(d), dx, dy, dt_frame),
    'M03_median': lambda d: sindy_discover(preprocess_median(d), dx, dy, dt_frame),
    'M04_tv': lambda d: sindy_discover(preprocess_tv(d), dx, dy, dt_frame),
    'M05_bilateral': lambda d: sindy_discover(preprocess_bilateral(d), dx, dy, dt_frame),
    'M06_temporal': lambda d: sindy_discover(preprocess_temporal(d), dx, dy, dt_frame),
    'M07_huber': lambda d: sindy_discover(d, dx, dy, dt_frame, regressor='huber'),
    'M08_ransac': lambda d: sindy_discover(d, dx, dy, dt_frame, regressor='ransac'),
    'M09_fourier': lambda d: sindy_discover(d, dx, dy, dt_frame, use_fourier=True),
    'M10_weak_form': lambda d: weak_form_sindy(d, dx, dy, dt_frame),
    'M11_ensemble': lambda d: ensemble_sindy(d, dx, dy, dt_frame),
    'M12_dmd': lambda d: sindy_discover(standard_dmd(d, rank=30), dx, dy, dt_frame),
    'M13_dmd_opt': lambda d: sindy_discover(standard_dmd(d, rank=50), dx, dy, dt_frame),
    'M14_dmd_fourier': lambda d: sindy_discover(standard_dmd(d, rank=40), dx, dy, dt_frame, use_fourier=True),
    'M15_sparse_dmd': lambda d: sindy_discover(sparse_dmd(d, rank=30), dx, dy, dt_frame),
    'M16_multiscale_dmd': lambda d: sindy_discover(multiscale_dmd(d), dx, dy, dt_frame),
    'M17_physics_dmd': lambda d: sindy_discover(multiscale_dmd(d, low_rank=8, high_rank=20), dx, dy, dt_frame, use_fourier=True),
    'M18_detrend': lambda d: sindy_discover(preprocess_detrend(d), dx, dy, dt_frame),
    'M19_detrend_global': lambda d: sindy_discover(preprocess_detrend_global(d), dx, dy, dt_frame),
    'M20_detrend_dmd': lambda d: sindy_discover(standard_dmd(preprocess_detrend(d), rank=40), dx, dy, dt_frame),
    'M21_detrend_multiscale': lambda d: sindy_discover(multiscale_dmd(preprocess_detrend(d)), dx, dy, dt_frame),
}

# =============================================================================
# RUN BENCHMARK
# =============================================================================

def compute_error(coef_discovered, coef_true):
    """Compute relative error"""
    return np.linalg.norm(coef_discovered - coef_true) / np.linalg.norm(coef_true) * 100

print("\n" + "="*70)
print("RUNNING COMPREHENSIVE BENCHMARK")
print("="*70)

results = {}
all_results = []

# First test on clean data
print("\n--- Testing on N0_clean ---")
results['N0_clean'] = {}
for method_name, method_fn in METHODS.items():
    try:
        start = time.time()
        coef = method_fn(u_clean)
        elapsed = time.time() - start
        error = compute_error(coef, TRUE_COEF)
        
        results['N0_clean'][method_name] = {
            'error': float(error),
            'coef': coef.tolist(),
            'time': float(elapsed)
        }
        
        all_results.append({
            'noise': 'N0_clean',
            'method': method_name,
            'error': float(error),
            'coef': coef.tolist()
        })
        
        status = "✓" if error < 50 else "✗"
        print(f"   {status} {method_name}: {error:.1f}% ({elapsed:.1f}s)")
    except Exception as e:
        results['N0_clean'][method_name] = {
            'error': float('inf'),
            'coef': [0, 0, 0],
            'time': 0,
            'error_msg': str(e)
        }
        print(f"   ✗ {method_name}: FAILED - {str(e)[:30]}")

# Test on all noise types
for noise_name, noisy_data in noisy_datasets.items():
    print(f"\n--- Testing on {noise_name} ---")
    results[noise_name] = {}
    
    for method_name, method_fn in METHODS.items():
        try:
            start = time.time()
            coef = method_fn(noisy_data)
            elapsed = time.time() - start
            error = compute_error(coef, TRUE_COEF)
            
            results[noise_name][method_name] = {
                'error': float(error),
                'coef': coef.tolist(),
                'time': float(elapsed)
            }
            
            all_results.append({
                'noise': noise_name,
                'method': method_name,
                'error': float(error),
                'coef': coef.tolist()
            })
            
            status = "✓" if error < 50 else "✗"
            print(f"   {status} {method_name}: {error:.1f}% ({elapsed:.1f}s)")
            
        except Exception as e:
            results[noise_name][method_name] = {
                'error': float('inf'),
                'coef': [0, 0, 0],
                'time': 0,
                'error_msg': str(e)
            }
            print(f"   ✗ {method_name}: FAILED")

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Best method per noise type
print("\n[Best method per noise type]")
all_noise_types = ['N0_clean'] + list(NOISE_CONFIGS.keys())
for noise_name in all_noise_types:
    if noise_name in results:
        best = min(results[noise_name].items(), 
                   key=lambda x: x[1]['error'] if x[1]['error'] != float('inf') else 1e10)
        print(f"   {noise_name}: {best[0]} ({best[1]['error']:.1f}%)")

# Average error per method across all noise types
print("\n[Average error per method (across all noise types)]")
method_avg = {}
for method_name in METHODS.keys():
    errors = []
    for noise_name in all_noise_types:
        if noise_name in results and method_name in results[noise_name]:
            err = results[noise_name][method_name]['error']
            if err != float('inf'):
                errors.append(err)
    if errors:
        method_avg[method_name] = np.mean(errors)

# Sort by average error
sorted_methods = sorted(method_avg.items(), key=lambda x: x[1])
print("\n   RANK  METHOD                  AVG ERROR")
print("   " + "-"*45)
for rank, (method, avg_err) in enumerate(sorted_methods, 1):
    marker = "★" if rank <= 3 else " "
    print(f"   {marker} {rank:2d}.  {method:<22} {avg_err:>8.1f}%")

# Create results table
print("\n[Full results table]")
print("\n" + " "*20, end="")
for noise in all_noise_types:
    print(f"{noise[:8]:>10}", end="")
print()
print("-"*100)

for method in METHODS.keys():
    print(f"{method:<20}", end="")
    for noise in all_noise_types:
        if noise in results and method in results[noise]:
            err = results[noise][method]['error']
            if err != float('inf'):
                print(f"{err:>10.1f}", end="")
            else:
                print(f"{'FAIL':>10}", end="")
        else:
            print(f"{'-':>10}", end="")
    print()

# =============================================================================
# SAVE RESULTS
# =============================================================================

output = {
    'true_coef': TRUE_COEF.tolist(),
    'noise_configs': ['N0_clean'] + list(NOISE_CONFIGS.keys()),
    'methods': list(METHODS.keys()),
    'results': results,
    'rankings': {m: float(e) for m, e in sorted_methods},
    'best_overall': sorted_methods[0][0] if sorted_methods else None,
    'best_per_noise': {n: min(results[n].items(), key=lambda x: x[1]['error'])[0] 
                       for n in results.keys()}
}

with open('/home/abhishek/Downloads/s3/mldm/sindy/research_ideas/benchmark_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "="*70)
print("FINAL RANKINGS")
print("="*70)
print(f"\n🥇 BEST OVERALL: {sorted_methods[0][0]} ({sorted_methods[0][1]:.1f}% avg error)")
print(f"🥈 2nd PLACE:    {sorted_methods[1][0]} ({sorted_methods[1][1]:.1f}% avg error)")
print(f"🥉 3rd PLACE:    {sorted_methods[2][0]} ({sorted_methods[2][1]:.1f}% avg error)")

print("\n✓ Results saved to benchmark_results.json")
print("="*70)
