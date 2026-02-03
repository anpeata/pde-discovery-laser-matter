# Methodology

## Overview

This document describes the complete methodology for data-driven PDE discovery from experimental laser-matter interaction images.

## Pipeline Architecture

```
Raw Images → Preprocessing → Derivative Computation → Library Construction → Sparse Regression → Validation
```

## 1. Data Preprocessing

### 1.1 Optical Flow Registration
**Purpose:** Compensate for motion artifacts and camera drift in image sequences.

**Methods:**
- **Farnebäck Optical Flow**: Dense polynomial expansion approach
- **DIS (Dense Inverse Search)**: Faster variational method

**Results:**
- Farnebäck: 18.5% RMSE improvement (6.90 → 5.62 px)
- DIS: 16.9% RMSE improvement (6.90 → 5.73 px)

### 1.2 Blockwise Temporal Averaging
**Purpose:** Reduce noise amplification in derivative computation.

**Implementation:**
- Group consecutive frames into blocks (typically 3-5 frames)
- Compute block-averaged derivatives
- Significantly reduces derivative noise without excessive smoothing

**Performance:**
- 8× reduction in coefficient error compared to pointwise methods
- Essential for robust real-world data processing

### 1.3 Denoising
**Purpose:** Remove high-frequency sensor noise.

**Methods:**
- Gaussian smoothing (σ = 1.0-2.0 pixels)
- Applied before derivative computation
- Preserves spatial features while reducing noise

## 2. Derivative Computation

### 2.1 Spatial Derivatives
**Method:** Central finite differences

```python
u_x[i,j] = (u[i+1,j] - u[i-1,j]) / (2*dx)
u_y[i,j] = (u[i,j+1] - u[i,j-1]) / (2*dy)
u_xx[i,j] = (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / (dx²)
u_yy[i,j] = (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / (dy²)
∇²u = u_xx + u_yy
```

### 2.2 Temporal Derivative
**Method:** Forward/backward finite differences

```python
u_t[i,j,t] = (u[i,j,t+1] - u[i,j,t]) / dt
```

### 2.3 Stabilization Techniques
- **Translation alignment**: Register derivatives to reference frame
- **Sign constraints**: Enforce physical priors (e.g., diffusion term must be negative)
- **Outlier removal**: Filter extreme derivative values

## 3. Library Construction

### 3.1 Standard Library
**Linear terms:**
- u (reaction)
- u_x, u_y (advection)
- ∇²u (diffusion)

**Nonlinear terms:**
- u² (nonlinear reaction)
- u·u_x, u·u_y (nonlinear advection)
- u_x·u_y (shear coupling)

### 3.2 Library Matrix
Construct design matrix Θ where each column is a candidate term evaluated at all spatiotemporal points:

```
Θ = [u, u_x, u_y, ∇²u, u², u·u_x, u·u_y, u_x·u_y, ...]
```

Target vector: ∂u/∂t (computed temporal derivative)

## 4. Sparse Regression (STRidge)

### 4.1 Algorithm
**Sequentially Thresholded Ridge Regression**

```
1. Initialize: ξ = (Θ^T Θ + λI)^(-1) Θ^T (∂u/∂t)
2. Threshold: Set small coefficients to zero (|ξ_i| < threshold)
3. Refit: Solve Ridge regression on remaining terms
4. Repeat: Until convergence or max iterations
```

### 4.2 Hyperparameters
- **Ridge parameter (λ)**: 0.001 - 0.01 (regularization strength)
- **Threshold**: 0.01 - 0.1 (sparsity control)
- **Max iterations**: 10-20

### 4.3 Model Selection Criteria
- **R² (coefficient of determination)**: Goodness of fit
- **Sparsity**: Number of active terms
- **Rollout stability**: k-step forward simulation error
- **Physical consistency**: Sign and magnitude of coefficients

## 5. Validation Framework

### 5.1 Time Holdout
**Setup:** 80% training, 20% testing (temporal split)

**Metrics:**
- R² (train/test)
- RMSE (Root Mean Square Error)
- nRMSE (normalized RMSE)
- Correlation coefficient

### 5.2 Spatial Holdout
**Setup:** Divide image into patches, cross-validate

**Purpose:** Test spatial generalization (not just temporal)

### 5.3 One-Step Prediction
**Method:** Use discovered PDE to predict next frame

**Formula:** u(t+1) = u(t) + dt·f(u,u_x,u_y,∇²u,...)

**Metric:** Mean absolute error between predicted and actual

### 5.4 Rollout Stability (k-Step)
**Method:** Iteratively apply PDE for k time steps

**Implementation:**
```python
u_rollout[0] = u_initial
for i in range(k):
    u_rollout[i+1] = u_rollout[i] + dt*RHS(u_rollout[i])
```

**Metrics:**
- nRMSE at each step
- Error growth rate
- Stability (divergence detection)

**Standard k values:** k=5, 10, 20

## 6. Robustness Enhancements

### 6.1 Blockwise vs Pointwise
**Blockwise (recommended for noisy data):**
- Average over temporal blocks
- Reduces noise by √block_size
- Essential for real-world data

**Pointwise (for clean synthetic data):**
- No averaging
- Maximum temporal resolution
- Can recover perfect coefficients on clean data

### 6.2 Sign Constraints
**Purpose:** Enforce physical priors

**Example:**
- Diffusion (∇²u): must be negative (smoothing)
- Advection: no sign constraint
- Reaction: problem-dependent

**Implementation:** Constrained optimization in regression step

### 6.3 Translation Stabilization
**Purpose:** Handle frame-to-frame registration errors

**Method:**
- Estimate per-frame translations
- Align all frames to reference
- Compute derivatives on aligned sequence

## 7. Ground-Truth Validation (KS2D)

### 7.1 Kuramoto-Sivashinsky 2D Equation
**True PDE:**
```
∂u/∂t = -∇²u - ∇⁴u - 0.5|∇u|²
```

### 7.2 Synthetic Data Generation
- Numerical solution on 64×64 grid
- 100 time steps
- Known ground-truth coefficients

### 7.3 Corruption Tests
- **N0 (Clean)**: No noise
- **N2 (5% noise)**: Gaussian noise, 5% relative
- **N5 (Jitter + noise)**: Translation + noise

### 7.4 Performance Metrics
- Coefficient relative error: |ξ_estimated - ξ_true| / |ξ_true|
- Rollout RMSE over 50 steps
- R² on test set

## 8. Real Image Analysis

### 8.1 Dataset
- 51 TIFF frames
- Laser-matter interaction experiment
- Unknown ground-truth PDE

### 8.2 Preprocessing Pipeline
1. Load and normalize images
2. Downsample (if needed for computational efficiency)
3. Gaussian denoising
4. Optical flow registration
5. Blockwise averaging

### 8.3 Model Comparison
**Model 3 (Linear):**
```
u_t = -0.144·u + 0.018·u_x + 0.014·u_y - 0.0003·∇²u
```
- Most rollout-stable
- Lower R² but robust dynamics

**Model 4 (Nonlinear):**
```
u_t = 0.345·u + 0.008·u_x + 0.005·u_y - 0.0001·∇²u - 0.617·u²
```
- Highest R² (0.459 on test)
- Less stable for long rollouts

## 9. Implementation Details

### 9.1 Software Dependencies
- Python 3.8+
- NumPy (numerical arrays)
- OpenCV (optical flow)
- SciPy (optimization)
- scikit-learn (regression)
- Matplotlib (visualization)

### 9.2 Computational Requirements
- RAM: 8-16 GB (depends on image resolution)
- GPU: Optional (for accelerated optical flow)
- Runtime: 5-30 minutes (full pipeline on 51 images)

### 9.3 Reproducibility
- Fixed random seeds
- Documented hyperparameters
- Version-controlled code

## References

1. **SINDy Framework**: Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." PNAS.

2. **PDE-FIND**: Rudy, S. H., Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2017). "Data-driven discovery of partial differential equations." Science Advances.

3. **Weak-Form Discovery**: Schaeffer, H., & McCalla, S. G. (2017). "Sparse model selection via integral terms." Physical Review E.

4. **STRidge**: Rudy et al. (2017). Sequentially Thresholded Ridge regression for PDE coefficient identification.
