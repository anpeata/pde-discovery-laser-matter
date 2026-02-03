# Robustness Strategies for PDE Discovery

## Overview

This document provides practical guidance on achieving robust PDE coefficient recovery under realistic measurement corruptions (noise, drift, registration errors).

---

## Quick Reference

| Data Quality | Recommended Method | Key Parameters |
|--------------|-------------------|----------------|
| Clean synthetic | Pointwise | Standard STRidge |
| Low noise (<2%) | Blockwise | `block_size=3` |
| Medium noise (2-5%) | Blockwise | `block_size=5` |
| High noise (>5%) | Blockwise + constraints | `block_size=5`, `sign_constraints` |
| Motion artifacts | Blockwise + stabilization | `--stabilize-shifts` |
| Real images | All techniques | Full pipeline |

---

## 1. Noise Handling: Blockwise Averaging

### 1.1 Why Blockwise Works

**Problem:** Finite differences amplify noise
```
u_t ≈ (u[t+1] - u[t]) / dt
```
If u has noise σ, then u_t has noise √2·σ/dt (amplified by 1/dt).

**Solution:** Average over temporal blocks
```
u_block = mean(u[t], u[t+1], ..., u[t+k])
u_t_block = (u_block[next] - u_block[current]) / dt_block
```
Noise reduced by √k (where k is block size).

### 1.2 Implementation

**Blockwise averaging in practice:**
```python
block_size = 5
num_blocks = len(frames) // block_size

for i in range(num_blocks):
    block = frames[i*block_size : (i+1)*block_size]
    u_block[i] = np.mean(block, axis=0)
    
# Compute derivatives on u_block instead of raw frames
```

### 1.3 Trade-offs

**Pros:**
- ✅ Dramatic noise reduction (5-8× less error)
- ✅ Stable rollouts
- ✅ Simple to implement

**Cons:**
- ⚠️ Reduced temporal resolution (fewer effective time points)
- ⚠️ May smooth out fast dynamics

**Recommendation:** Use block_size = 3-5 for most real data.

---

## 2. Motion Artifact Handling: Translation Stabilization

### 2.1 The Problem

Real image sequences often have:
- Camera drift
- Sample motion
- Registration errors

These create spurious spatial gradients that corrupt derivative estimation.

### 2.2 Solution: Optical Flow Registration

**Step 1: Estimate motion**
```python
import cv2

# Farnebäck optical flow (dense)
flow = cv2.calcOpticalFlowFarneback(
    prev_frame, curr_frame,
    None, 0.5, 3, 15, 3, 5, 1.2, 0
)

# DIS optical flow (faster alternative)
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
flow = dis.calc(prev_frame, curr_frame, None)
```

**Step 2: Register frames**
```python
# Create coordinate grid
h, w = frame.shape
x, y = np.meshgrid(np.arange(w), np.arange(h))

# Apply flow
x_new = x + flow[...,0]
y_new = y + flow[...,1]

# Remap to registered frame
registered = cv2.remap(curr_frame, x_new, y_new, cv2.INTER_LINEAR)
```

**Performance:**
- Farnebäck: 18.5% RMSE improvement
- DIS: 16.9% RMSE improvement

### 2.3 When to Use

**Always use for real images** unless:
- Data is already registered (e.g., synthetic)
- No motion expected (fixed sample)
- Motion is the phenomenon of interest

---

## 3. Physical Constraints: Sign Constraints

### 3.1 Motivation

Known physics can guide coefficient recovery:
- Diffusion (∇²u): should smooth → **negative coefficient**
- Advection (u_x, u_y): transport → **no sign constraint**
- Reaction: problem-dependent

### 3.2 Implementation

**In STRidge regression:**
```python
from scipy.optimize import lsq_linear

# Standard STRidge (unconstrained)
coef = np.linalg.lstsq(Theta, ut, rcond=None)[0]

# With sign constraints
# Example: [∇²u, ∇⁴u, |∇u|²] should all be negative
bounds = (
    [-np.inf, -np.inf, -np.inf],  # lower bounds
    [0, 0, 0]                       # upper bounds (≤0)
)
result = lsq_linear(Theta, ut, bounds=bounds)
coef = result.x
```

### 3.3 When to Use

**Use sign constraints when:**
- Physics clearly dictates coefficient sign
- Data is very noisy (N5-level corruption)
- Unconstrained regression gives unphysical results

**Example (KS2D):**
```bash
python scripts/ks2d_stridge_benchmark.py \
    --regression sign_constrained \
    --sign-constraints=-1,-1,-1  # All negative (diffusion terms)
```

**Results:**
- N5 (jitter + noise): 21-34% error vs >100% unconstrained
- Stable rollouts vs divergence

---

## 4. Recommended Pipelines

### 4.1 Synthetic Clean Data (KS2D N0)

```bash
python scripts/ks2d_stridge_benchmark.py \
    --dictionary true \
    --method pointwise \
    --perturbation none
```

**Expected:** Perfect recovery (0% error, R²=1.0)

---

### 4.2 Synthetic Noisy Data (KS2D N2: 5% noise)

```bash
python scripts/ks2d_stridge_benchmark.py \
    --dictionary true \
    --method blockwise \
    --perturbation N2_noise \
    --noise-rel 0.05
```

**Expected:** ~5% error on linear terms, R²≈0.35

---

### 4.3 Synthetic Severe Corruption (KS2D N5: jitter + noise)

```bash
python scripts/ks2d_stridge_benchmark.py \
    --dictionary true \
    --method blockwise \
    --perturbation N5_shifts_noise \
    --shift-mode jitter \
    --shift-max 0.5 \
    --stabilize-shifts \
    --stabilize-mode to_first \
    --regression sign_constrained \
    --sign-constraints=-1,-1,-1
```

**Expected:** 21-34% error, stable rollout

---

### 4.4 Real Image Data (Laser-Matter)

```bash
# Full pipeline (registration + blockwise + validation)
python scripts/analyze_results.py
```

**Pipeline stages:**
1. Load images
2. Gaussian denoise (σ=1.5)
3. Optical flow registration (Farnebäck)
4. Blockwise averaging (block_size=3)
5. Compute derivatives (central finite diff)
6. Build library (u, u_x, u_y, ∇²u, u², ...)
7. STRidge regression (λ=0.01, threshold=0.01)
8. Validate (time holdout, spatial holdout, rollout k=10)

---

## 5. Model Selection Guidelines

### 5.1 Competing Objectives

| Objective | Favors |
|-----------|--------|
| Best pointwise fit | Complex model (more terms) |
| Stable rollout | Simple model (fewer terms) |
| Interpretability | Sparse model |
| Generalization | Conservative model |

### 5.2 Decision Tree

```
Is your goal simulation/forward integration?
├─ YES → Prioritize rollout stability (Model 3-type)
│   └─ Accept lower R² for stable dynamics
└─ NO → Is prediction horizon short (1-2 steps)?
    ├─ YES → Optimize for R² (Model 4-type)
    │   └─ Nonlinear terms acceptable
    └─ NO → Use ensemble or conservative choice
```

### 5.3 Red Flags

**Warning signs of overfitting:**
- ❌ R² > 0.8 on real noisy data
- ❌ >10 active terms
- ❌ Rollout diverges rapidly (k<5)
- ❌ Coefficients change sign with small perturbations

**Action:** Increase sparsity, remove nonlinear terms, or use blockwise averaging.

---

## 6. Hyperparameter Tuning

### 6.1 Critical Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `block_size` | 3 | 1-10 | Noise reduction vs temporal resolution |
| `ridge_lambda` | 0.01 | 0.001-0.1 | Regularization strength |
| `threshold` | 0.01 | 0.001-0.1 | Sparsity level |
| `denoise_sigma` | 1.5 | 0.5-3.0 | Spatial smoothing |

### 6.2 Tuning Strategy

**Step 1: Fix noise reduction**
- Start with blockwise (block_size=3)
- If still noisy, increase to 5
- If too smooth, reduce to 1 (pointwise)

**Step 2: Tune sparsity**
- Start with threshold=0.01
- If too many terms, increase to 0.05
- If too few, decrease to 0.005

**Step 3: Validate**
- Check rollout stability (k=10)
- If unstable, increase sparsity or simplify library
- If stable but poor fit, add terms or reduce threshold

**Cross-validation:**
- Use time-series split (not random shuffle)
- Validate on multiple criteria (R², rollout, one-step)

---

## 7. Debugging Guide

### 7.1 Poor R² (<0 or near 0)

**Possible causes:**
- Insufficient registration
- Too much smoothing (block_size too large)
- Library missing key terms
- Wrong temporal derivative (forward vs backward)

**Solutions:**
- Check registration quality (compute RMSE before/after)
- Reduce block_size or denoise_sigma
- Add terms to library
- Verify derivative computation

---

### 7.2 Unstable Rollout

**Possible causes:**
- Overfitting (too many terms)
- Numerical instability (time step too large)
- Incorrect coefficient signs
- Inadequate validation set

**Solutions:**
- Increase sparsity threshold
- Reduce dt for forward Euler integration
- Apply sign constraints
- Use spatial holdout in addition to time holdout

---

### 7.3 Wildly Varying Coefficients

**Possible causes:**
- Insufficient data
- Highly correlated library terms
- Noise dominating signal
- Registration errors

**Solutions:**
- Collect more frames
- Remove redundant terms (e.g., u² and u³ may be collinear)
- Increase blockwise averaging
- Improve registration (try DIS vs Farnebäck)

---

## 8. Best Practices Summary

### ✅ Do's

1. **Always use blockwise averaging** on real data (minimum block_size=3)
2. **Always register frames** with optical flow for real images
3. **Validate with rollouts**, not just R²
4. **Start with simple models** (linear terms only)
5. **Use sign constraints** when physics is known
6. **Cross-validate** with spatial holdout
7. **Document hyperparameters** for reproducibility
8. **Visualize derivatives** before regression (sanity check)

### ❌ Don'ts

1. **Don't trust high R² alone** (check rollout stability)
2. **Don't use pointwise on noisy data** (use blockwise)
3. **Don't skip registration** on real images
4. **Don't add terms without justification** (sparsity matters)
5. **Don't use random train-test split** (use temporal or spatial)
6. **Don't forget to normalize** (scale u to [0,1] or [-1,1])
7. **Don't over-smooth** (destroys signal)
8. **Don't ignore negative R²** (worse than baseline)

---

## 9. Comparative Performance

### 9.1 Method Ranking (Noisy Data)

| Rank | Method | Coefficient Error | Implementation Complexity |
|------|--------|-------------------|--------------------------|
| 🥇 | Blockwise + constraints | ~5-20% | Medium |
| 🥈 | Blockwise | ~5-10% | Low |
| 🥉 | Weakform (not implemented) | ~10-20% | High |
| 4th | Pointwise + denoising | ~20-40% | Low |
| 5th | Pointwise (no preprocessing) | >40% | Very low |

### 9.2 Computational Cost

| Method | Time per Frame | Total (51 frames) |
|--------|---------------|-------------------|
| Pointwise | <0.1 sec | ~5 sec |
| Blockwise | <0.1 sec | ~5 sec |
| Optical flow (Farnebäck) | 1-2 sec | 50-100 sec |
| Optical flow (DIS) | 0.5-1 sec | 25-50 sec |
| Full pipeline | — | 5-10 min |

**Bottleneck:** Optical flow registration (but essential for real data)

---

## 10. References

### Key Papers

1. **Blockwise/temporal averaging:**
   - Schaeffer, H. (2017). "Learning partial differential equations via data discovery and sparse optimization." *Proc. Roy. Soc. A*.

2. **Robustness strategies:**
   - Messenger, D. A., & Bortz, D. M. (2021). "Weak SINDy for partial differential equations." *Journal of Computational Physics*.

3. **Optical flow:**
   - Farnebäck, G. (2003). "Two-frame motion estimation based on polynomial expansion."
   - Kroeger, T., et al. (2016). "Fast optical flow using dense inverse search." *ECCV*.

### Implementation Resources

- Our code: `scripts/ks2d_stridge_benchmark.py`, `scripts/analyze_results.py`
- OpenCV optical flow: [docs.opencv.org](https://docs.opencv.org/master/d4/dee/tutorial_optical_flow.html)
- SINDy examples: [github.com/dynamicslab/pysindy](https://github.com/dynamicslab/pysindy)

---

## Conclusion

Robust PDE discovery requires careful preprocessing and validation. The key insight is that **blockwise averaging + optical flow registration** can reduce errors by an order of magnitude compared to naive pointwise methods. Combined with proper model selection (rollout stability > R²), this enables reliable discovery from real experimental data.
