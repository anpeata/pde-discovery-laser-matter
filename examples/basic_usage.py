"""
Example: Basic PDE Discovery Workflow

This script demonstrates a minimal working example of PDE discovery
from image sequences using the project's methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Simulated data for demonstration (replace with real images)
def generate_synthetic_data(n_frames=20, h=50, w=50):
    """
    Generate synthetic spatiotemporal data for demonstration.
    In practice, replace this with your image loading code.
    """
    x = np.linspace(0, 10, w)
    y = np.linspace(0, 10, h)
    t = np.linspace(0, 5, n_frames)
    
    X, Y = np.meshgrid(x, y)
    
    # Simple advection-diffusion dynamics
    data = np.zeros((n_frames, h, w))
    for i, ti in enumerate(t):
        data[i] = np.exp(-0.1*ti) * np.sin(X - 0.5*ti) * np.cos(Y - 0.3*ti)
    
    return data, x, y, t


def compute_derivatives(u, dx, dy, dt):
    """
    Compute spatial and temporal derivatives using finite differences.
    
    Parameters:
    - u: spatiotemporal data (T, H, W)
    - dx, dy, dt: grid spacing
    
    Returns:
    - u_t, u_x, u_y, u_xx, u_yy (derivatives)
    """
    T, H, W = u.shape
    
    # Temporal derivative (forward difference)
    u_t = np.zeros((T-1, H, W))
    for i in range(T-1):
        u_t[i] = (u[i+1] - u[i]) / dt
    
    # Spatial derivatives (central differences, interior points only)
    u_x = np.zeros_like(u)
    u_y = np.zeros_like(u)
    u_xx = np.zeros_like(u)
    u_yy = np.zeros_like(u)
    
    for i in range(T):
        # First derivatives
        u_x[i, :, 1:-1] = (u[i, :, 2:] - u[i, :, :-2]) / (2*dx)
        u_y[i, 1:-1, :] = (u[i, 2:, :] - u[i, :-2, :]) / (2*dy)
        
        # Second derivatives (Laplacian)
        u_xx[i, :, 1:-1] = (u[i, :, 2:] - 2*u[i, :, 1:-1] + u[i, :, :-2]) / (dx**2)
        u_yy[i, 1:-1, :] = (u[i, 2:, :] - 2*u[i, 1:-1, :] + u[i, :-2, :]) / (dy**2)
    
    # Trim to match u_t dimensions and remove boundaries
    u_trimmed = u[:-1, 2:-2, 2:-2]
    u_x_trimmed = u_x[:-1, 2:-2, 2:-2]
    u_y_trimmed = u_y[:-1, 2:-2, 2:-2]
    lap_u = (u_xx + u_yy)[:-1, 2:-2, 2:-2]
    u_t_trimmed = u_t[:, 2:-2, 2:-2]
    
    return u_t_trimmed, u_trimmed, u_x_trimmed, u_y_trimmed, lap_u


def build_library(u, u_x, u_y, lap_u):
    """
    Construct library matrix of candidate terms.
    
    Returns:
    - Theta: (N, n_terms) library matrix
    - term_names: list of term descriptions
    """
    # Flatten spatial dimensions
    u_flat = u.flatten()
    u_x_flat = u_x.flatten()
    u_y_flat = u_y.flatten()
    lap_u_flat = lap_u.flatten()
    
    # Library: [1, u, u_x, u_y, lap_u, u^2]
    Theta = np.column_stack([
        np.ones_like(u_flat),
        u_flat,
        u_x_flat,
        u_y_flat,
        lap_u_flat,
        u_flat**2
    ])
    
    term_names = ['1', 'u', 'u_x', 'u_y', 'lap(u)', 'u^2']
    
    return Theta, term_names


def stridge_regression(Theta, u_t, alpha=0.01, threshold=0.01, max_iter=10):
    """
    Sequential Thresholded Ridge Regression (STRidge).
    
    Parameters:
    - Theta: library matrix (N, n_terms)
    - u_t: temporal derivative (N,)
    - alpha: Ridge regularization
    - threshold: sparsity threshold
    - max_iter: maximum iterations
    
    Returns:
    - coef: discovered coefficients
    """
    n_terms = Theta.shape[1]
    coef = np.ones(n_terms)  # Initialize
    
    for iteration in range(max_iter):
        # Ridge regression
        A = Theta.T @ Theta + alpha * np.eye(n_terms)
        b = Theta.T @ u_t
        coef = np.linalg.solve(A, b)
        
        # Threshold small coefficients
        mask = np.abs(coef) < threshold
        coef[mask] = 0
        
        # Refit on active terms
        active = ~mask
        if np.sum(active) == 0:
            break
        
        Theta_active = Theta[:, active]
        A_active = Theta_active.T @ Theta_active + alpha * np.eye(np.sum(active))
        b_active = Theta_active.T @ u_t
        coef_active = np.linalg.solve(A_active, b_active)
        
        coef[active] = coef_active
    
    return coef


def print_equation(coef, term_names, threshold=1e-6):
    """Print discovered PDE in readable form."""
    print("\nDiscovered PDE:")
    print("u_t = ", end="")
    
    terms = []
    for c, name in zip(coef, term_names):
        if abs(c) > threshold:
            sign = "+" if c > 0 else ""
            terms.append(f"{sign}{c:.4f}*{name}")
    
    if len(terms) == 0:
        print("0 (all terms eliminated)")
    else:
        equation = " ".join(terms)
        # Clean up formatting
        equation = equation.replace("+ -", "- ").replace("*1 ", " ")
        if equation.startswith("+"):
            equation = equation[1:]
        print(equation)
    
    print(f"\nNumber of active terms: {np.sum(np.abs(coef) > threshold)}/{len(coef)}")


def main():
    """Run complete PDE discovery example."""
    
    print("=" * 60)
    print("PDE Discovery Example")
    print("=" * 60)
    
    # Step 1: Load/generate data
    print("\n[1/5] Loading data...")
    u, x, y, t = generate_synthetic_data(n_frames=30, h=60, w=60)
    print(f"  Data shape: {u.shape} (time, height, width)")
    
    # Step 2: Compute derivatives
    print("\n[2/5] Computing derivatives...")
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = t[1] - t[0]
    
    u_t, u, u_x, u_y, lap_u = compute_derivatives(u, dx, dy, dt)
    print(f"  Derivative arrays: {u_t.shape}")
    
    # Step 3: Build library
    print("\n[3/5] Building library matrix...")
    Theta, term_names = build_library(u, u_x, u_y, lap_u)
    print(f"  Library shape: {Theta.shape}")
    print(f"  Terms: {', '.join(term_names)}")
    
    # Step 4: Sparse regression
    print("\n[4/5] Running STRidge regression...")
    u_t_flat = u_t.flatten()
    coef = stridge_regression(Theta, u_t_flat, alpha=0.01, threshold=0.01)
    
    # Step 5: Display results
    print("\n[5/5] Results:")
    print_equation(coef, term_names)
    
    # Compute R²
    u_t_pred = Theta @ coef
    ss_res = np.sum((u_t_flat - u_t_pred)**2)
    ss_tot = np.sum((u_t_flat - np.mean(u_t_flat))**2)
    r2 = 1 - ss_res / ss_tot
    print(f"\nR²: {r2:.3f}")
    
    # Visualization
    print("\n[Bonus] Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    # Sample frame
    axes[0].imshow(u[10], cmap='viridis')
    axes[0].set_title('Sample Frame (t=10)')
    axes[0].axis('off')
    
    # Coefficient bar chart
    active_mask = np.abs(coef) > 1e-6
    axes[1].bar(np.arange(len(coef))[active_mask], 
                coef[active_mask])
    axes[1].set_xticks(np.arange(len(coef))[active_mask])
    axes[1].set_xticklabels([term_names[i] for i in np.where(active_mask)[0]], 
                             rotation=45)
    axes[1].set_ylabel('Coefficient Value')
    axes[1].set_title('Discovered Coefficients')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Predicted vs actual
    sample = np.random.choice(len(u_t_flat), size=1000, replace=False)
    axes[2].scatter(u_t_flat[sample], u_t_pred[sample], 
                    alpha=0.5, s=1)
    lim = max(abs(u_t_flat[sample]).max(), abs(u_t_pred[sample]).max())
    axes[2].plot([-lim, lim], [-lim, lim], 'r--', lw=2, label='Perfect fit')
    axes[2].set_xlabel('Actual u_t')
    axes[2].set_ylabel('Predicted u_t')
    axes[2].set_title(f'Prediction Quality (R²={r2:.3f})')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pde_discovery_example.png', dpi=150, bbox_inches='tight')
    print("  Saved: pde_discovery_example.png")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
