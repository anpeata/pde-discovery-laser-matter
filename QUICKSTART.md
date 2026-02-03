# Quick Start Guide

Get your PDE discovery project up and running in minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

## Installation

### 1. Clone the Repository (after publishing to GitHub)

```bash
git clone https://github.com/YOUR_USERNAME/pde-discovery-laser-matter.git
cd pde-discovery-laser-matter
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Code

### Option 1: Full Pipeline (Recommended for First Run)

```bash
python scripts/run_all.py
```

This will:
- Generate all presentation figures
- Run patch-based analysis
- Perform transport-form PDE discovery
- Create comparative visualizations
- Output results to `outputs/latest/`

**Note:** Full pipeline may take 5-30 minutes depending on your hardware.

### Option 2: Quick Demo (Skip Heavy Computations)

```bash
python scripts/run_all.py --skip-heavy
```

### Option 3: Individual Scripts

**Ground-truth benchmark (synthetic data):**
```bash
python scripts/ks2d_stridge_benchmark.py --method blockwise
```

**Real image analysis:**
```bash
python scripts/analyze_results.py
```

**Generate figures only:**
```bash
python scripts/generate_presentation_figures.py
```

## Expected Outputs

After running, you should see:

### Console Output
```
Loading images...
Computing optical flow...
Registration RMSE: 6.90 → 5.62 px (18.5% improvement)
Building library matrix...
Running STRidge regression...
Model 3 - R² (test): -1.178, Rollout nRMSE: 4.631
Model 4 - R² (test): 0.459, Rollout nRMSE: 14.104
Best rollout-stable model: Model 3
Figures saved to figures/
```

### Generated Files

**Figures** (in `figures/`):
- `MODEL_COMPARISON.png` - Side-by-side model comparison
- `SLIDE3_PDE_Results_51images.png` - Main results slide
- `fig1_data_overview.png` - Input data visualization
- ... and more

**Outputs** (in `outputs/latest/`):
- JSON files with numerical results
- LaTeX tables
- Additional diagnostic figures

## Exploring the Results

### View Key Figures

**Windows:**
```bash
start figures\MODEL_COMPARISON.png
start figures\SLIDE3_PDE_Results_51images.png
```

**Linux/Mac:**
```bash
open figures/MODEL_COMPARISON.png
open figures/SLIDE3_PDE_Results_51images.png
```

### Run Jupyter Notebooks

```bash
# Install Jupyter (if not already installed)
pip install jupyter

# Launch notebook
jupyter notebook notebooks/PDE_Discovery_Registration_SINDy.ipynb
```

## Understanding the Results

### Key Metrics Explained

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **R²** | Variance explained | Close to 1.0 |
| **RMSE** | Root mean square error | Close to 0.0 |
| **nRMSE** | Normalized RMSE | <1.0 (lower is better) |
| **Rollout nRMSE** | Multi-step stability | <5.0 is stable |

### Model Selection

- **Model 3** (Linear): Best for forward simulation (stable rollout)
- **Model 4** (Nonlinear): Best for short-term prediction (high R²)

See [docs/RESULTS.md](docs/RESULTS.md) for detailed explanation.

## Customization

### Run with Different Parameters

**Change blockwise averaging size:**
```python
# Edit scripts/analyze_results.py
block_size = 5  # Default is 3
```

**Try different optical flow method:**
```python
# In analyze_results.py, change:
method = cv2.DISOpticalFlow_create(...)  # Instead of Farnebäck
```

**Adjust sparsity threshold:**
```python
# In STRidge regression section:
threshold = 0.05  # Default is 0.01 (higher = sparser)
```

### Add New Library Terms

Edit the library construction in `scripts/analyze_results.py`:

```python
# Add fourth-order diffusion
u_xxxx = laplacian_of_laplacian(u)
library = np.column_stack([u, u_x, u_y, lap_u, u_xxxx, u**2, ...])
```

## Troubleshooting

### ImportError: No module named 'cv2'

**Solution:**
```bash
pip install opencv-python
```

### Memory Error

**Solution:** Reduce image resolution or use fewer frames
```python
# In image loading section:
img = cv2.resize(img, None, fx=0.5, fy=0.5)  # Downsample by 2×
```

### "data/Real-Images/" not found

**Note:** Original data files are not included in the repository.

**Solution:** 
- Use synthetic benchmark instead: `python scripts/ks2d_stridge_benchmark.py`
- Or provide your own image sequence in `data/Real-Images/`

### Figures look different from documentation

**Reason:** Stochastic elements (random seeds, numerical precision)

**Solution:** This is normal; results should be qualitatively similar.

## Next Steps

1. ✅ Run the full pipeline
2. 📊 Explore generated figures
3. 📓 Open Jupyter notebooks for interactive analysis
4. 📚 Read [docs/METHODOLOGY.md](docs/METHODOLOGY.md) to understand the approach
5. 🔬 Experiment with different parameters
6. 🚀 Apply to your own image data

## Getting Help

- **Documentation:** See `docs/` folder for detailed guides
- **Issues:** Open an issue on GitHub (after publishing)
- **Email:** [your.email@example.com] (update this!)

## Advanced Usage

### Run Specific Benchmark Cases

**Clean data (perfect recovery):**
```bash
python scripts/ks2d_stridge_benchmark.py \
    --dictionary true \
    --method pointwise \
    --perturbation none
```

**Noisy data (robust methods):**
```bash
python scripts/ks2d_stridge_benchmark.py \
    --dictionary true \
    --method blockwise \
    --perturbation N2_noise \
    --noise-rel 0.05
```

**Severe corruption (all techniques):**
```bash
python scripts/ks2d_stridge_benchmark.py \
    --dictionary true \
    --method blockwise \
    --perturbation N5_shifts_noise \
    --stabilize-shifts \
    --regression sign_constrained \
    --sign-constraints=-1,-1,-1
```

### Batch Processing

Create a script to run multiple configurations:

```bash
# run_benchmark_suite.sh
for noise in 0.01 0.02 0.05 0.10; do
    python scripts/ks2d_stridge_benchmark.py \
        --method blockwise \
        --noise-rel $noise \
        --output-dir results/noise_$noise/
done
```

## Performance Tips

1. **Use DIS instead of Farnebäck** for faster optical flow (2× speedup)
2. **Downsample images** if processing is slow (set `downsample=4` or `8`)
3. **Reduce block size** if you have many frames (less memory)
4. **Use `--skip-heavy`** flag when testing changes

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pde_discovery_laser,
  author = {Your Name},
  title = {PDE Discovery for Laser-Matter Interaction},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/pde-discovery-laser-matter}
}
```

---

**Ready to start?** Just run `python scripts/run_all.py` and explore the results! 🚀
