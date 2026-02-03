# PDE Discovery for Laser-Matter Interaction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Data-driven discovery of governing partial differential equations from experimental laser–matter interaction images using multiple sparse and regression-based approaches, supported by robust preprocessing pipelines.

*Presented at the 15th SLEIGHT Science Event (SSE #15), Manutech-SLEIGHT Graduate School.*

## Quick Start

```bash
git clone https://github.com/anpeata/pde-discovery-laser-matter.git
cd pde-discovery-laser-matter
pip install -r requirements.txt
python examples/basic_usage.py
```

## Overview

**Challenge:** Extract governing equations from noisy experimental image sequences

**Solution:**
- Optical flow registration → Motion compensation
- Blockwise averaging → Noise reduction  
- STRidge regression → Sparse coefficients
- Multi-metric validation → Stability analysis

## Key Results

### Synthetic Benchmark (KS 2D)
| Scenario | Method | Error | R² |
|----------|--------|-------|-----|
| Clean | Pointwise | 0.0% | 1.000 |
| 5% noise | Blockwise | 5.1% | 0.347 |
| Jitter+noise | Stabilized | 21-34% | 0.012 |

**Finding:** Blockwise averaging → 8× error reduction (42% → 5%)

### Real Laser Data

**Model 4** (Best Predictor): R²=0.459  
`u_t = 0.345u + 0.008u_x + 0.005u_y - 0.0001∇²u - 0.617u²`

**Model 3** (Most Stable): R²=-1.18  
`u_t = -0.144u + 0.018u_x + 0.014u_y - 0.0003∇²u`

## Repository Structure

```
scripts/        # 16 analysis scripts (run_all.py, analyze_results.py, ks2d_stridge_benchmark.py, etc.)
notebooks/      # 10 Jupyter notebooks (01-04 core, 05-10 collaborative research)
figures/        # 22 visualizations (core, presentation, publication)
results/        # Model metrics (JSON, CSV, TXT)
docs/           # METHODOLOGY.md, RESULTS.md, ROBUSTNESS.md + presentation/
examples/       # basic_usage.py
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Installation & first steps
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** - Technical details
- **[docs/RESULTS.md](docs/RESULTS.md)** - Experimental results
- **[docs/ROBUSTNESS.md](docs/ROBUSTNESS.md)** - Noise handling

## Methodology

1. **Preprocessing:** Optical flow registration (16.9-18.5% improvement), blockwise averaging
2. **Derivatives:** Central FD (spatial), forward FD (temporal)
3. **Library:** Linear (u, u_x, u_y, ∇²u) + nonlinear (u², u·u_x, etc.)
4. **Regression:** STRidge with sequential thresholding
5. **Validation:** Time holdout + spatial patches + rollout stability

## Contributors

**Team:** An, Sinjini, Abhishek, Ayomide (MLDM 2024-26)

## License

MIT License - See [LICENSE](LICENSE)

---

⭐ Star this repo if useful for your research!
