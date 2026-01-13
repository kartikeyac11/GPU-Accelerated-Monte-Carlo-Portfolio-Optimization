# Monte Carlo Portfolio Optimization

**Exploring the Observable-Space Geometry of Regularized Sharpe-Optimal Portfolios**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project implements a GPU-accelerated Monte Carlo simulation framework for portfolio optimization. It generates thousands of random portfolios across multiple synthetic market environments and identifies Sharpe-optimal allocations subject to diversification constraints.

The key insight is visualizing portfolio performance in a 3D "observable space" defined by:
- **Risk** (portfolio volatility)
- **Return** (expected portfolio return)  
- **Entropy** (diversification measure)

This reveals the geometric structure underlying optimal portfolio selection.

## Features

- **GPU Acceleration**: Leverages PyTorch for CUDA-enabled parallel computation
- **Monte Carlo Sampling**: Generates 150,000 random portfolios per market scenario
- **Regularized Optimization**: L2 penalty prevents over-concentration in single assets
- **Box Constraints**: Maximum allocation caps ensure diversification
- **3D Visualization**: Interactive scatter plot of portfolio observables

## Mathematical Framework

### Portfolio Metrics

For a portfolio with weights $w$ and $n$ assets:

- **Expected Return**: $R_p = w^T \mu$
- **Portfolio Risk**: $\sigma_p = \sqrt{w^T \Sigma w}$
- **Sharpe Ratio**: $S = R_p / \sigma_p$
- **Entropy**: $H = -\sum_i w_i \log(w_i)$

### Regularized Objective

The optimization maximizes a regularized Sharpe ratio:

$$S_{reg} = \frac{w^T \mu}{\sqrt{w^T \Sigma w}} - \lambda \|w\|_2^2$$

Subject to constraints:
- $\sum_i w_i = 1$ (fully invested)
- $0 \leq w_i \leq w_{max}$ (box constraints)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-optimization.git
cd portfolio-optimization

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
```

## Usage

```python
python podistribution.py
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_assets` | 50 | Number of assets in universe |
| `n_sets` | 26 | Number of random market environments |
| `n_samples_mc` | 150,000 | Monte Carlo samples per environment |
| `w_max` | 0.30 | Maximum weight per asset |
| `lambda_reg` | 0.05 | L2 regularization strength |

## Output

The script produces:

1. **Console Output**: Per-environment metrics (return, risk, Sharpe, entropy)
2. **3D Scatter Plot**: Observable-space geometry colored by Sharpe ratio

### Sample Output

```
Set 001 | Return=0.142  Risk=0.312  Sharpe=0.455  Entropy=3.421
Set 002 | Return=0.138  Risk=0.298  Sharpe=0.463  Entropy=3.389
...
```

## Visualization

The 3D plot reveals the geometric structure of optimal portfolios:

- **X-axis**: Portfolio Risk (volatility)
- **Y-axis**: Portfolio Return
- **Z-axis**: Portfolio Entropy (diversification)
- **Color**: Sharpe Ratio (risk-adjusted return)

Higher entropy portfolios tend to cluster in specific risk-return regions, revealing the trade-off between diversification and performance.

## Key Insights

1. **Diversification-Performance Trade-off**: High-entropy portfolios often sacrifice peak Sharpe ratios for stability
2. **Constraint Effects**: Box constraints create natural clustering in observable space
3. **Market Dependence**: The geometry shifts significantly across different covariance structures

## Extending the Project

Ideas for future development:

- [ ] Add transaction cost modeling
- [ ] Implement rolling window backtesting
- [ ] Compare with analytical mean-variance optimization
- [ ] Add sector/factor constraints
- [ ] Export optimal portfolios to CSV

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification

---

*Built with PyTorch and curiosity about the geometry of optimal portfolios.*
