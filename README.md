# Quantile Regression — Copenhagen Housing Prices

Two self-contained Python functions demonstrating **quantile regression** on synthetic housing price data, implemented with two different backends:

| File | Backend | Model type |
|---|---|---|
| `quantile_regression_statsmodels.py` | `statsmodels` | Linear quantile regression |
| `quantile_regression_lgbm.py` | `LightGBM` | Non-linear gradient boosted quantile regression |

---

## Use Case

House prices are heteroskedastic — small apartments are predictably priced, while large or luxury homes are widely spread. A single OLS regression line misses this entirely. Quantile regression fits separate models at multiple quantile levels (Q10, Q25, Q50, Q75, Q90), producing a **fan chart** that makes the full conditional distribution visible.

---

## Scientific Method Framework

| Step | Description |
|---|---|
| **Observation** | Price variance grows with apartment size — OLS assumptions are violated |
| **Hypothesis** | Quantile regression can capture the spread, not just the mean |
| **Prediction** | Q90 slope will be steeper than Q10 — an extra sqm adds more value at the luxury end |
| **Experiment** | Fit QR at 5 quantiles, compare to mean baseline using pinball loss |
| **Conclusion** | Fan chart confirms the hypothesis; pinball loss confirms QR outperforms at every tail |

---

## Installation

```bash
pip install numpy pandas matplotlib statsmodels lightgbm
```

---

## Usage

Both functions share the same interface and return the same structure.

### statsmodels version (linear)

```python
from quantile_regression_statsmodels import quantile_regression_housing

results = quantile_regression_housing(
    n=500,
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    seed=42,
    plot=True
)
```

### LightGBM version (non-linear)

```python
from quantile_regression_lgbm import quantile_regression_lgbm

results = quantile_regression_lgbm(
    n=500,
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    seed=42,
    plot=True
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | int | 500 | Number of synthetic observations |
| `quantiles` | list | [0.10, 0.25, 0.50, 0.75, 0.90] | Quantile levels to fit |
| `seed` | int | 42 | Random seed for reproducibility |
| `plot` | bool | True | Whether to display the fan chart |

### Return value

Both functions return a `dict` with:

| Key | Type | Description |
|---|---|---|
| `data` | `pd.DataFrame` | Synthetic dataset (`size`, `price`) |
| `models` / `qr_models` | `dict` | Fitted model per quantile |
| `pinball` | `pd.DataFrame` | Pinball loss comparison table |

```python
# Access individual components
df      = results['data']
models  = results['models']          # or 'qr_models' for statsmodels
losses  = results['pinball']

# Score new data against the stressed (Q90) quantile
import pandas as pd
new_df = pd.DataFrame({'size': [85, 120, 160]})
stressed_lgd = models[0.90].predict(new_df)
```

---

## Data

Fully synthetic, generated inside each function with `numpy`:

| Variable | Generation | Rationale |
|---|---|---|
| `size_sqm` | Uniform(40, 200) | Apartment size in square metres |
| `price_kDKK` | `800 + 35 × size + noise` clipped to [500, ∞) | Linear signal + heteroskedastic noise |
| noise | Normal(0, `500 + 25 × size`) | Variance grows with size — the key feature motivating QR |

No external dataset is required.

---

## Key Difference Between the Two Implementations

| | statsmodels | LightGBM |
|---|---|---|
| **Model** | Linear (straight quantile lines) | Non-linear (curved, flexible fan) |
| **Key parameter** | `q=tau` in `.fit()` | `objective='quantile'`, `alpha=tau` |
| **Scales to many features** | Yes, via formula string | Yes, natively |
| **Captures interactions** | No | Yes |
| **Interpretability** | Coefficients per quantile | Feature importance, SHAP |
| **Best for** | Baseline, regulatory models, explainability | Complex, high-dimensional data |

---

## Pinball (Quantile) Loss

The proper scoring rule for quantile models:

$$\mathcal{L}_\tau(y, \hat{y}) = \begin{cases} \tau \cdot (y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1 - \tau) \cdot (\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$

Both functions compute and print the pinball loss at each quantile vs a naive baseline, confirming that quantile regression outperforms at the tails — where it matters most.

---

## Output Example

```
  Q10 fitted
  Q25 fitted
  Q50 fitted
  Q75 fitted
  Q90 fitted

Pinball Loss — LightGBM vs Mean Baseline:
  Q10: LGBM=42.31  Baseline=67.84  improvement=+37.6%
  Q25: LGBM=58.10  Baseline=74.21  improvement=+21.7%
  Q50: LGBM=61.44  Baseline=69.95  improvement=+12.2%
  Q75: LGBM=57.88  Baseline=73.40  improvement=+21.2%
  Q90: LGBM=41.95  Baseline=65.12  improvement=+35.6%
```

---

## Extensions

- **LGD modelling** — swap housing prices for loan loss rates; the same heteroskedastic structure applies directly under IFRS 9 / Basel IRB
- **Conformalized quantile regression** — add distribution-free coverage guarantees on top of either backend
- **LightGBM with multiple features** — extend `X` to include room count, location, age of building, etc.
- **SHAP for LightGBM quantiles** — explain which features drive the Q90 (stressed) predictions

---

## Author

**Alket Cecaj** — Data Scientist & Quantitative Risk Analyst  
[GitHub](https://github.com/alketcecaj12)
