import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb


def quantile_regression_lgbm(
    n: int = 500,
    quantiles: list = [0.10, 0.25, 0.50, 0.75, 0.90],
    seed: int = 42,
    plot: bool = True,
) -> dict:
    """
    Quantile regression with LightGBM on synthetic Copenhagen housing data.

    Parameters
    ----------
    n         : number of observations
    quantiles : list of quantile levels to fit
    seed      : random seed for reproducibility
    plot      : whether to display the fan chart

    Returns
    -------
    dict with keys:
        'data'      : pd.DataFrame with size and price columns
        'models'    : dict {quantile: trained LGBMRegressor}
        'pinball'   : pd.DataFrame with pinball loss comparison
    """
    np.random.seed(seed)

    # ── 1. Generate data ─────────────────────────────────────────────────────
    size_sqm   = np.random.uniform(40, 200, n)
    noise      = np.random.normal(0, 500 + 25 * size_sqm, n)
    price_kDKK = np.clip(800 + 35 * size_sqm + noise, 500, None)
    df         = pd.DataFrame({'price': price_kDKK, 'size': size_sqm})

    X = df[['size']]
    y = df['price']

    # ── 2. Fit one LightGBM model per quantile ────────────────────────────────
    lgbm_params = dict(
        objective='quantile',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=seed,
        verbose=-1,
    )

    models = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(alpha=q, **lgbm_params)
        model.fit(X, y)
        models[q] = model
        print(f"  Q{int(q*100):>2} fitted")

    # Also fit Q50 as the "median baseline" for pinball comparison with mean
    ols_baseline = df['price'].mean()  # trivial mean predictor for reference

    # ── 3. Pinball loss ───────────────────────────────────────────────────────
    def _pinball(y_true, y_pred, tau):
        e = np.array(y_true) - np.array(y_pred)
        return np.mean(np.where(e >= 0, tau * e, (tau - 1) * e))

    rows = []
    for q in quantiles:
        lgbm_pred = models[q].predict(X)
        mean_pred = np.full(len(y), ols_baseline)
        ql  = _pinball(y, lgbm_pred, q)
        bl  = _pinball(y, mean_pred, q)
        rows.append({'quantile': q, 'LGBM_loss': ql, 'Mean_baseline_loss': bl,
                     'improvement_pct': (bl - ql) / bl * 100})

    pinball_df = pd.DataFrame(rows)

    print("\nPinball Loss — LightGBM vs Mean Baseline:")
    for _, row in pinball_df.iterrows():
        print(f"  Q{int(row['quantile']*100)}: "
              f"LGBM={row['LGBM_loss']:.2f}  "
              f"Baseline={row['Mean_baseline_loss']:.2f}  "
              f"improvement={row['improvement_pct']:+.1f}%")

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    if plot:
        palette = {0.10: '#2166ac', 0.25: '#74add1', 0.50: '#1a9641',
                   0.75: '#f46d43', 0.90: '#d73027'}
        labels  = {0.10: 'Q10 — Cheap end', 0.25: 'Q25', 0.50: 'Q50 — Median',
                   0.75: 'Q75',             0.90: 'Q90 — Luxury end'}

        x_range  = np.linspace(40, 200, 300).reshape(-1, 1)
        pred_df  = pd.DataFrame(x_range, columns=['size'])

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.scatter(df['size'], df['price'], alpha=0.2, s=12,
                   color='gray', label='Observed')
        ax.fill_between(
            x_range.flatten(),
            models[0.10].predict(pred_df),
            models[0.90].predict(pred_df),
            alpha=0.10, color='steelblue', label='Q10–Q90 band'
        )
        for q in quantiles:
            ax.plot(x_range.flatten(), models[q].predict(pred_df),
                    color=palette[q], lw=2, label=labels[q])

        ax.axhline(ols_baseline, color='black', lw=2,
                   linestyle='--', label=f'Mean baseline ({ols_baseline:.0f} kDKK)')
        ax.set_xlabel('Size (sqm)', fontsize=12)
        ax.set_ylabel('Price (kDKK)', fontsize=12)
        ax.set_title('Quantile Regression with LightGBM — Copenhagen Housing Prices\n'
                     'Non-linear fan: price spread widens for larger homes', fontsize=13)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

    return {'data': df, 'models': models, 'pinball': pinball_df}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = quantile_regression_lgbm()