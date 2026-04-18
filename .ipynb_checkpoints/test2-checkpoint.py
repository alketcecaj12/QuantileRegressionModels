import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def quantile_regression_housing(
    n: int = 500,
    quantiles: list = [0.10, 0.25, 0.50, 0.75, 0.90],
    seed: int = 42,
    plot: bool = True,
) -> dict:
    """
    Quantile regression on synthetic Copenhagen housing price data.

    Parameters
    ----------
    n         : number of observations
    quantiles : list of quantile levels to fit
    seed      : random seed for reproducibility
    plot      : whether to display the fan chart

    Returns
    -------
    dict with keys:
        'data'        : pd.DataFrame with size and price columns
        'qr_models'   : dict {quantile: fitted QuantReg result}
        'ols_model'   : fitted OLS result
        'pinball'     : pd.DataFrame with pinball loss comparison
    """
    np.random.seed(seed)

    # ── 1. Generate data ─────────────────────────────────────────────────────
    size_sqm   = np.random.uniform(40, 200, n)
    noise      = np.random.normal(0, 500 + 25 * size_sqm, n)
    price_kDKK = np.clip(800 + 35 * size_sqm + noise, 500, None)
    df         = pd.DataFrame({'price': price_kDKK, 'size': size_sqm})

    # ── 2. Fit models ─────────────────────────────────────────────────────────
    qr_models = {q: smf.quantreg('price ~ size', df).fit(q=q) for q in quantiles}
    ols_model = smf.ols('price ~ size', df).fit()

    # ── 3. Coefficient summary ───────────────────────────────────────────────
    print(f"{'Model':<12} {'Intercept':>12} {'Slope (size)':>14}")
    print("-" * 40)
    for q in quantiles:
        print(f"  Q{int(q*100):<8} "
              f"{qr_models[q].params['Intercept']:>12.1f} "
              f"{qr_models[q].params['size']:>14.2f}")
    print(f"  {'OLS':<10} "
          f"{ols_model.params['Intercept']:>12.1f} "
          f"{ols_model.params['size']:>14.2f}")

    # ── 4. Pinball loss ───────────────────────────────────────────────────────
    def _pinball(y, yhat, tau):
        e = np.array(y) - np.array(yhat)
        return np.mean(np.where(e >= 0, tau * e, (tau - 1) * e))

    rows = []
    for q in quantiles:
        ql  = _pinball(df['price'], qr_models[q].predict(df), q)
        ol  = _pinball(df['price'], ols_model.predict(df), q)
        rows.append({'quantile': q, 'QR_loss': ql, 'OLS_loss': ol,
                     'improvement_pct': (ol - ql) / ol * 100})
    pinball_df = pd.DataFrame(rows)

    print("\nPinball Loss — QR vs OLS:")
    for _, row in pinball_df.iterrows():
        print(f"  Q{int(row['quantile']*100)}: "
              f"QR={row['QR_loss']:.2f}  "
              f"OLS={row['OLS_loss']:.2f}  "
              f"improvement={row['improvement_pct']:+.1f}%")

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    if plot:
        palette = {0.10: '#2166ac', 0.25: '#74add1', 0.50: '#1a9641',
                   0.75: '#f46d43', 0.90: '#d73027'}
        labels  = {0.10: 'Q10 — Cheap end', 0.25: 'Q25', 0.50: 'Q50 — Median',
                   0.75: 'Q75',             0.90: 'Q90 — Luxury end'}

        x_range = np.linspace(40, 200, 300)
        pred_df = pd.DataFrame({'size': x_range})

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.scatter(df['size'], df['price'], alpha=0.2, s=12,
                   color='gray', label='Observed')
        ax.fill_between(x_range,
                        qr_models[0.10].predict(pred_df),
                        qr_models[0.90].predict(pred_df),
                        alpha=0.10, color='steelblue', label='Q10–Q90 band')
        for q in quantiles:
            ax.plot(x_range, qr_models[q].predict(pred_df),
                    color=palette[q], lw=2, label=labels[q])
        ax.plot(x_range, ols_model.predict(pred_df),
                color='black', lw=2, linestyle='--', label='OLS mean')

        ax.set_xlabel('Size (sqm)', fontsize=12)
        ax.set_ylabel('Price (kDKK)', fontsize=12)
        ax.set_title('Quantile Regression — Copenhagen Housing Prices\n'
                     'Price spread widens for larger homes', fontsize=13)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

    return {'data': df, 'qr_models': qr_models,
            'ols_model': ols_model, 'pinball': pinball_df}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = quantile_regression_housing()

    # Access individual components
    df        = results['data']
    qr_models = results['qr_models']
    ols       = results['ols_model']
    pinball   = results['pinball']