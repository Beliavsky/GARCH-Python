""" Loads S&P 500 data, computes return statistics, fits GARCH and/or
GJR‑GARCH models, and prints ACFs of squared returns vs. squared
standardised returns.  Optionally plots GJR‑GARCH results. """
import numpy as np
import arch.data.sp500
from scipy.stats import skew, kurtosis
from arch import arch_model
import matplotlib.pyplot as plt

# ---------------- User‑set toggles -----------------
plot_gjr_garch = False
fit_garch      = False
fit_gjr_garch  = True          # set True to fit the GJR-GARCH(1,1,1)
max_lag        = 30            # number of lags for the ACF table
# ---------------------------------------------------

data   = arch.data.sp500.load()
market = data["Adj Close"]
print("symbol = sp500")
print("\nfirst and last dates:\n" + market.iloc[[0, -1]].to_string())

xret = 100 * market.pct_change().dropna()          # returns in %
print(
    "\nreturn stats:\n#obs" +
    "".join("%11s" % label for label in ["mean", "sd", "skew",
                                         "kurt", "min", "max"])
)
print(
    "%5d" % len(xret),
    *("%10.4f" % stat for stat in [np.mean(xret),  np.std(xret),
                                   skew(xret),     kurtosis(xret),
                                   np.min(xret),   np.max(xret)]),
    end="\n\n",
)

def _acf(series: np.ndarray, k: int) -> float:
    """Simple (unbiased) lag‑k autocorrelation."""
    return np.corrcoef(series[k:], series[:-k])[0, 1]

def _print_acf_table(sq_raw, sq_std, lags):
    print("Lag    ACF(ret^2) ACF(std_ret^2)")
    for k in lags:
        print(f"{k:3d}    {sq_raw[k-1]:10.4f}     {sq_std[k-1]:10.4f}")
    print()

if fit_garch:
    am  = arch_model(xret)
    res = am.fit(update_freq=0, disp="off")
    print(res.summary(), end="\n\n")

    # --- ACFs for squared returns vs. squared standardised returns
    cond_vol = res.conditional_volatility
    valid    = cond_vol.notna()
    sq_raw   = (xret[valid]**2).to_numpy()
    sq_std   = ((xret[valid] / cond_vol[valid])**2).to_numpy()
    acf_sq   = [_acf(sq_raw,  k) for k in range(1, max_lag + 1)]
    acf_std  = [_acf(sq_std,  k) for k in range(1, max_lag + 1)]
    _print_acf_table(acf_sq, acf_std, range(1, max_lag + 1))

if fit_gjr_garch:
    am  = arch_model(xret, p=1, o=1, q=1)          # GJR‑GARCH(1,1,1)
    res = am.fit(update_freq=0, disp="off")
    print(res.summary())

    # --- ACFs for squared returns vs. squared standardised returns
    cond_vol = res.conditional_volatility
    valid    = cond_vol.notna()
    sq_raw   = (xret[valid]**2).to_numpy()
    sq_std   = ((xret[valid] / cond_vol[valid])**2).to_numpy()
    if max_lag > 0:
        acf_sq   = [_acf(sq_raw,  k) for k in range(1, max_lag + 1)]
        acf_std  = [_acf(sq_std,  k) for k in range(1, max_lag + 1)]
        _print_acf_table(acf_sq, acf_std, range(1, max_lag + 1))

    if plot_gjr_garch:
        fig = res.plot(annualize="D")
        plt.show()
