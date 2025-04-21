"""Load S&P 500 data, fit univariate volatility models
(GARCH, GJR-GARCH, APARCH), compare ACFs, fit a skew-t to each
model's normalized returns, and collect all parameters + likelihoods
in a table (without mixture parameters)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import arch.data.sp500
from scipy.stats import skew, kurtosis
from arch import arch_model
import matplotlib.pyplot as plt

from stats import acf, print_acf_table, print_stats_table, plot_norm_kde

try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# -------- User-set toggles ---------
plot_gjr_garch = False     # plot fitted GJR-GARCH volatility plots
plot_norm_dist = False     # draw KDE + mixture + N(0,1) + skew-t densities
fit_garch      = True
fit_gjr_garch  = True      # fit the GJR-GARCH(1,1,1)
fit_aparch     = True      # fit the APARCH(1,1,1) with power=1.0
max_lag        = 30        # lags for the ACF table (0 -> skip ACFs)
# -----------------------------------

# ---------- Load data ----------
data   = arch.data.sp500.load()
market = data["Adj Close"]
print("symbol = sp500")
print("\nfirst and last dates:\n" + market.iloc[[0, -1]].to_string())

xret = 100 * market.pct_change().dropna()  # daily % returns
raw_stats_row = (
    len(xret),
    np.mean(xret),  np.std(xret),
    skew(xret),     kurtosis(xret),
    np.min(xret),   np.max(xret),
)

# ---------- storage for results ----------
model_results  = []   # one dict per model
norm_stats_row = None
std_ret_series = None
acf_done       = False

# ---------- shared post-processing ----------
def process_std_ret(std_series):
    """Compute global normalized-return stats and print ACFs once."""
    global norm_stats_row, std_ret_series, acf_done

    if norm_stats_row is None:
        norm_stats_row = (
            len(std_series),
            np.mean(std_series),  np.std(std_series),
            skew(std_series),     kurtosis(std_series),
            np.min(std_series),   np.max(std_series),
        )
    std_ret_series = std_series

    if max_lag > 0 and not acf_done:
        sq_raw = (xret.loc[std_series.index]**2).to_numpy()
        sq_std = (std_series**2).to_numpy()
        print_acf_table(
            [acf(sq_raw, k) for k in range(1, max_lag+1)],
            [acf(sq_std, k) for k in range(1, max_lag+1)],
            range(1, max_lag+1),
        )
        acf_done = True

# ---------- helpers to fit conditional distributions ----------
def fit_mixture(std_series):
    """Fit 2-component normal mixture; return params for printing only."""
    if not SKLEARN_OK:
        print("sklearn not available -> mixture fit skipped")
        return

    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          random_state=0)
    gmm.fit(std_series.values.reshape(-1, 1))
    w   = gmm.weights_
    mu  = gmm.means_.flatten()
    sig = np.sqrt(gmm.covariances_.flatten())

    print("\n2-component normal mixture fitted to normalized returns")
    print("comp   weight     mean       sd")
    print(f"1     {w[0]:10.4f} {mu[0]:10.4f} {sig[0]:10.4f}")
    print(f"2     {w[1]:10.4f} {mu[1]:10.4f} {sig[1]:10.4f}")

def fit_skewt_params(std_series):
    """Fit a skewed Student-t if available, else symmetric Student-t."""
    try:
        from scipy.stats import skewt
        is_skew = True
    except ImportError:
        from scipy.stats import t as skewt
        is_skew = False

    params = skewt.fit(std_series)
    if is_skew:
        alpha, df, loc, scale = params
    else:
        df, loc, scale = params
        alpha = 0.0

    print("\n%s distribution fitted to normalized returns" %
          ("skew-t" if is_skew else "Student-t"))
    print(f"alpha={alpha:.4f}, df={df:.4f}, loc={loc:.4f}, scale={scale:.4f}")
    return alpha, df, loc, scale

# ---------- generic model runner ----------
def run_model(name, **am_kwargs):
    """
    Fit arch_model(xret, **am_kwargs), record results in model_results,
    and run shared post-processing.
    """
    res = arch_model(xret, **am_kwargs).fit(update_freq=0, disp="off")
    print(res.summary(), end="\n\n")

    # record volatility-model params + loglikelihood
    d = {"model": name, "loglikelihood": res.loglikelihood}
    d.update(res.params.to_dict())

    # compute normalized returns
    cond_vol = res.conditional_volatility.dropna()
    std_ret  = xret.loc[cond_vol.index] / cond_vol
    process_std_ret(std_ret)

    # mixture params: call for printing only
    fit_mixture(std_ret)

    # skew-t params: record in dict
    a, df, loc, scale = fit_skewt_params(std_ret)
    d.update({
        "skewt_alpha": a,
        "skewt_df":    df,
        "skewt_loc":   loc,
        "skewt_scale": scale,
    })

    model_results.append(d)

    # optional diagnostic plot
    if plot_gjr_garch:
        res.plot(annualize="D")
        plt.show()

# ----- GARCH(1,1) -----
if fit_garch:
    run_model("GARCH(1,1)")

# ----- GJR-GARCH(1,1,1) -----
if fit_gjr_garch:
    run_model("GJR-GARCH(1,1,1)", p=1, o=1, q=1)

# ----- APARCH(1,1,1) -----
if fit_aparch:
    run_model("APARCH(1,1,1)", vol="APARCH", p=1, o=1, q=1, power=1.0)

# ---------- final stats print ----------
print_stats_table(raw_stats_row, norm_stats_row)

if plot_norm_dist and std_ret_series is not None:
    plot_norm_kde(std_ret_series, log_ratio=True)

# ---------- compile results into DataFrame ----------
df_results = pd.DataFrame(model_results)
print("\nModel comparison (parameters + loglik + skew-t):")
print(df_results.to_string(index=False))
