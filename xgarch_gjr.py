"""Load S&P 500 data, fit GARCH / GJR-GARCH, compare ACFs of squared
returns and normalised returns, fit a 2-component normal mixture, and
(optionally) plot KDE + mixture + N(0,1) densities."""
from __future__ import annotations
import numpy as np
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

# -------- User‑set toggles ---------
plot_gjr_garch = False     # plot fitted GJR-GARCH volatility, etc.
plot_norm_dist = True      # draw KDE + mixture + N(0,1) densities
fit_garch      = False
fit_gjr_garch  = True      # fit the GJR-GARCH(1,1,1)
max_lag        = 30        # lags for the ACF table (0 -> skip ACFs)
dist           = "skewt"
# -----------------------------------

# ---------- Load data ----------
data   = arch.data.sp500.load()
market = data["Adj Close"]
print("symbol = sp500")
print("\nfirst and last dates:\n" + market.iloc[[0, -1]].to_string())

xret = 100 * market.pct_change().dropna()          # daily % returns
raw_stats_row = (
    len(xret),
    np.mean(xret),  np.std(xret),
    skew(xret),     kurtosis(xret),
    np.min(xret),   np.max(xret),
)

# ---------- containers / flags ----------
norm_stats_row = None
std_ret_series = None
acf_done       = False
mix_model      = None      # will store fitted GaussianMixture

# ---------- shared post‑processing ----------
def process_std_ret(std_series):
    """Compute stats, ACFs, and mixture fit."""
    global norm_stats_row, std_ret_series, acf_done, mix_model

    if norm_stats_row is None:
        norm_stats_row = (
            len(std_series),
            np.mean(std_series),  np.std(std_series),
            skew(std_series),     kurtosis(std_series),
            np.min(std_series),   np.max(std_series),
        )
    std_ret_series = std_series            # keep for optional plotting

    if max_lag > 0 and not acf_done:
        sq_raw = (xret.loc[std_series.index]**2).to_numpy()
        sq_std = (std_series**2).to_numpy()
        print_acf_table(
            [acf(sq_raw, k) for k in range(1, max_lag + 1)],
            [acf(sq_std, k) for k in range(1, max_lag + 1)],
            range(1, max_lag + 1),
        )
        acf_done = True

    if mix_model is None and SKLEARN_OK:
        gmm = GaussianMixture(n_components=2, covariance_type="full",
                              random_state=0)
        gmm.fit(std_series.values.reshape(-1, 1))
        mix_model = gmm

        w   = gmm.weights_
        mu  = gmm.means_.flatten()
        sig = np.sqrt(gmm.covariances_.flatten())

        print("\n2-component normal mixture fitted to normalised returns")
        print("comp   weight     mean       sd")
        for i in range(2):
            print(f"{i+1:3d} {w[i]:10.4f} {mu[i]:10.4f} {sig[i]:10.4f}")
    elif mix_model is None:
        print("\nscikit-learn not available -> mixture fit skipped")

# ----- GARCH(1,1) -----
if fit_garch:
    res = arch_model(xret, dist=dist).fit(update_freq=0, disp="off")
    print(res.summary(), end="\n\n")
    cond_vol = res.conditional_volatility.dropna()
    process_std_ret(xret.loc[cond_vol.index] / cond_vol)

# ----- GJR-GARCH(1,1,1) -----
if fit_gjr_garch:
    res = arch_model(xret, p=1, o=1, q=1, dist=dist).fit(update_freq=0, disp="off")
    print(res.summary())
    cond_vol = res.conditional_volatility.dropna()
    process_std_ret(xret.loc[cond_vol.index] / cond_vol)

    if plot_gjr_garch:
        res.plot(annualize="D")
        plt.show()

# ---------- final output ----------
print_stats_table(raw_stats_row, norm_stats_row)

if plot_norm_dist and std_ret_series is not None:
    plot_norm_kde(std_ret_series, gmm=mix_model, log_ratio=True)
