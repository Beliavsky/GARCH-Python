"""Load S&P 500 data, fit GARCH / GJR-GARCH, compare ACFs of squared
returns and normalised returns, fit a 2-component normal mixture, and
(optionally) plot KDE + mixture + N(0,1) densities. Also bin by model-
implied volatility and store raw-return stats in a DataFrame."""
from __future__ import annotations
import numpy as np
import pandas as pd
import arch.data.sp500
from scipy.stats import skew, kurtosis
from arch import arch_model
import matplotlib.pyplot as plt

from stats import acf, print_acf_table, stats_table, plot_norm_kde

# -------- User-set toggles ---------
pd.set_option('display.float_format', '{:.4f}'.format)
plot_gjr_garch      = False     # plot fitted GJR-GARCH volatility, etc.
plot_norm_dist      = False     # draw KDE + mixture + N(0,1) densities
fit_garch           = True
fit_gjr_garch       = False     # fit the GJR-GARCH(1,1,1)
max_lag             = 0         # lags for the ACF table (0 -> skip ACFs)
dist                = "skewt"
rf_rate             = 0.03
days_year           = 252.0
bin_width           = 1.0       # width of volatility bins for return stats
max_vol_threshold   = 2.0       # largest vol threshold: last bin is >= this
# -----------------------------------

# ---------- Load data ----------
data   = arch.data.sp500.load()
market = data["Adj Close"]
print("symbol = sp500")
print("\nfirst and last dates:\n" + market.iloc[[0, -1]].to_string())

xret = 100 * (market.pct_change().dropna() - rf_rate/days_year)  # daily % returns
raw_stats_row = (
    len(xret),
    np.mean(xret),  np.std(xret),
    np.sqrt(days_year)*np.mean(xret)/np.std(xret),
    skew(xret),     kurtosis(xret),
    np.min(xret),   np.max(xret),
)

# ---------- containers / flags ----------
norm_stats_row = None
std_ret_series = None
acf_done       = False

# ---------- shared post-processing ----------
def process_std_ret(std_series):
    """Compute stats, ACFs for normalized returns"""
    global norm_stats_row, std_ret_series, acf_done

    if norm_stats_row is None:
        norm_stats_row = (
            len(std_series),
            np.mean(std_series),  np.std(std_series),
            np.sqrt(days_year)*np.mean(std_series)/np.std(std_series),
            skew(std_series),     kurtosis(std_series),
            np.min(std_series),   np.max(std_series),
        )
    std_ret_series = std_series  # keep for optional plotting

    if max_lag > 0 and not acf_done:
        sq_raw = (xret.loc[std_series.index]**2).to_numpy()
        sq_std = (std_series**2).to_numpy()
        print_acf_table(
            [acf(sq_raw, k) for k in range(1, max_lag + 1)],
            [acf(sq_std, k) for k in range(1, max_lag + 1)],
            range(1, max_lag + 1),
        )
        acf_done = True

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
stats_table(raw_stats_row, norm_stats_row)

if plot_norm_dist and std_ret_series is not None:
    plot_norm_kde(std_ret_series, log_ratio=True)

# --- Bin by conditional volatility and collect stats ---
df_vol = pd.DataFrame({
    "ret": xret.loc[cond_vol.index],
    "vol": cond_vol
})

if bin_width is not None:
    # Define bin edges [0, 0.5, 1.0, …, max_vol_threshold] plus ∞
    edges = np.arange(0, max_vol_threshold + bin_width, bin_width)
    edges = list(edges) + [np.inf]

    # Labels: "0.0-0.5", …, ">=2.0"
    labels = [
        f"{edges[i]:.1f}-{edges[i+1]:.1f}"
        for i in range(len(edges)-2)
    ] + [f">={max_vol_threshold:.1f}"]

    df_vol["vol_bin"] = pd.cut(df_vol["vol"], bins=edges, labels=labels, right=False)

    # Gather stats into list of dicts
    stats_list = []
    for bin_label, grp in df_vol.groupby("vol_bin"):
        r = grp["ret"]
        n = len(r)
        m = r.mean()
        s = r.std(ddof=1)
        sharpe = np.sqrt(days_year) * m / s if s != 0 else np.nan
        sk = skew(r)
        kt = kurtosis(r)
        mn = r.min()
        mx = r.max()
        stats_list.append({
            "vol_bin":  str(bin_label),
            "n_obs":    n,
            "mean":     m,
            "sd":       s,
            "Sharpe":   sharpe,
            "skew":     sk,
            "kurtosis": kt,
            "min":      mn,
            "max":      mx,
        })

    # Build DataFrame of volatility-bin statistics
    df_vol_stats = pd.DataFrame(stats_list)

    # Display or further process as needed
    print("\nVolatility-bin statistics:")
    print(df_vol_stats.to_string(index=False))
