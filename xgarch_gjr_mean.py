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
from pandas_util import read_csv_date_index
from stats import (acf, print_acf_table, print_stats_table, plot_norm_kde,
    return_stats, vol_bin_stats)

# -------- User-set toggles ---------
pd.set_option('display.float_format', '{:.4f}'.format)
plot_gjr_garch      = False
plot_norm_dist      = False
fit_garch           = True
fit_gjr_garch       = False
max_lag             = 0
dist                = "skewt"
rf_rate             = 0.03
days_year           = 252.0
vol_bin_width       = 1.0
max_vol_threshold   = 2.0
prices_file         = "spy.csv"
# -----------------------------------

# ---------- Load data ----------
if prices_file is not None:
    market = read_csv_date_index(prices_file)["SPY"]
else:
    data   = arch.data.sp500.load()
    market = data["Adj Close"]

print("symbol = sp500")
print("\nfirst and last dates:\n" + market.iloc[[0, -1]].to_string())

xret = 100 * (market.pct_change().dropna() - rf_rate/days_year)

# Compute raw-return stats via helper
raw_stats_row = return_stats(xret.to_numpy(), days_year)

# ---------- containers / flags ----------
norm_stats_row = None
std_ret_series = None
acf_done       = False

# ---------- shared post-processing ----------
def process_std_ret(std_series: pd.Series):
    global norm_stats_row, std_ret_series, acf_done

    if norm_stats_row is None:
        norm_stats_row = return_stats(std_series.to_numpy(), days_year)
    std_ret_series = std_series

    if max_lag > 0 and not acf_done:
        sq_raw = (xret.loc[std_series.index]**2).to_numpy()
        sq_std = (std_series**2).to_numpy()
        print_acf_table(
            [acf(sq_raw, k) for k in range(1, max_lag + 1)],
            [acf(sq_std, k) for k in range(1, max_lag + 1)],
            range(1, max_lag + 1),
        )
        acf_done = True

cond_vol = None

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
    plot_norm_kde(std_ret_series, log_ratio=True)

# ---------- volatility‐bin statistics ----------
if vol_bin_width is not None and cond_vol is not None:
    df_vol_stats = vol_bin_stats(xret.loc[cond_vol.index], cond_vol,
        vol_bin_width, max_vol_threshold, days_year)
    print("Volatility-bin statistics:\n" + df_vol_stats.to_string(
        index=False))
