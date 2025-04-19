"""Load S&P 500 data, fit GARCH or GJR‑GARCH models, compare ACFs of
squared returns and normalised returns, and optionally plot the KDE of
normalised returns."""
import numpy as np
import arch.data.sp500
from scipy.stats import skew, kurtosis, norm, gaussian_kde
from arch import arch_model
import matplotlib.pyplot as plt

# -------- User‑set toggles ---------
plot_gjr_garch = False   # plot fitted GJR‑GARCH volatility etc.
plot_norm_dist = True   # plot KDE of normalised returns
fit_garch      = False
fit_gjr_garch  = True    # fit the GJR‑GARCH(1,1,1)
max_lag        = 30      # lags for the ACF table (0 → skip ACFs)
# -----------------------------------

# ---------- Load data & raw stats ----------
data   = arch.data.sp500.load()
market = data["Adj Close"]
print("symbol = sp500")
print("\nfirst and last dates:\n" + market.iloc[[0, -1]].to_string())

xret = 100 * market.pct_change().dropna()          # daily % returns
raw_stats = (
    len(xret),
    np.mean(xret),  np.std(xret),
    skew(xret),     kurtosis(xret),
    np.min(xret),   np.max(xret),
)

# ---------- helpers ----------
def _acf(series: np.ndarray, k: int) -> float:
    return np.corrcoef(series[k:], series[:-k])[0, 1]

def _print_acf_table(sq_raw, sq_std, lags):
    print("Lag    ACF(ret^2) ACF(norm_ret^2)")
    for k in lags:
        print(f"{k:3d}    {sq_raw[k-1]:10.4f}     {sq_std[k-1]:10.4f}")
    print()

def _stats_table(raw, norm=None):
    lbls = ["mean", "sd", "skew", "kurt", "min", "max"]
    header = "series  #obs  " + " ".join(f"{l:>10}" for l in lbls)
    print("\nreturn statistics")
    print(header)
    print(f"raw    {raw[0]:5d} ", *[f"{v:10.4f}" for v in raw[1:]])
    if norm is not None:
        print(f"norm   {norm[0]:5d} ", *[f"{v:10.4f}" for v in norm[1:]])
    print()

def _plot_norm_kde(series):
    """Kernel‑density estimate of normalised returns with N(0,1) overlay."""
    kde = gaussian_kde(series)
    x   = np.linspace(-4, 4, 501)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, kde(x), lw=1.5, label="KDE (norm. returns)")
    ax.plot(x, norm.pdf(x), lw=1.5, ls="--", label="N(0,1)")
    ax.set_title("Kernel‑density estimate of normalised returns")
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    plt.show()

norm_stats = None
std_ret    = None
acf_done   = False

# ---------- GARCH(1,1) ----------
if fit_garch:
    res = arch_model(xret).fit(update_freq=0, disp="off")
    print(res.summary(), end="\n\n")

    cond_vol = res.conditional_volatility.dropna()
    std_ret  = xret.loc[cond_vol.index] / cond_vol
    norm_stats = (
        len(std_ret),
        np.mean(std_ret),  np.std(std_ret),
        skew(std_ret),     kurtosis(std_ret),
        np.min(std_ret),   np.max(std_ret),
    )

    if max_lag > 0 and not acf_done:
        sq_raw = (xret.loc[cond_vol.index]**2).to_numpy()
        sq_std = (std_ret**2).to_numpy()
        _print_acf_table(
            [_acf(sq_raw, k) for k in range(1, max_lag + 1)],
            [_acf(sq_std, k) for k in range(1, max_lag + 1)],
            range(1, max_lag + 1),
        )
        acf_done = True

# ---------- GJR‑GARCH(1,1,1) ----------
if fit_gjr_garch:
    res = arch_model(xret, p=1, o=1, q=1).fit(update_freq=0, disp="off")
    print(res.summary())

    cond_vol = res.conditional_volatility.dropna()
    std_ret  = xret.loc[cond_vol.index] / cond_vol
    if norm_stats is None:
        norm_stats = (
            len(std_ret),
            np.mean(std_ret),  np.std(std_ret),
            skew(std_ret),     kurtosis(std_ret),
            np.min(std_ret),   np.max(std_ret),
        )

    if max_lag > 0 and not acf_done:
        sq_raw = (xret.loc[cond_vol.index]**2).to_numpy()
        sq_std = (std_ret**2).to_numpy()
        _print_acf_table(
            [_acf(sq_raw, k) for k in range(1, max_lag + 1)],
            [_acf(sq_std, k) for k in range(1, max_lag + 1)],
            range(1, max_lag + 1),
        )
        acf_done = True

    if plot_gjr_garch:
        res.plot(annualize="D")
        plt.show()

# ---------- final stats & optional plot ----------
_stats_table(raw_stats, norm_stats)

if plot_norm_dist and std_ret is not None:
    _plot_norm_kde(std_ret)
