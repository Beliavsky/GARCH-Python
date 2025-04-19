"""Load S&P 500 data, fit GARCH or GJR-GARCH models, compare ACFs of
squared returns and normalised returns, fit a 2-component normal
mixture to the normalised returns, and optionally plot KDE + mixture
density + standard-normal density."""
import numpy as np
import arch.data.sp500
from scipy.stats import skew, kurtosis, norm, gaussian_kde
from arch import arch_model
import matplotlib.pyplot as plt
import sys

try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# -------- User‑set toggles ---------
plot_gjr_garch = False   # plot fitted GJR-GARCH volatility etc.
plot_norm_dist = True    # plot KDE + mixture + N(0,1) densities
fit_garch      = False
fit_gjr_garch  = True    # fit the GJR-GARCH(1,1,1)
max_lag        = 30      # lags for the ACF table (0 -> skip ACFs)
# -----------------------------------

# ---------- Load data ----------
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

def _plot_norm_kde(series, gmm=None):
    """Plot KDE, mixture density, and N(0,1) for normalised returns."""
    kde = gaussian_kde(series)
    x   = np.linspace(-5, 5, 601)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, kde(x), lw=1.5, label="KDE (norm. returns)")
    if gmm is not None:
        ax.plot(x, np.exp(gmm.score_samples(x.reshape(-1, 1))),
                lw=1.5, ls="-.", label="2-comp mixture")
    ax.plot(x, norm.pdf(x), lw=1.5, ls="--", label="N(0,1)")
    ax.set_title("Densities of normalised returns")
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    plt.show()

# ---------- containers / flags ----------
norm_stats = None
std_ret    = None
acf_done   = False
mix_done   = False
mix_model  = None          # will store fitted GaussianMixture

# ---------- shared post‑processing ----------
def _process_std_ret(std_series):
    """Compute stats, ACFs, and mixture (only first call)."""
    global norm_stats, std_ret, acf_done, mix_done, mix_model

    if norm_stats is None:
        norm_stats = (
            len(std_series),
            np.mean(std_series),  np.std(std_series),
            skew(std_series),     kurtosis(std_series),
            np.min(std_series),   np.max(std_series),
        )
    std_ret = std_series            # keep for optional plotting

    if max_lag > 0 and not acf_done:
        sq_raw = (xret.loc[std_series.index]**2).to_numpy()
        sq_std = (std_series**2).to_numpy()
        _print_acf_table(
            [_acf(sq_raw, k) for k in range(1, max_lag + 1)],
            [_acf(sq_std, k) for k in range(1, max_lag + 1)],
            range(1, max_lag + 1),
        )
        acf_done = True

    if not mix_done:
        if SKLEARN_OK:
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
        else:
            print("\nscikit-learn not available -> mixture fit skipped")
        mix_done = True

# ----- GARCH(1,1) -----
if fit_garch:
    res = arch_model(xret).fit(update_freq=0, disp="off")
    print(res.summary(), end="\n\n")
    cond_vol = res.conditional_volatility.dropna()
    _process_std_ret(xret.loc[cond_vol.index] / cond_vol)

# ----- GJR-GARCH(1,1,1) -----
if fit_gjr_garch:
    res = arch_model(xret, p=1, o=1, q=1).fit(update_freq=0, disp="off")
    print(res.summary())
    cond_vol = res.conditional_volatility.dropna()
    _process_std_ret(xret.loc[cond_vol.index] / cond_vol)

    if plot_gjr_garch:
        res.plot(annualize="D")
        plt.show()

# ---------- final output ----------
_stats_table(raw_stats, norm_stats)

if plot_norm_dist and std_ret is not None:
    _plot_norm_kde(std_ret, mix_model)
