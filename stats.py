"""Generic helpers for simple time-series diagnostics and plotting."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, skew, kurtosis

# --- optional skew-t import ------------------------------------------
try:
    from scipy.stats import skewt           # SciPy >= 1.11
    SKEWT_OK = True
except ImportError:
    SKEWT_OK = False

def acf(series: np.ndarray, k: int) -> float:
    """  lag-k autocorrelation """
    if k <= 0 or k >= series.size:
        raise ValueError("k must be in {1, …, len(series)‑1}")
    return np.corrcoef(series[k:], series[:-k])[0, 1]

def print_acf_table(acf_raw: list[float], acf_std: list[float], lags):
    """ print autocorrelations stored in acf_raw and acf_std """
    print("Lag    ACF(ret^2) ACF(norm_ret^2)")
    for k in lags:
        print(f"{k:3d}    {acf_raw[k-1]:10.4f}     {acf_std[k-1]:10.4f}")
    print()

def return_stats(series: np.ndarray,
    days_year: float = 252.0) -> tuple[int, float, float,
    float, float, float, float, float]:
    """ Compute return-series statistics: (nobs, mean, sd,
    annualized Sharpe, skew, kurtosis, min, max) """
    n      = series.size
    m      = series.mean()
    s      = series.std(ddof=0)
    sharpe = np.sqrt(days_year) * m / s if s != 0 else np.nan
    sk     = skew(series)
    kt     = kurtosis(series)
    mn     = series.min()
    mx     = series.max()
    return (n, m, s, sharpe, sk, kt, mn, mx)

def print_stats_table(row_raw: tuple,
    row_norm: tuple | None = None):
    """ print summary stats stored in row_raw and row_norm """
    labels = ["mean", "sd", "Sharpe", "skew", "kurt", "min", "max"]
    header = "series  #obs  " + " ".join(f"{lab:>10}" for lab in labels)
    print("\nreturn statistics")
    print(header)
    print(f"raw    {row_raw[0]:5d} ", *[f"{v:10.4f}" for v in row_raw[1:]])
    if row_norm is not None:
        print(f"norm   {row_norm[0]:5d} ", *[f"{v:10.4f}" for v in row_norm[1:]])
    print()

def plot_norm_kde(series, gmm=None, title=
    "Densities of normalised returns", log_ratio=False, eps=1e-12):
    """ Plot KDE of the data, a fitted 2-component mixture (if given),
    a standard-normal PDF, and (when available) a fitted skew‑t PDF.
    With log_ratio=True a second panel shows log-ratios versus KDE.
    """
    kde = gaussian_kde(series)
    x   = np.linspace(-5, 5, 601)

    # ---------------- figure layout -----------------
    if log_ratio:
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(6, 5.8), sharex=True,
            gridspec_kw=dict(height_ratios=[2, 1], hspace=0.25)
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 3.7))
        ax2 = None

    # ---------------- top panel: densities ----------
    ax.plot(x, kde(x), lw=1.5, label="KDE (norm. returns)")

    dens_mix = None
    if gmm is not None:
        dens_mix = np.exp(gmm.score_samples(x.reshape(-1, 1)))
        ax.plot(x, dens_mix, lw=1.4, ls="-.", label="2-comp mixture")

    # standard normal
    dens_norm = norm.pdf(x)
    ax.plot(x, dens_norm, lw=1.4, ls="--", label="N(0,1)")

    # skewed Student‑t (if SciPy provides it)
    dens_skewt = None
    if SKEWT_OK:
        try:
            a, df, loc, scale = skewt.fit(series)
            dens_skewt = skewt.pdf(x, a, df, loc, scale)
            ax.plot(x, dens_skewt, lw=1.4, ls=":", label="skew‑t")
        except Exception:
            dens_skewt = None  # silently skip on failure

    ax.set_title(title)
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---------------- bottom panel: log‑ratios -------
    if log_ratio and ax2 is not None:
        dens_kde = kde(x)

        if dens_mix is not None:
            ax2.plot(
                x, np.log((dens_kde + eps)/(dens_mix + eps)),
                lw=1.1, label="log[KDE / mixture]"
            )

        ax2.plot(
            x, np.log((dens_kde + eps)/(dens_norm + eps)),
            lw=1.1, ls="--", label="log[KDE / N(0,1)]"
        )

        if dens_skewt is not None:
            ax2.plot(
                x, np.log((dens_kde + eps)/(dens_skewt + eps)),
                lw=1.1, ls=":", label="log[KDE / skew‑t]"
            )

        ax2.set_xlabel("value")
        ax2.set_ylabel("log ratio")
        ax2.legend()
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()

def vol_bin_stats(returns: pd.Series, cond_vol: pd.Series,
    vol_bin_width: float, max_vol_threshold: float,
    days_year: float = 252.0) -> pd.DataFrame:
    """Compute return-stats in bins of conditional volatility."""
    # align returns and volatility
    df_vol = pd.DataFrame({
        "ret": returns,
        "vol": cond_vol
    })

    # create bin edges: [0, w, 2w, ..., max] and one final +inf bin
    edges = np.arange(0, max_vol_threshold + vol_bin_width, vol_bin_width)
    edges = list(edges) + [np.inf]

    # labels for each bin except the last
    labels = [
        f"{edges[i]:.1f}-{edges[i+1]:.1f}"
        for i in range(len(edges) - 2)
    ] + [f">={max_vol_threshold:.1f}"]

    df_vol["vol_bin"] = pd.cut(
        df_vol["vol"],
        bins=edges,
        labels=labels,
        right=False
    )

    # aggregate stats per bin
    stats_list: list[dict] = []
    for bin_label, grp in df_vol.groupby("vol_bin"):
        r = grp["ret"].to_numpy()
        n, m, s, sharpe, sk, kt, mn, mx = return_stats(r, days_year)
        stats_list.append({
            "vol_bin": str(bin_label),
            "#obs": n,
            "mean": m,
            "sd": s,
            "Sharpe": sharpe,
            "skew": sk,
            "kurt": kt,
            "min": mn,
            "max": mx
        })
    return pd.DataFrame(stats_list)
