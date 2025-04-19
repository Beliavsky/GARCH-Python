"""Generic helpers for simple time‑series diagnostics and plotting."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

# --- optional skew-t import ------------------------------------------
try:
    from scipy.stats import skewt           # SciPy >= 1.11
    SKEWT_OK = True
except ImportError:
    SKEWT_OK = False
# ---------------------------------------------------------------------


# ------------------------------------------------------------------
# 1.  Basic (unbiased) lag-k autocorrelation
# ------------------------------------------------------------------
def acf(series: np.ndarray, k: int) -> float:
    if k <= 0 or k >= series.size:
        raise ValueError("k must be in {1, …, len(series)‑1}")
    return np.corrcoef(series[k:], series[:-k])[0, 1]


# ------------------------------------------------------------------
# 2.  Convenience printer for two ACF sequences
# ------------------------------------------------------------------
def print_acf_table(acf_raw: list[float], acf_std: list[float], lags):
    print("Lag    ACF(ret^2) ACF(norm_ret^2)")
    for k in lags:
        print(f"{k:3d}    {acf_raw[k-1]:10.4f}     {acf_std[k-1]:10.4f}")
    print()


# ------------------------------------------------------------------
# 3.  Summary‑statistics table
# ------------------------------------------------------------------
def stats_table(row_raw: tuple, row_norm: tuple | None = None):
    labels = ["mean", "sd", "skew", "kurt", "min", "max"]
    header = "series  #obs  " + " ".join(f"{lab:>10}" for lab in labels)
    print("\nreturn statistics")
    print(header)
    print(f"raw    {row_raw[0]:5d} ", *[f"{v:10.4f}" for v in row_raw[1:]])
    if row_norm is not None:
        print(f"norm   {row_norm[0]:5d} ", *[f"{v:10.4f}" for v in row_norm[1:]])
    print()


# ------------------------------------------------------------------
# 4.  Density plot: KDE, mixture, standard normal, skew-t
# ------------------------------------------------------------------
def plot_norm_kde(
    series,
    gmm=None,
    title="Densities of normalised returns",
    log_ratio=False,
    eps=1e-12,
):
    """
    Plot KDE of the data, a fitted 2‑component mixture (if given),
    a standard‑normal PDF, and (when available) a fitted skew‑t PDF.
    With log_ratio=True a second panel shows log‑ratios versus KDE.
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
