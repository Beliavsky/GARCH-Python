""" Loads S&P 500 data, computes return statistics, fits GARCH and/or
GJR-GARCH models, and plots GJR-GARCH results. """
import numpy as np
import arch.data.sp500
from scipy.stats import skew, kurtosis
from arch import arch_model
import matplotlib.pyplot as plt

plot_gjr_garch = False
fit_garch = False
fit_gjr_garch = True
data = arch.data.sp500.load()
market = data["Adj Close"]
print("symbol = sp500")
print("\nfirst and last dates:\n" + market.iloc[[0, -1]].to_string())
returns = 100 * market.pct_change().dropna()
print("\nreturn stats:\n#obs", "".join("%11s"%label for label in
    ["mean", "sd", "skew", "kurt", "min", "max"]))
print("%5d"%len(returns), *("%10.4f"%stat for stat in [np.mean(returns),
    np.std(returns), skew(returns), kurtosis(returns),
    np.min(returns), np.max(returns)]), end="\n\n")
if fit_garch:
    am = arch_model(returns)
    res = am.fit(update_freq=0, disp="off")
    print(res.summary(), end="\n\n")
if fit_gjr_garch:
    am = arch_model(returns, p=1, o=1, q=1)
    res = am.fit(update_freq=0, disp="off")
    print(res.summary())
    if plot_gjr_garch:
        fig = res.plot(annualize="D")
        plt.show()
