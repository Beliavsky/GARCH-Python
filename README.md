# GARCH-Python
GARCH estimation using the [arch](https://github.com/bashtage/arch) package

`python xgarch_gjr_mean.py` gives
```
symbol = sp500

first and last dates:
1993-01-29    24.4525
2025-04-21   516.7299
                           Constant Mean - GARCH Model Results                           
=========================================================================================
Dep. Variable:                               SPY   R-squared:                       0.000
Mean Model:                        Constant Mean   Adj. R-squared:                  0.000
Vol Model:                                 GARCH   Log-Likelihood:               -10782.0
Distribution:      Standardized Skew Student's t   AIC:                           21576.0
Method:                       Maximum Likelihood   BIC:                           21618.1
                                                   No. Observations:                 8111
Date:                           Mon, Apr 21 2025   Df Residuals:                     8110
Time:                                   11:08:43   Df Model:                            1
                                 Mean Model                                 
============================================================================
                 coef    std err          t      P>|t|      95.0% Conf. Int.
----------------------------------------------------------------------------
mu             0.0601  8.534e-03      7.045  1.858e-12 [4.339e-02,7.684e-02]
                              Volatility Model                              
============================================================================
                 coef    std err          t      P>|t|      95.0% Conf. Int.
----------------------------------------------------------------------------
omega          0.0128  2.557e-03      5.001  5.703e-07 [7.775e-03,1.780e-02]
alpha[1]       0.1125  9.875e-03     11.390  4.673e-30   [9.313e-02,  0.132]
beta[1]        0.8830  9.782e-03     90.268      0.000     [  0.864,  0.902]
                                Distribution                               
===========================================================================
                 coef    std err          t      P>|t|     95.0% Conf. Int.
---------------------------------------------------------------------------
eta            6.4613      0.459     14.066  6.128e-45    [  5.561,  7.362]
lambda        -0.0955  1.398e-02     -6.831  8.456e-12 [ -0.123,-6.812e-02]
===========================================================================

Covariance estimator: robust


return statistics
series  #obs        mean         sd     Sharpe       skew       kurt        min        max
raw     8111      0.0327     1.1808     0.4395    -0.0074    11.7703   -10.9543    14.5078
norm    8111      0.0317     1.0026     0.5014    -0.5314     2.1002    -7.9006     4.2413

Volatility-bin statistics:
vol_bin  #obs   mean     sd  Sharpe    skew   kurt      min     max
0.0-1.0  4851 0.0247 0.7344  0.5349 -0.5599 2.0302  -4.1942  2.9320
1.0-2.0  2838 0.0331 1.3385  0.3925 -0.3243 1.8351  -7.2593  5.7957
  >=2.0   422 0.1213 2.9225  0.6591  0.2165 2.8707 -10.9543 14.5078
```
