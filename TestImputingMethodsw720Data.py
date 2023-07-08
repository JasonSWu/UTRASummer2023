import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from scipy.special import chebyt
from BeckersRixen import BeckersRixen
import numbers
import numpy as np
import matplotlib.pyplot as plt
import pylab 
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples


def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, rug=False,
           rug_length=0.05, rug_kwargs=None, **kwargs):
    """Draw a quantile-quantile plot for `x` versus `y`.

    Parameters
    ----------
    x, y : array-like
        One-dimensional numeric arrays.

    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. If not provided, the current axes will be used.

    quantiles : int or array-like, optional
        Quantiles to include in the plot. This can be an array of quantiles, in
        which case only the specified quantiles of `x` and `y` will be plotted.
        If this is an int `n`, then the quantiles will be `n` evenly spaced
        points between 0 and 1. If this is None, then `min(len(x), len(y))`
        evenly spaced quantiles between 0 and 1 will be computed.

    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        Specify the interpolation method used to find quantiles when `quantiles`
        is an int or None. See the documentation for numpy.quantile().

    rug : bool, optional
        If True, draw a rug plot representing both samples on the horizontal and
        vertical axes. If False, no rug plot is drawn.

    rug_length : float in [0, 1], optional
        Specifies the length of the rug plot lines as a fraction of the total
        vertical or horizontal length.

    rug_kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.axvline() and
        matplotlib.axes.Axes.axhline() when drawing rug plots.

    kwargs : dict of keyword arguments
        Keyword arguments to pass to matplotlib.axes.Axes.scatter() when drawing
        the q-q plot.
    """
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, method=interpolation)
    y_quantiles = np.quantile(y, quantiles, method=interpolation)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, **kwargs)

data = pd.read_csv('./EOF_720_log10.csv')

subset = data.iloc[[0, 3, 4, 5, 6, 7, 8], :].to_numpy()
subset2 = subset[:, ~np.isnan(subset).any(axis=0)]
subset2_rand = subset2[:, np.random.permutation(subset2.shape[1])]

M0 = subset2_rand.copy()
M0[0, 174:] = np.nan

# Initialize training and testing sets for RF algorithm
M0trainX = M0[1:7, :174].T
M0trainY = M0[0, :174]
M0predX = M0[1:7, 174:].T

# Run each imputation algorithm on each variable in the subset matrix

# Initialize diagnostics tables
rmses = np.zeros((subset2_rand.shape[0], 6))
skews = np.zeros((subset2_rand.shape[0], 6))
kurts = np.zeros((subset2_rand.shape[0], 6))
kl_divs = np.zeros((subset2_rand.shape[0], 6))
for i in range(7):
    trainsubset = np.delete(subset2_rand.copy(), i, axis=0)

    # Prepare M0 for input to KNN
    M0 = subset2_rand.copy()
    M0[i, 174:] = np.nan

    # Prepare M0 for KNN with Chebyshev Polynomials
    n = subset2_rand.shape[0] - 1 #6
    m = subset2_rand.shape[1]     #281
    M0Cheby = np.zeros((31, m))
    M0Cheby[:7, :] = subset2_rand.copy()
    M0Cheby[i, 174:] = np.nan

    M0Cheby[7:13, :] = chebyt(2)(trainsubset[:, :].flatten()).reshape((6, -1))
    M0Cheby[13:19, :] = chebyt(3)(trainsubset[:, :].flatten()).reshape((6, -1))
    M0Cheby[19:25, :] = chebyt(4)(trainsubset[:, :].flatten()).reshape((6, -1))
    M0Cheby[25:31, :] = chebyt(5)(trainsubset[:, :].flatten()).reshape((6, -1))

    # Prepare M0 for Beckers Rixen
    M02 = subset2_rand.copy()
    M02[i, 174:] = 0

    # Prepare M0 for Beckers Rixen with Chebyshev Polynomials
    M02Cheby = M0Cheby.copy()
    M02Cheby[i, 174:] = 0

    # Prepare M0 for RF
    M0trainX = trainsubset[:, :174]
    M0trainY = M0[i, :174]
    M0predX = trainsubset[:, 174:]

    # RF w/ Chebyshev Polynomials
    M0trainXCheby = np.zeros((30, 174))
    M0trainXCheby[:6, :] = trainsubset[:, :174]
    M0trainXCheby[6:12, :] = chebyt(2)(trainsubset[:, :174].flatten()).reshape((6, -1))
    M0trainXCheby[12:18, :] = chebyt(3)(trainsubset[:, :174].flatten()).reshape((6, -1))
    M0trainXCheby[18:24, :] = chebyt(4)(trainsubset[:, :174].flatten()).reshape((6, -1))
    M0trainXCheby[24:30, :] = chebyt(5)(trainsubset[:, :174].flatten()).reshape((6, -1))
    M0trainYCheby = M0[i,:174]
    M0predXCheby = np.zeros((30, trainsubset.shape[1] - 174))
    M0predXCheby[:6] = trainsubset[:, 174:]
    M0predXCheby[6:12, :] = chebyt(2)(trainsubset[:, 174:].flatten()).reshape((6, -1))
    M0predXCheby[12:18, :] = chebyt(3)(trainsubset[:, 174:].flatten()).reshape((6, -1))
    M0predXCheby[18:24, :] = chebyt(4)(trainsubset[:, 174:].flatten()).reshape((6, -1))
    M0predXCheby[24:30, :] = chebyt(5)(trainsubset[:, 174:].flatten()).reshape((6, -1))

    # Run imputation algorithms and calculate diagnostics

    # KNN
    k=4
    imputer = KNNImputer(n_neighbors=k)
    M0imputed_knn = imputer.fit_transform(M0.T).T
    M0imputed_knn_cheby = imputer.fit_transform(M0Cheby.T).T

    # Beckers-Rixen
    M0imputed_br, U1, S1, V1 = BeckersRixen(M02)
    M0imputed_br_cheby, U2, S2, V2 = BeckersRixen(M02Cheby)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(M0trainX.T, M0trainY)
    M0imputed_rf = rf.predict(M0predX.T).T
    rf_cheby = RandomForestRegressor(n_estimators=100)
    rf_cheby.fit(M0trainXCheby.T, M0trainY)
    M0imputed_rf_cheby = rf_cheby.predict(M0predXCheby.T).T

    # Calculate diagnostics
    target_vals = subset2_rand[i, 174:]
    
    rmses[i, 0] = np.sqrt(np.mean((target_vals - M0imputed_knn[i, 174:]) ** 2))
    rmses[i, 1] = np.sqrt(np.mean((target_vals - M0imputed_knn_cheby[i, 174:]) ** 2))
    rmses[i, 2] = np.sqrt(np.mean((target_vals - M0imputed_br[i, 174:]) ** 2))
    rmses[i, 3] = np.sqrt(np.mean((target_vals - M0imputed_br_cheby[i, 174:]) ** 2))
    rmses[i, 4] = np.sqrt(np.mean((target_vals - M0imputed_rf) ** 2))
    rmses[i, 5] = np.sqrt(np.mean((target_vals - M0imputed_rf_cheby) ** 2))

    skews[i, 0] = pd.Series(M0imputed_knn[i, 174:]).skew()
    skews[i, 1] = pd.Series(M0imputed_knn_cheby[i, 174:]).skew()
    skews[i, 2] = pd.Series(M0imputed_br[i, 174:]).skew()
    skews[i, 3] = pd.Series(M0imputed_br_cheby[i, 174:]).skew()
    skews[i, 4] = pd.Series(M0imputed_rf).skew()
    skews[i, 5] = pd.Series(M0imputed_rf_cheby).skew()

    kurts[i, 0] = pd.Series(M0imputed_knn[i, 174:]).kurtosis()
    kurts[i, 1] = pd.Series(M0imputed_knn_cheby[i, 174:]).kurtosis()
    kurts[i, 2] = pd.Series(M0imputed_br[i, 174:]).kurtosis()
    kurts[i, 3] = pd.Series(M0imputed_br_cheby[i, 174:]).kurtosis()
    kurts[i, 4] = pd.Series(M0imputed_rf).kurtosis()
    kurts[i, 5] = pd.Series(M0imputed_rf_cheby).kurtosis()

    pp_x = sm.ProbPlot(target_vals)
    pp_y = sm.ProbPlot(M0imputed_br[i, 174:])
    qqplot_2samples(pp_x, pp_y, line="q") #line = "45", "s", "r", "q"
    plt.show()

    # QQ Plots for each method
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    qqplot(target_vals, M0imputed_knn[i, 174:], ax=axs[0, 0], interpolation='linear')
    axs[0, 0].plot(target_vals, target_vals, color='red')
    axs[0, 0].set_title('Imputing with KNN (k=4)')

    qqplot(target_vals, M0imputed_br[i, 174:], ax=axs[0, 1])
    axs[0, 1].plot(target_vals, target_vals, color='red')
    axs[0, 1].set_title('Imputing with BeckersRixen')

    qqplot(target_vals, M0imputed_rf, ax=axs[0, 2])
    axs[0, 2].plot(target_vals, target_vals, color='red')
    axs[0, 2].set_title('Imputing with Random Forest')

    qqplot(target_vals, M0imputed_knn_cheby[i, 174:], ax=axs[1, 0])
    axs[1, 0].plot(target_vals, target_vals, color='red')
    axs[1, 0].set_title('KNN (k=4) & Chebyshev')

    qqplot(target_vals, M0imputed_br_cheby[i, 174:], ax=axs[1, 1])
    axs[1, 1].plot(target_vals, target_vals, color='red')
    axs[1, 1].set_title('BeckersRixen & Chebyshev')

    qqplot(target_vals, M0imputed_rf_cheby, ax=axs[1, 2])
    axs[1, 2].plot(target_vals, target_vals, color='red')
    axs[1, 2].set_title('RF w/ Chebyshev')

    plt.tight_layout()
    plt.show()






