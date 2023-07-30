import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from scipy.special import chebyt
from BeckersRixen import BeckersRixen
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples
from tabulate import tabulate


def qqplot(x, y, ax=None, line="q"):
    pp_x = sm.ProbPlot(x)
    pp_y = sm.ProbPlot(y)
    qqplot_2samples(pp_x, pp_y, ax=ax, line=line) #line = "45", "s", "r", "q"

data = pd.read_csv('./EOF_720_log10.csv')

def ImputingMethods():
    subset = data.iloc[[0, 3, 4, 5, 6, 7, 8], :].to_numpy()
    subset2 = subset[:, ~np.isnan(subset).any(axis=0)]
    subset2_rand = subset2[:, np.random.permutation(subset2.shape[1])]

    # Run each imputation algorithm on each variable in the subset matrix

    # Initialize diagnostics tables, 4 methods of imputation
    num_methods = 4
    rmses = np.zeros((subset2_rand.shape[0], num_methods))
    skews = np.zeros((subset2_rand.shape[0], num_methods))
    kurts = np.zeros((subset2_rand.shape[0], num_methods))
    r2 = np.zeros((subset2_rand.shape[0], num_methods))
    maes = np.zeros((subset2_rand.shape[0], num_methods))
    kl_divs = np.zeros((subset2_rand.shape[0], num_methods))
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
        '''M0trainXCheby = np.zeros((30, 174))
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
        M0predXCheby[24:30, :] = chebyt(5)(trainsubset[:, 174:].flatten()).reshape((6, -1))'''

        # Run imputation algorithms and calculate diagnostics

        # KNN
        k=4
        imputer = KNNImputer(n_neighbors=k, weights='distance')
        M0imputed_knn = imputer.fit_transform(M0.T).T
        #M0imputed_knn_cheby = imputer.fit_transform(M0Cheby.T).T

        # Beckers-Rixen
        M0imputed_br, U1, S1, V1 = BeckersRixen(M02, M02)
        M0imputed_br_cheby, U2, S2, V2 = BeckersRixen(M02Cheby, M02Cheby)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(M0trainX.T, M0trainY)
        M0imputed_rf = rf.predict(M0predX.T).T
        '''rf_cheby = RandomForestRegressor(n_estimators=100)
        rf_cheby.fit(M0trainXCheby.T, M0trainY)
        M0imputed_rf_cheby = rf_cheby.predict(M0predXCheby.T).T'''

        # Calculate diagnostics
        target_vals = subset2_rand[i, 174:]
        
        rmses[i, 0] = np.sqrt(np.mean((target_vals - M0imputed_knn[i, 174:]) ** 2))
        #rmses[i, 1] = np.sqrt(np.mean((target_vals - M0imputed_knn_cheby[i, 174:]) ** 2))
        rmses[i, 1] = np.sqrt(np.mean((target_vals - M0imputed_br[i, 174:]) ** 2))
        rmses[i, 2] = np.sqrt(np.mean((target_vals - M0imputed_br_cheby[i, 174:]) ** 2))
        rmses[i, 3] = np.sqrt(np.mean((target_vals - M0imputed_rf) ** 2))
        #rmses[i, 5] = np.sqrt(np.mean((target_vals - M0imputed_rf_cheby) ** 2))

        maes[i, 0] = np.mean(np.abs(target_vals - M0imputed_knn[i, 174:]))
        #maes[i, 1] = np.mean(np.abs(target_vals - M0imputed_knn_cheby[i, 174:]))
        maes[i, 1] = np.mean(np.abs(target_vals - M0imputed_br[i, 174:]))
        maes[i, 2] = np.mean(np.abs(target_vals - M0imputed_br_cheby[i, 174:]))
        maes[i, 3] = np.mean(np.abs(target_vals - M0imputed_rf))
        #maes[i, 5] = np.mean(np.abs(target_vals - M0imputed_rf_cheby))

        pd_knn = pd.Series(M0imputed_knn[i, 174:])
        #pd_knn_cheby = pd.Series(M0imputed_knn_cheby[i, 174:])
        pd_br = pd.Series(M0imputed_br[i, 174:])
        pd_br_cheby = pd.Series(M0imputed_br_cheby[i, 174:])
        pd_rf = pd.Series(M0imputed_rf)
        #pd_rf_cheby = pd.Series(M0imputed_rf_cheby)

        pd_target = pd.Series(target_vals)

        r2[i, 0] = pd_knn.corr(pd_target)
        #r2[i, 1] = pd_knn_cheby.corr(pd_target)
        r2[i, 1] = pd_br.corr(pd_target)
        r2[i, 2] = pd_br_cheby.corr(pd_target)
        r2[i, 3] = pd_rf.corr(pd_target)
        #r2[i, 5] = pd_rf_cheby.corr(pd_target)

        skews[i, 0] = pd_knn.skew()
        #skews[i, 1] = pd_knn_cheby.skew()
        skews[i, 1] = pd_br.skew()
        skews[i, 2] = pd_br_cheby.skew()
        skews[i, 3] = pd_rf.skew()
        #skews[i, 5] = pd_rf_cheby.skew()

        kurts[i, 0] = pd_knn.kurtosis()
        #kurts[i, 1] = pd_knn_cheby.kurtosis()
        kurts[i, 1] = pd_br.kurtosis()
        kurts[i, 2] = pd_br_cheby.kurtosis()
        kurts[i, 3] = pd_rf.kurtosis()
        #kurts[i, 5] = pd_rf_cheby.kurtosis()

        line_mode = "45"

        # QQ Plots for each method
        #-fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        #-qqplot(target_vals, M0imputed_knn[i, 174:], ax=axs[0, 0], line=line_mode)
        #-axs[0, 0].set_title('Imputing with KNN (k=4)')

        #-qqplot(target_vals, M0imputed_br[i, 174:], ax=axs[0, 1], line=line_mode)
        #-axs[0, 1].set_title('Imputing with BeckersRixen')

        #-qqplot(target_vals, M0imputed_rf, ax=axs[1, 0], line=line_mode)
        #-axs[1, 0].set_title('Imputing with Random Forest')

        #qqplot(target_vals, M0imputed_knn_cheby[i, 174:], ax=axs[1, 0], line=line_mode)
        #axs[1, 0].set_title('KNN (k=4) & Chebyshev')

        #-qqplot(target_vals, M0imputed_br_cheby[i, 174:], ax=axs[1, 1], line=line_mode)
        #-axs[1, 1].set_title('BeckersRixen & Chebyshev')

        #qqplot(target_vals, M0imputed_rf_cheby, ax=axs[1, 2], line=line_mode)
        #axs[1, 2].set_title('RF w/ Chebyshev')

        #plt.tight_layout()
        #plt.show()

    '''wo_br_idx = np.array([0,2,4])
    w_br_idx = np.array([1,3,5])
    # Performance metrics with Cheby
    stats_without_cheby = [
        np.mean(skews[:,wo_br_idx]), np.mean(kurts[:,wo_br_idx]), np.mean(rmses[:,wo_br_idx]), np.mean(r2[:,wo_br_idx]), np.mean(maes[:,wo_br_idx])]
    # Performance metrics without Cheby
    stats_with_cheby = [
        np.mean(skews[:,w_br_idx]), np.mean(kurts[:,w_br_idx]), np.mean(rmses[:,w_br_idx]), np.mean(r2[:,w_br_idx]), np.mean(maes[:,w_br_idx])]

    cheby_headers = ['', 'skew', 'kurtosis', 'RMS', 'r2', 'MAE']

    print(tabulate([['w/o Cheby'] + stats_without_cheby, ['w/ Cheby'] + stats_with_cheby], headers = cheby_headers, tablefmt='fancy_grid'))'''

    # Performance metrics per method
    avg_rms = np.mean(rmses, axis=0)
    avg_r2 = np.mean(r2, axis=0)
    avg_mae = np.mean(maes, axis=0)

    #methods_headers = ['', 'KNN', 'KNN Cheby', 'Beckers Rixen', 'BR-Cheby', 'Random Forest', 'RF Cheby']
    method_headers = np.array(['KNN', 'Beckers Rixen', 'BR-Cheby', 'Random Forest'])

    idx_sorted = np.argsort(avg_rms)
    #print(idx_sorted)
    avg_rms = avg_rms[idx_sorted]
    avg_r2 = avg_r2[idx_sorted]
    avg_mae = avg_mae[idx_sorted]
    method_headers = method_headers[idx_sorted]

    #print(
    #    tabulate(
    #        [['avg RMS'] + avg_rms.tolist(), ['avg r2'] + avg_r2.tolist(), ['avg MAE'] + avg_mae.tolist()],
    #        headers = method_headers,
    #        tablefmt='fancy_grid'))
    bare_table = np.array([rmses, r2, maes])
    return bare_table

n_runs = int(input())
accumulated_table = ImputingMethods()
for _ in range(n_runs):
    accumulated_table += ImputingMethods()
accumulated_table /= n_runs + 1

#print(accumulated_table)


