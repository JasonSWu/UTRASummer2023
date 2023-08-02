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

def ImputingMethods(plot=False):
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
        KNN_mat = subset2_rand.copy()
        KNN_mat[i, 174:] = np.nan

        # Prepare M0 for Beckers Rixen
        BR_mat = subset2_rand.copy()
        BR_mat[i, 174:] = 0
        BR_train = BR_mat[:, :174]
        BR_test = BR_mat[:, 174:]

        # Prepare M0 for Beckers Rixen with Chebyshev Polynomials
        n = subset2_rand.shape[0] - 1 #6
        m = subset2_rand.shape[1]     #281
        BRCheby_mat = np.zeros((31, m))
        BRCheby_mat[:7, :] = subset2_rand.copy()
        BRCheby_mat[i, 174:] = 0

        BRCheby_mat[7:13, :] = chebyt(2)(trainsubset.flatten()).reshape((6, -1))
        BRCheby_mat[13:19, :] = chebyt(3)(trainsubset.flatten()).reshape((6, -1))
        BRCheby_mat[19:25, :] = chebyt(4)(trainsubset.flatten()).reshape((6, -1))
        BRCheby_mat[25:31, :] = chebyt(5)(trainsubset.flatten()).reshape((6, -1))
        BRCheby_train = BRCheby_mat[:, :174]
        BRCheby_test = BRCheby_mat[:, 174:]

        # Prepare M0 for RF
        RFtrainX = trainsubset[:, :174]
        RFtrainY = KNN_mat[i, :174]
        RFpredX = trainsubset[:, 174:]

        # KNN
        k=4
        imputer = KNNImputer(n_neighbors=k, weights='distance')
        M0imputed_knn = imputer.fit_transform(KNN_mat.T).T

        # Beckers-Rixen
        M0imputed_br = BeckersRixen(BR_train, BR_test)
        M0imputed_br_cheby = BeckersRixen(BRCheby_train, BRCheby_test)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(RFtrainX.T, RFtrainY)
        M0imputed_rf = rf.predict(RFpredX.T).T

        # Calculate diagnostics
        target_vals = subset2_rand[i, 174:]
        
        rmses[i, 0] = np.sqrt(np.mean((target_vals - M0imputed_knn[i, 174:]) ** 2))
        rmses[i, 1] = np.sqrt(np.mean((target_vals - M0imputed_br[i, 174:]) ** 2))
        rmses[i, 2] = np.sqrt(np.mean((target_vals - M0imputed_br_cheby[i, 174:]) ** 2))
        rmses[i, 3] = np.sqrt(np.mean((target_vals - M0imputed_rf) ** 2))

        maes[i, 0] = np.mean(np.abs(target_vals - M0imputed_knn[i, 174:]))
        maes[i, 1] = np.mean(np.abs(target_vals - M0imputed_br[i, 174:]))
        maes[i, 2] = np.mean(np.abs(target_vals - M0imputed_br_cheby[i, 174:]))
        maes[i, 3] = np.mean(np.abs(target_vals - M0imputed_rf))

        pd_knn = pd.Series(M0imputed_knn[i, 174:])
        pd_br = pd.Series(M0imputed_br[i, 174:])
        pd_br_cheby = pd.Series(M0imputed_br_cheby[i, 174:])
        pd_rf = pd.Series(M0imputed_rf)

        pd_target = pd.Series(target_vals)

        r2[i, 0] = pd_knn.corr(pd_target)
        r2[i, 1] = pd_br.corr(pd_target)
        r2[i, 2] = pd_br_cheby.corr(pd_target)
        r2[i, 3] = pd_rf.corr(pd_target)

        skews[i, 0] = pd_knn.skew()
        skews[i, 1] = pd_br.skew()
        skews[i, 2] = pd_br_cheby.skew()
        skews[i, 3] = pd_rf.skew()

        kurts[i, 0] = pd_knn.kurtosis()
        kurts[i, 1] = pd_br.kurtosis()
        kurts[i, 2] = pd_br_cheby.kurtosis()
        kurts[i, 3] = pd_rf.kurtosis()

        line_mode = "45"
        
        # QQ Plots for each method
        if plot:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))

            qqplot(target_vals, M0imputed_knn[i, 174:], ax=axs[0, 0], line=line_mode)
            axs[0, 0].set_title('Imputing with KNN (k=4)')

            qqplot(target_vals, M0imputed_br[i, 174:], ax=axs[0, 1], line=line_mode)
            axs[0, 1].set_title('Imputing with BeckersRixen')

            qqplot(target_vals, M0imputed_rf, ax=axs[1, 0], line=line_mode)
            axs[1, 0].set_title('Imputing with Random Forest')

            qqplot(target_vals, M0imputed_br_cheby[i, 174:], ax=axs[1, 1], line=line_mode)
            axs[1, 1].set_title('BeckersRixen & Chebyshev')

            plt.tight_layout()
            plt.show()

    # Performance metrics per method
    avg_rms = np.mean(rmses, axis=0)
    avg_r2 = np.mean(r2, axis=0)
    avg_mae = np.mean(maes, axis=0)

    method_headers = np.array(['KNN', 'Beckers Rixen', 'BR-Cheby', 'Random Forest'])

    #idx_sorted = np.argsort(avg_rms)
    #print(idx_sorted)
    #avg_rms = avg_rms[idx_sorted]
    #avg_r2 = avg_r2[idx_sorted]
    #avg_mae = avg_mae[idx_sorted]
    #method_headers = method_headers[idx_sorted]

    #print(
    #    tabulate(
    #        [['avg RMS'] + avg_rms.tolist(), ['avg r2'] + avg_r2.tolist(), ['avg MAE'] + avg_mae.tolist()],
    #        headers = method_headers,
    #        tablefmt='fancy_grid'))
    bare_table = np.array([avg_rms, avg_r2, avg_mae])
    return bare_table

n_runs = 100
accumulated_table = ImputingMethods(False)
for _ in range(n_runs):
    accumulated_table += ImputingMethods()
accumulated_table /= n_runs + 1

for header, row in zip(["rms", "r2", "mae"], accumulated_table):
    print(f"{header}: {row}")
