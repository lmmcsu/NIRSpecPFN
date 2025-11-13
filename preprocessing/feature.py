import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from scipy.linalg import qr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from scipy.stats import ttest_1samp
from sklearn.feature_selection import f_regression

# ---- RFE----
def rfe(x_train, y_train, x_test=None, n_features_to_select=None):
        n_features_to_select = n_features_to_select or 100
        estimator = RandomForestRegressor(n_estimators=10, random_state=42)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(x_train, y_train)
        x_train_rfe = rfe.transform(x_train)
        if x_test is not None:
            x_test_rfe = rfe.transform(x_test)
            return x_train_rfe, x_test_rfe
        return x_train_rfe


# ---- SPA----
def _qr_proj(Z, k, M):
    Z = Z.copy()
    Z[:, k] *= 2 * np.max(np.sum(Z**2, axis=0)) / (np.sum(Z[:, k]**2) + 1e-12)
    _, _, order = qr(Z, mode='economic', pivoting=True)
    return order[:M]

def _loocv_press(X, y, idx):
    idx = np.asarray(idx, int)
    E = np.empty_like(y)
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        A = np.c_[np.ones(mask.sum()), X[mask][:, idx]]
        b = np.linalg.lstsq(A, y[mask], rcond=None)[0]
        E[i] = y[i] - np.hstack([1., X[i][idx]]) @ b
    return E.T @ E

def spa(x_train, y_train, x_test, m_max=None):
    N, K = x_train.shape
    m_max = min(N-1, K) if m_max is None else m_max


    μ, σ = x_train.mean(axis=0), x_train.std(axis=0, ddof=1) + 1e-12
    Z_train = (x_train - μ) / σ

    best_press, best_vars = np.inf, None
    for k in range(K):
        path = _qr_proj(Z_train, k, m_max)
        for m in range(1, m_max+1):
            press = _loocv_press(x_train, y_train, path[:m]) 
            if press < best_press:
                best_press, best_vars = press, path[:m]

    return x_train[:, best_vars], x_test[:, best_vars]

#---- UVE----
def uve(x_train, y_train, x_test, count=None):
    count = count or 100
    pls = PLSRegression(n_components=min(10, x_train.shape[1], x_train.shape[0]))
    y_pred = cross_val_predict(pls, x_train, y_train, cv=len(y_train))
    res = y_train - y_pred

    stability = np.abs(np.mean(x_train * res[:, None], axis=0)) / (
        np.std(x_train * res[:, None], axis=0) + 1e-12)

    relevance = np.abs([ttest_1samp(x_train[:, i] * res, 0)[0]
                        for i in range(x_train.shape[1])])
    relevance[np.isnan(relevance)] = 0

    scores = stability * relevance
    top_idx = np.argsort(scores)[-count:]

    return x_train[:, top_idx], x_test[:, top_idx]

#---- Univariate----
def univariate(x_train, y_train, x_test):
    f_vals, _ = f_regression(x_train, y_train)
    count = max(1, int(0.6 * x_train.shape[1]))        
    top_idx = np.argsort(f_vals)[-count:]               
    return x_train[:, top_idx], x_test[:, top_idx]