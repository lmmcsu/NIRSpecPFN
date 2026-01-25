import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor
import sys

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # current directory
project_root = os.path.dirname(current_dir)               # relative directory
sys.path.insert(0, project_root)                         # Add to search path
# Load processing function
from preprocessing.process import (msc, snv, derivative, savitzky_golay, airPLS)
plt.rcParams.update({'font.size': 14})


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def apply_preprocessing(X, method):
    if method == "msc":
        return msc(X)
    elif method == "snv":
        return snv(X)
    elif method == "derivative":
        return derivative(X)
    elif method == "sg":
        return savitzky_golay(X)
    elif method == "airPLS":
        return airPLS(X)
    else:
        raise ValueError(f"Unknown preprocessing: {method}")


PRE_LIST = ["msc", "snv", "derivative", "sg", "airPLS"]

# ==============================================================
# Load TabPFN
# ==============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", r"D:\workspace\TabPFN\tabpfn")

def load_tabpfn():
    fname = "tabpfn-v2.5-regressor-v2.5_real.ckpt"
    model_path = os.path.join(os.environ["TABPFN_MODEL_CACHE_DIR"], fname)
    model = TabPFNRegressor(model_path=model_path, device=device, random_state=42)
    setattr(model, "ignore_pretraining_limits", True)
    return model


# ==============================================================
# Step 1: Select best preprocessing for a model (on training subset)
# ==============================================================

def select_best_preprocessing(model_name, X_train, y_train):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_method = None
    best_score = np.inf

    for method in PRE_LIST:
        X_p = apply_preprocessing(X_train, method)
        cv_rmse = []

        for tr_idx, val_idx in kf.split(X_p):
            X_tr, X_val = X_p[tr_idx], X_p[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            # TabPFN
            if model_name == "TabPFN":
                model = load_tabpfn()
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)

            # PLSR
            elif model_name == "PLSR":
                grid = GridSearchCV(
                    PLSRegression(),
                    {"n_components": [5, 10, 15, 20]},
                    cv=3, scoring="neg_mean_squared_error"
                )
                grid.fit(X_tr, y_tr)
                pred = grid.best_estimator_.predict(X_val).ravel()

            # SVR
            elif model_name == "SVR":
                grid = GridSearchCV(
                    SVR(),
                    {"C": [100, 200, 300, 400], "epsilon": [0.01, 0.03, 0.05, 0.1]},
                    cv=3, scoring="neg_mean_squared_error"
                )
                grid.fit(X_tr, y_tr)
                pred = grid.best_estimator_.predict(X_val)

            cv_rmse.append(rmse(y_val, pred))

        avg_rmse = np.mean(cv_rmse)
        if avg_rmse < best_score:
            best_score = avg_rmse
            best_method = method

    return best_method


# ==============================================================
# Step 2: Train each model properly with small sample subset
# ==============================================================

def run_single_model_learning_curve(model_name, X_train, y_train, X_test, y_test,
                                    train_sizes=[10, 20, 30, 40, 50, 60], n_repeat=5):

    results = []

    for n in train_sizes:
        print(f"\n=== {model_name} — Train size {n} ===")

        n_rmse = []

        for _ in range(n_repeat):
            idx = np.random.choice(len(X_train), n, replace=False)
            X_sub, y_sub = X_train[idx], y_train[idx]

            # Step 1: find best preprocessing for THIS subset
            best_method = select_best_preprocessing(model_name, X_sub, y_sub)

            # Step 2: re-apply preprocessing
            X_sub_p = apply_preprocessing(X_sub, best_method)
            X_test_p = apply_preprocessing(X_test, best_method)

            # Step 3: train final model
            if model_name == "TabPFN":
                model = load_tabpfn()
                model.fit(X_sub_p, y_sub)
                pred = model.predict(X_test_p)

            elif model_name == "PLSR":
                grid = GridSearchCV(
                    PLSRegression(),
                    {"n_components": [5, 10, 15, 20]},
                    cv=5, scoring="neg_mean_squared_error"
                )
                grid.fit(X_sub_p, y_sub)
                pred = grid.best_estimator_.predict(X_test_p).ravel()

            elif model_name == "SVR":
                grid = GridSearchCV(
                    SVR(),
                    {"C": [100, 200, 300, 400], "epsilon": [0.01, 0.03, 0.05, 0.1]},
                    cv=5, scoring="neg_mean_squared_error"
                )
                grid.fit(X_sub_p, y_sub)
                pred = grid.best_estimator_.predict(X_test_p)

            n_rmse.append(rmse(y_test, pred))

        results.append(n_rmse)

    return np.array(results)


# ==============================================================
# Load data
# ==============================================================

df = pd.read_excel("D:/A/CSU/NIRdatasets/wheat/Cal_ManufacturerA3.xlsx")
spectra = df.iloc[:, 2:743].values
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(spectra, y, test_size=0.2, random_state=42)

# ==============================================================
# Run Learning Curves
# ==============================================================

train_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
n_repeat = 5

print("\nRunning TabPFN...")
tab_r = run_single_model_learning_curve("TabPFN", X_train, y_train, X_test, y_test,
                                        train_sizes, n_repeat)

print("\nRunning PLSR...")
pls_r = run_single_model_learning_curve("PLSR", X_train, y_train, X_test, y_test,
                                        train_sizes, n_repeat)

print("\nRunning SVR...")
svr_r = run_single_model_learning_curve("SVR", X_train, y_train, X_test, y_test,
                                        train_sizes, n_repeat)


# ==============================================================
# Compute Mean + 95% CI
# ==============================================================

def mean_ci(arr):
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    ci = 1.96 * std / np.sqrt(arr.shape[1])
    return mean, ci

tab_mean, tab_ci = mean_ci(tab_r)
pls_mean, pls_ci = mean_ci(pls_r)
svr_mean, svr_ci = mean_ci(svr_r)


# ==============================================================
# Plot
# ==============================================================

plt.figure(figsize=(10, 6))

plt.plot(train_sizes, tab_mean, marker="o", label="NIRSpecPFN")
plt.fill_between(train_sizes, tab_mean - tab_ci, tab_mean + tab_ci, alpha=0.2)

plt.plot(train_sizes, pls_mean, marker="s", label="PLSR")
plt.fill_between(train_sizes, pls_mean - pls_ci, pls_mean + pls_ci, alpha=0.2)

plt.plot(train_sizes, svr_mean, marker="^", label="SVR")
plt.fill_between(train_sizes, svr_mean - svr_ci, svr_mean + svr_ci, alpha=0.2)

plt.xlabel("Training Sample Size")
plt.ylabel("RMSEP")
plt.grid(True, alpha=0.4)
plt.legend()
plt.title("Learning Curve")
plt.tight_layout()
plt.show()




