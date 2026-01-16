import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import os
import sys

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # current directory
project_root = os.path.dirname(current_dir)               # relative directory
sys.path.insert(0, project_root)                         # Add to search path

# Preprocessing functions
from preprocessing.process import (msc, snv, derivative, savitzky_golay, airPLS)

# ==============================================================
# Metrics
# ==============================================================

def calculate_sep(y_true, y_pred):
    """Calculate SEP (standard error of prediction)."""
    n = len(y_true)
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - 1))

def evaluate(y_true, y_pred):
    """Return a dictionary containing evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return dict(
        r2=r2_score(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        mse=mse,
        rmse=rmse,
        sep=calculate_sep(y_true, y_pred)
    )


# ==============================================================
# Preprocessing
# ==============================================================

def apply_preprocessing(X, method):
    """Apply selected preprocessing method to spectral data."""
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


# ==============================================================
# Load Dataset
# ==============================================================

data_path = "D:/A/CSU/NIRdatasets/corn_xlsl/m5_corn.xlsx"
df = pd.read_excel(data_path)

spectra = df.iloc[:, :700].values
y = df.iloc[:, 701].values
# Moisture:y = df.iloc[:, 700].values
# Protein:y = df.iloc[:, 701].values
# Oil:y = df.iloc[:, 702].values
# Starch:y = df.iloc[:, 703].values

preprocess_methods = ["msc", "snv", "derivative", "sg", "airPLS"]
print("\nPreprocessing candidates:", preprocess_methods)

# ==============================================================
# Run 5 Repeated Experiments
# ==============================================================

rmse_list = []
all_results = []
cv_rows = []

last_run_train_pred = None
last_run_test_pred = None
last_run_best_method = None
last_run_y_train = None
last_run_y_test = None

for run in range(1, 5):
    print(f"\n\n==========================")
    print(f"   RUN {run} / 45")
    print(f"==========================\n")

    # Split train/test with different seeds
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, y, test_size=0.2, random_state=run
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_method = None
    best_score = float('inf')

    print("\n=== Performing 5-fold CV to select best preprocessing ===")

    # Try each preprocessing method
    for method in preprocess_methods:
        cv_scores = []
        fold_id = 1

        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            # Apply preprocessing
            X_tr_p = apply_preprocessing(X_tr, method)
            X_val_p = apply_preprocessing(X_val, method)

            # Grid search for n_components
            param_grid = {"n_components": [5, 10, 15, 20]}
            pls = PLSRegression()

            grid = GridSearchCV(pls, param_grid, cv=3,
                                scoring="neg_mean_squared_error")
            grid.fit(X_tr_p, y_tr)

            best_pls = grid.best_estimator_
            preds = best_pls.predict(X_val_p).ravel()

            fold_rmse = root_mean_squared_error(y_val, preds)
            cv_scores.append(fold_rmse)

            # Save CV results
            cv_rows.append({
                "run": run,
                "method": method,
                "fold": fold_id,
                "rmse": fold_rmse
            })
            fold_id += 1

        method_avg = np.mean(cv_scores)
        print(f"Method: {method:12s} | CV RMSE = {method_avg:.4f}")

        if method_avg < best_score:
            best_score = method_avg
            best_method = method

    print("\n>>> Best preprocessing method:", best_method)
    print(">>> Best CV RMSE:", best_score)

    # ==============================================================
    # Train Final Model
    # ==============================================================

    X_train_best = apply_preprocessing(X_train, best_method)
    X_test_best = apply_preprocessing(X_test, best_method)

    param_grid = {"n_components": [5, 10, 15, 20]}
    grid = GridSearchCV(
        PLSRegression(),
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    grid.fit(X_train_best, y_train)
    final_model = grid.best_estimator_

    print("\n>>> Final PLSR best n_components =", grid.best_params_["n_components"])

    y_train_pred = final_model.predict(X_train_best).ravel()
    y_test_pred = final_model.predict(X_test_best).ravel()

    train_metrics = evaluate(y_train, y_train_pred)
    test_metrics = evaluate(y_test, y_test_pred)

    print("\n=== Final Model Performance (this run) ===")
    print("Train:", train_metrics)
    print("Test :", test_metrics)

    rmse_list.append(test_metrics["rmse"])

    row = test_metrics.copy()
    row["best_preprocessing"] = best_method
    row["run"] = run
    all_results.append(row)

    last_run_train_pred = y_train_pred
    last_run_test_pred = y_test_pred
    last_run_best_method = best_method
    last_run_y_train = y_train
    last_run_y_test = y_test


# ==============================================================
# Save Results
# ==============================================================

results_df = pd.DataFrame(all_results)
cv_df = pd.DataFrame(cv_rows)

out_xlsx = "PLSR_5runs_results.xlsx"
with pd.ExcelWriter(out_xlsx) as writer:
    results_df.to_excel(writer, sheet_name="final_test", index=False)
    cv_df.to_excel(writer, sheet_name="cv_folds", index=False)

print("\n===================================")
print("PLSR 5 RUNS FINISHED. RESULTS SAVED TO:")
print(out_xlsx)
print("===================================\n")

print("\nRMSE values across 5 runs:", rmse_list)


# ==============================================================
# Plot Last Run (True vs Predicted)
# ==============================================================

plt.figure(figsize=(7, 7))
plt.scatter(last_run_y_train, last_run_train_pred, alpha=0.6, label="Train")
plt.scatter(last_run_y_test, last_run_test_pred, alpha=0.75, label="Test")

min_v = min(y.min(), last_run_train_pred.min(), last_run_test_pred.min())
max_v = max(y.max(), last_run_train_pred.max(), last_run_test_pred.max())

plt.plot([min_v, max_v], [min_v, max_v], "k--", label="Ideal")

plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title(f"PLSR — True vs Predicted (Best preprocessing = {last_run_best_method})")
plt.legend()
plt.grid(True)
plt.show()



