import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor, TabPFNClassifier

# Preprocessing functions
from preprocessing.process import (
    msc, snv, derivative, savitzky_golay, airPLS
)

# ==============================================================
# Load TabPFN model
# ==============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR",
                      r"D:\workspace\TabPFN\tabpfn")

def load_local_tabpfn(kind="regressor", version="2.5", variant="real", device=device):
    """
    Load local TabPFN model (regressor or classifier).
    """
    cache = os.environ["TABPFN_MODEL_CACHE_DIR"]

    if version in ("2", "2.0"):
        fname = f"tabpfn-v2-{kind}.ckpt"
    elif version in ("2.5", "2_5"):
        var = (variant or "default").lower()
        fname = f"tabpfn-v2.5-{kind}-v2.5_{var}.ckpt"
    else:
        raise ValueError(f"Unsupported version: {version}")

    path = os.path.join(cache, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model_class = TabPFNRegressor if kind == "regressor" else TabPFNClassifier
    model = model_class(model_path=path, device=device, random_state=42)
    setattr(model, "ignore_pretraining_limits", True)

    return model

# ==============================================================
# Metrics
# ==============================================================

def calculate_sep(y_true, y_pred):
    n = len(y_true)
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - 1))

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)  

    return dict(
        r2 = r2_score(y_true, y_pred),
        mae = mean_absolute_error(y_true, y_pred),
        mse = mse,
        rmse = rmse,
        sep = calculate_sep(y_true, y_pred)
    )

# ==============================================================
# Preprocessing
# ==============================================================

def apply_preprocessing(X, method):
    if method == "none":
        return X
    elif method == "msc":
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
        raise ValueError(f"Unknown preprocessing method: {method}")

# ==============================================================
# Load dataset
# ==============================================================

data_path = "D:/A/CSU/NIRdatasets/wheat/Cal_ManufacturerA3.xlsx" 
#A1:Cal_ManufacturerA1.xlsx  A2:Cal_ManufacturerA2.xlsx  A3:Cal_ManufacturerA3.xlsx  A4:Test_ManufacturerA.xlsx
df = pd.read_excel(data_path)

spectra = df.iloc[:, 2:743].values
y = df.iloc[:, 1].values

preprocess_methods = ["none", "msc", "snv", "derivative", "sg", "airPLS"]

print("\nPreprocessing candidates:", preprocess_methods)

# ==============================================================
# Cross-validation to choose best preprocessing (with per-run final training)
# ==============================================================

rmse_list = []            # tracks test RMSE for each run
all_results = []          # collects final test metrics per run
cv_rows = []              # collects per-fold CV RMSE rows

# Keep last run's predictions for plotting at the end
last_run_train_pred = None
last_run_test_pred = None
last_run_best_method = None
last_run_y_train = None
last_run_y_test = None

for run in range(1, 6):
    print(f"\n\n==========================")
    print(f"   RUN {run} / 5")
    print(f"==========================\n")

    # Train-test split changes every run
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, y, test_size=0.2, random_state=run
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_method = None
    best_score = float('inf')  # looking for minimum RMSE

    print("\n=== Performing 5-fold CV to select best preprocessing ===")

    for method in preprocess_methods:
        cv_scores = []
        fold_id = 1  # initialize fold counter for this method

        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            # Apply preprocessing per fold
            X_tr_p = apply_preprocessing(X_tr, method)
            X_val_p = apply_preprocessing(X_val, method)

            # Train a TabPFN model on the current fold
            model = load_local_tabpfn(kind="regressor", version="2.5", variant="real")
            model.fit(X_tr_p, y_tr)
            preds = model.predict(X_val_p)

            # Compute RMSE for the validation fold
            fold_rmse = root_mean_squared_error(y_val, preds)
            cv_scores.append(fold_rmse)

            # Record fold result
            cv_rows.append({
                "run": run,
                "method": method,
                "fold": fold_id,
                "rmse": fold_rmse
            })
            fold_id += 1

        # Average CV RMSE for this preprocessing method
        method_avg = np.mean(cv_scores)
        print(f"Method: {method:12s} | CV RMSE = {method_avg:.4f}")
        if method_avg < best_score:
            best_score = method_avg
            best_method = method

    print("\n>>> Best preprocessing method:", best_method)
    print(">>> Best CV RMSE:", best_score)

    # ==============================================================
    # Train final model for this run using the best preprocessing
    # ==============================================================

    X_train_best = apply_preprocessing(X_train, best_method)
    X_test_best = apply_preprocessing(X_test, best_method)

    final_model = load_local_tabpfn(kind="regressor", version="2.5", variant="real")
    final_model.fit(X_train_best, y_train)

    y_train_pred = final_model.predict(X_train_best)
    y_test_pred = final_model.predict(X_test_best)

    train_metrics = evaluate(y_train, y_train_pred)
    test_metrics = evaluate(y_test, y_test_pred)

    print("\n=== Final Model Performance (this run) ===")
    print("Train Metrics:", train_metrics)
    print("Test  Metrics:", test_metrics)

    # Track per-run test RMSE and summary row
    rmse_list.append(test_metrics["rmse"])
    row = test_metrics.copy()
    row["best_preprocessing"] = best_method
    row["run"] = run
    all_results.append(row)

    # Save last run predictions and labels for plotting after the loop
    last_run_train_pred = y_train_pred
    last_run_test_pred = y_test_pred
    last_run_best_method = best_method
    last_run_y_train = y_train
    last_run_y_test = y_test

# ==============================================================
# Save all results to Excel (two sheets: final_test and cv_folds)
# ==============================================================

results_df = pd.DataFrame(all_results)
cv_df = pd.DataFrame(cv_rows)
save_path = "TabPFN_5runs_results.xlsx"
with pd.ExcelWriter(save_path) as writer:
    # Sheet 1: final test metrics per run
    results_df.to_excel(writer, sheet_name="final_test", index=False)
    # Sheet 2: 5-fold CV RMSEs per run and method
    cv_df.to_excel(writer, sheet_name="cv_folds", index=False)

print("\n===================================")
print("5 RUNS FINISHED. RESULTS SAVED TO:")
print(save_path)
print("===================================\n")

print("\nRMSE values across 5 runs:", rmse_list)

# ==============================================================
# Plot True vs Predicted (from the last run)
# ==============================================================

plt.figure(figsize=(7, 7))
plt.scatter(last_run_y_train, last_run_train_pred, alpha=0.6, label="Train")
plt.scatter(last_run_y_test, last_run_test_pred, alpha=0.8, label="Test")

min_v = min(y.min(), last_run_train_pred.min(), last_run_test_pred.min())
max_v = max(y.max(), last_run_train_pred.max(), last_run_test_pred.max())

plt.plot([min_v, max_v], [min_v, max_v], "k--", label="Ideal")
plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title(f"True vs Predicted (Best preprocessing = {last_run_best_method})")
plt.legend()
plt.grid(True)
plt.show()