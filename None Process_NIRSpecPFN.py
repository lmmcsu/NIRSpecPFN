import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor

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

    model = TabPFNRegressor(model_path=path, device=device, random_state=42)
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
# Load dataset
# ==============================================================

data_path = "D:/A/CSU/NIRdatasets/wheat/Test_ManufacturerA.xlsx"
df = pd.read_excel(data_path)

spectra = df.iloc[:, 2:743].values
y = df.iloc[:, 1].values

print("\nRunning TabPFN without any preprocessing...")

# wheat
# data_path = pd.read_csv("D:/A/CSU/NIRdatasets/wheat/Test_ManufacturerA.csv")
# A1:wheat/Cal_ManufacturerA1.xlsx  A2:wheat/Cal_ManufacturerA2.xlsx  A3:wheat/Cal_ManufacturerA3.xlsx  A4:wheat/Test_ManufacturerA.xlsx
# spectra = df.iloc[:, 2:743].values
# y = df.iloc[:, 1].values

# corn
# data_path = "D:/A/CSU/NIRdatasets/corn_xlsl/m5_corn.xlsx"
# spectra = df.iloc[:, :700].values
# y = df.iloc[:, 701].values
# Moisture:y = df.iloc[:, 700].values
# Protein:y = df.iloc[:, 701].values
# Oil:y = df.iloc[:, 702].values
# Starch:y = df.iloc[:, 703].values

# ==============================================================
# Run 5 repetitions with different train/test splits
# ==============================================================

rmse_list = []
all_results = []

last_y_train = None
last_y_test = None
last_train_pred = None
last_test_pred = None

for run in range(1, 6):
    print(f"\n================ RUN {run} / 5 ================")

    # Random split changes every run
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, y, test_size=0.2, random_state=run
    )

    # Train TabPFN without any preprocessing
    model = load_local_tabpfn(kind="regressor", version="2.5", variant="real")
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_metrics = evaluate(y_train, y_train_pred)
    test_metrics = evaluate(y_test, y_test_pred)

    print("\nTrain Metrics:", train_metrics)
    print("Test  Metrics:", test_metrics)

    rmse_list.append(test_metrics["rmse"])

    row = test_metrics.copy()
    row["run"] = run
    all_results.append(row)

    # Save last run predictions for plotting
    last_y_train = y_train
    last_y_test = y_test
    last_train_pred = y_train_pred
    last_test_pred = y_test_pred

# ==============================================================
# Save results to Excel
# ==============================================================

results_df = pd.DataFrame(all_results)
save_path = "TabPFN_no_preprocessing_5runs.xlsx"
results_df.to_excel(save_path, index=False)

print("\n===================================")
print("FINISHED. RESULTS SAVED TO:")
print(save_path)
print("===================================\n")

print("\nRMSE values across 5 runs:", rmse_list)

# ==============================================================
# Plot True vs Predicted (last run)
# ==============================================================

plt.figure(figsize=(7, 7))
plt.scatter(last_y_train, last_train_pred, alpha=0.6, label="Train")
plt.scatter(last_y_test, last_test_pred, alpha=0.8, label="Test")

min_v = min(y.min(), last_train_pred.min(), last_test_pred.min())
max_v = max(y.max(), last_train_pred.max(), last_test_pred.max())

plt.plot([min_v, max_v], [min_v, max_v], "k--", label="Ideal")
plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title("True vs Predicted (No Preprocessing)")
plt.legend()
plt.grid(True)
plt.show()
