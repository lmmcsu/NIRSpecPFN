import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from preprocessing.process import (derivative, savitzky_golay, airPLS, msc, snv, detrend)

# ============================================================
# 1. Load data
# ============================================================
def load_xy_from_excel(path):
    """Load spectral matrix X, reference y, and sample IDs from Excel file."""
    df = pd.read_excel(path)

    id_col = "ID"
    y_col = "Protein"

    wavelength_cols = [c for c in df.columns if c not in [id_col, y_col]]

    X = df[wavelength_cols].to_numpy(float)
    y = df[y_col].to_numpy(float)
    IDs = df[id_col].to_numpy()

    return X, y, IDs, wavelength_cols


# ============================================================
# 2. Preprocessing
# ============================================================
def preprocess(X, method="sg"):
    """Apply spectral preprocessing."""
    if method == "none":
        return X
    if method == "sg":
        return savitzky_golay(X)
    if method == "snv":
        return snv(X)
    if method == "msc":
        return msc(X)
    if method == "detrend":
        return detrend(X)
    if method == "derivative":
        return derivative(X)
    if method == "airpls":
        return airPLS(X)
    raise ValueError("Unknown preprocessing method.")


# ============================================================
# 3. Select N paired samples
# ============================================================
def select_paired_samples(ids, N=50, random_state=0):
    """Randomly select N paired samples based on their IDs."""
    np.random.seed(random_state)
    n = len(ids)
    if N > n:
        N = n
    idx = np.random.choice(n, size=N, replace=False)
    return idx


# ============================================================
# 4. Direct Standardization (global DS)
# ============================================================
def ds_transfer(master_X, slave_X, pair_idx):
    """
    Direct Standardization (DS).

    Solve the global transformation matrix T:
        slave_X_pair * T ≈ master_X_pair

    Then apply:
        slave_trans = slave_X * T
    """
    Xm = master_X[pair_idx]   # N × wl
    Xs = slave_X[pair_idx]    # N × wl

    # Solve the global matrix T using least squares
    # T shape: (wl_slave × wl_master)
    T, _, _, _ = np.linalg.lstsq(Xs, Xm, rcond=None)

    # Apply DS transformation
    slave_trans = slave_X @ T
    return slave_trans


# ============================================================
# 5. Evaluation
# ============================================================
def evaluate(y, pred):
    """Compute RMSE and R²."""
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    return rmse, r2


# ============================================================
# main
# ============================================================
master_path = r"D:\A\CSU\NIRdatasets\wheat\Test_ManufacturerA.xlsx"
slave_path  = r"D:\A\CSU\NIRdatasets\wheat\Cal_ManufacturerA3.xlsx"
# masterA4:Test_ManufacturerA
# slaveA1:Cal_ManufacturerA1  slaveA2:Cal_ManufacturerA2  slaveA3:Cal_ManufacturerA3

# Load both instruments
master_X_raw, master_y, master_ID, wl = load_xy_from_excel(master_path)
slave_X_raw,  slave_y, slave_ID,  _   = load_xy_from_excel(slave_path)

# Check ID alignment
assert np.array_equal(master_ID, slave_ID), "ID mismatch detected!"
print("ID match confirmed.\n")

# ============================================================
# Preprocessing
# ============================================================
method = "sg"
master_X = preprocess(master_X_raw, method)
slave_X  = preprocess(slave_X_raw, method)

# ============================================================
# Paired samples for DS
# ============================================================
N = 60   # Number of paired samples
pair_idx = select_paired_samples(slave_ID, N=N)
print("Using paired samples:", len(pair_idx))

# ============================================================
# Baseline (no DS)
# ============================================================
pls = PLSRegression(n_components=10)
pls.fit(master_X, master_y)
pred_no = pls.predict(slave_X).ravel()
rmse_no, r2_no = evaluate(slave_y, pred_no)

print("\nNO-DS RMSE:", rmse_no)
print("NO-DS R²  :", r2_no)

# ============================================================
# Apply Direct Standardization (DS)
# ============================================================
slave_trans = ds_transfer(master_X, slave_X, pair_idx)

pls_ds = PLSRegression(n_components=10)
pls_ds.fit(master_X, master_y)

pred_ds = pls_ds.predict(slave_trans).ravel()
rmse_ds, r2_ds = evaluate(slave_y, pred_ds)

# ============================================================
# Results
# ============================================================
print("\n========== DS RESULTS ==========")
print("DS RMSE =", rmse_ds)
print("DS R²   =", r2_ds)
print("================================\n")


# ============================================================
# Plot: Predict vs True (NO-DS)
# ============================================================
plt.figure(figsize=(6, 6))
plt.scatter(slave_y, pred_no, alpha=0.7)
plt.plot([min(slave_y), max(slave_y)],
         [min(slave_y), max(slave_y)],
         linestyle='--', linewidth=1)

plt.xlabel("True Protein")
plt.ylabel("Predicted Protein")
plt.title(f"Predict vs True (NO-DS)\nRMSE={rmse_no:.4f}, R²={r2_no:.4f}")

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Plot 2: Predict vs True (DS)
# ============================================================
plt.figure(figsize=(6, 6))
plt.scatter(slave_y, pred_ds, alpha=0.7)
plt.plot([min(slave_y), max(slave_y)],
         [min(slave_y), max(slave_y)],
         linestyle='--', linewidth=1)

plt.xlabel("True Protein")
plt.ylabel("Predicted Protein")
plt.title(f"Predict vs True (DS)\nRMSE={rmse_ds:.4f}, R²={r2_ds:.4f}")

plt.grid(True)
plt.tight_layout()
plt.show()
