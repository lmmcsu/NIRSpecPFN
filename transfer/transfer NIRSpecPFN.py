import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing.process import derivative, savitzky_golay, airPLS, msc, snv, detrend
from tabpfn import TabPFNRegressor
import torch
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR",r"D:\workspace\TabPFN\tabpfn")


# ============================================================
# Load local TabPFN model (v2.5 real)
# ============================================================
def load_local_tabpfn(kind="regressor", version="2.5", variant="real", device=DEVICE):
    cache = os.environ["TABPFN_MODEL_CACHE_DIR"]

    if version in ("2", "2.0"):
        fname = f"tabpfn-v2-{kind}.ckpt"
    elif version in ("2.5", "2_5"):
        fname = f"tabpfn-v2.5-{kind}-v2.5_{variant}.ckpt"
    else:
        raise ValueError("Unsupported TabPFN version.")

    path = os.path.join(cache, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    model = TabPFNRegressor(model_path=path, device=device, random_state=42)
    setattr(model, "ignore_pretraining_limits", True)

    return model


# ============================================================
# Utility functions
# ============================================================
def normalize_wavelengths(cols):
    wl = []
    for c in cols:
        try:
            wl.append(float(c))
        except:
            continue
    return wl


def format_wl(w):
    s = "{:.3f}".format(w).rstrip("0").rstrip(".")
    if "." in s:
        return s
    return str(int(w))


def normalize_df_columns(df, y_col="Protein"):
    new_cols = []
    for col in df.columns:
        if col in ["ID", y_col]:
            new_cols.append(col)
            continue

        try:
            f = float(col)
            new_cols.append(format_wl(f))
        except:
            new_cols.append(col)
    df.columns = new_cols
    return df


def load_and_align(master_path, slave_path, y_col="Protein"):
    """ Master = A, Slave = B """
    A = pd.read_excel(master_path)
    B = pd.read_excel(slave_path)

    Aw = normalize_wavelengths(A.columns)
    Bw = normalize_wavelengths(B.columns)

    print("Master wavelength count:", len(Aw))
    print("Slave wavelength count:", len(Bw))

    common = sorted(list(set(Aw).intersection(set(Bw))))
    print("Common wavelength count:", len(common))

    common_cols = [format_wl(w) for w in common]

    A = normalize_df_columns(A, y_col)
    B = normalize_df_columns(B, y_col)

    A = A[["ID", y_col] + common_cols]
    B = B[["ID", y_col] + common_cols]

    print("Aligned shapes:", A.shape, B.shape)
    return A, B, common_cols


def sep(y_true, y_pred):
    n = len(y_true)
    return 0 if n <= 1 else np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - 1))


def evaluate(y, y_pred, title):
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    sep_val = sep(y, y_pred)

    print(f"{title}")
    print(f"R2={r2:.4f}   RMSE={rmse:.4f}   MAE={mae:.4f}   SEP={sep_val:.4f}\n")


# ============================================================
# Load data and preprocess
# ============================================================
master_path = r"D:\A\CSU\NIRdatasets\wheat\Test_ManufacturerA.xlsx"
slave_path  = r"D:\A\CSU\NIRdatasets\wheat\Cal_ManufacturerA1.xlsx"
# masterA4:Test_ManufacturerA
# slaveA1:Cal_ManufacturerA1  slaveA2:Cal_ManufacturerA2  slaveA3:Cal_ManufacturerA3

print("\n=== Loading & Aligning Wavelengths ===")
A, B, common_cols = load_and_align(master_path, slave_path)

yA = A["Protein"].values
yB = B["Protein"].values
XA = A[common_cols].values
XB = B[common_cols].values

# Preprocessing (SG smoothing)
XA_de = savitzky_golay(XA)
XB_de = savitzky_golay(XB)


# ============================================================
# Train TabPFN on A → Predict B
# ============================================================
print("\n=== TabPFN Calibration Transfer: A → B ===")

reg = load_local_tabpfn("regressor", "2.5", "real", DEVICE)

reg.fit(XA_de, yA)
pred = reg.predict(XB_de)

evaluate(yB, pred, "Calibration Transfer (A → B)")


# ============================================================
# Plot predict vs true
# ============================================================
y_true = yB
y_pred = pred

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

text_info = (
    f"RMSE = {rmse:.4f}\n"
    f"R²   = {r2:.4f}\n"
)
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())

plt.figure(figsize=(7, 6))
plt.scatter(y_true, y_pred, alpha=0.75)
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', linewidth=1)
plt.xlabel("True Protein")
plt.ylabel("Predicted Protein")
plt.title(f"Predict vs True (NIRSpecPFN)\nRMSE={rmse:.4f}, R²={r2:.4f}")
plt.grid(True)
plt.tight_layout()
plt.show()
