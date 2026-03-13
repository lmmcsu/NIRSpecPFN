import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from kennard_stone import train_test_split as ks_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.utils import resample
from scipy.stats import wilcoxon
from tabpfn import TabPFNRegressor
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tabpfn.*")

# ==============================================================
# 1. Configuration
# ==============================================================

file_path = r"C:\Users\zmzhang\Desktop\A4_processed.xlsx"
tabpfn_cache_dir = r"D:\workspace\TabPFN\tabpfn"

target_property = "Protein"

all_preprocessing_sheets = [
    "Raw",
    "MSC",
    "SNV",
    "First Derivative",
    "SG-2D",
    "airPLS"
]

output_dir = os.path.join(os.path.dirname(__file__), f"wheat_{target_property}_significance_results")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(tabpfn_cache_dir, "tabpfn-v2.5-regressor-v2.5_real.ckpt")

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", tabpfn_cache_dir)

# ==============================================================
# 2. Model evaluation functions (unchanged)
# ==============================================================

def evaluate_tabpfn(X_tr, y_tr, X_te, y_te, seed):
    """Train and evaluate TabPFN."""
    model = TabPFNRegressor(model_path=model_path, device=device, random_state=seed)
    setattr(model, "ignore_pretraining_limits", True)

    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, preds))
    return rmse


def evaluate_plsr(X_tr, y_tr, X_te, y_te, seed):
    """Train and evaluate PLSR with grid search."""
    param_grid = {"n_components": list(range(5, 16))}
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        PLSRegression(),
        param_grid,
        cv=cv_inner,
        scoring="neg_mean_squared_error",
    )

    grid.fit(X_tr, y_tr)
    preds = grid.best_estimator_.predict(X_te).ravel()

    rmse = np.sqrt(mean_squared_error(y_te, preds))
    return rmse


def evaluate_svr(X_tr, y_tr, X_te, y_te, seed):
    """Train and evaluate SVR with grid search."""
    param_grid = {
        "C": [100, 200, 300, 400, 500],
        "epsilon": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
        "kernel": ["rbf"],
    }

    cv_inner = KFold(n_splits=5, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        SVR(),
        param_grid,
        cv=cv_inner,
        scoring="neg_mean_squared_error",
    )

    grid.fit(X_tr, y_tr)
    preds = grid.best_estimator_.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, preds))
    return rmse


# ==============================================================
# 3. Fixed Kennard‑Stone split based on a reference sheet
# ==============================================================
print("\n>>> Creating fixed train/test split using reference sheet")

# Use the first sheet as reference (all sheets have the same sample order)
ref_sheet = all_preprocessing_sheets[0]  # "Raw spectral"
df_ref = pd.read_excel(file_path, sheet_name=ref_sheet)
X_ref = df_ref.iloc[:, 1:-2].to_numpy()
y_ref = df_ref[target_property].to_numpy()

# Append a column of indices to retrieve them after KS split
indices = np.arange(len(X_ref)).reshape(-1, 1)
X_with_idx = np.hstack([X_ref, indices])

# Perform Kennard‑Stone split
X_train_idx, X_test_idx, y_train, y_test = ks_split(X_with_idx, y_ref, test_size=0.3)

# Extract the indices of training and test samples
train_idx = X_train_idx[:, -1].astype(int)
test_idx  = X_test_idx[:, -1].astype(int)

print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")

# ==============================================================
# 4. Cache data for all sheets using the fixed indices
# ==============================================================
dataset_cache = {}
for sheet in all_preprocessing_sheets:
    df = pd.read_excel(file_path, sheet_name=sheet)
    X_all = df.iloc[:, 1:-2].to_numpy()
    y_all = df[target_property].to_numpy()

    # Use the same indices for every sheet
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_test  = X_all[test_idx]
    y_test  = y_all[test_idx]

    dataset_cache[sheet] = (X_train, X_test, y_train, y_test)

# ==============================================================
# 5. Determine best preprocessing for each model (now on unified data)
# ==============================================================
print("\n>>> Determining best preprocessing for each model")

best_rmse = {"NIRSpecPFN": np.inf, "PLSR": np.inf, "SVR": np.inf}
best_sheet = {"NIRSpecPFN": None, "PLSR": None, "SVR": None}

for sheet in all_preprocessing_sheets:
    print(f"Evaluating preprocessing: {sheet}")

    X_train, X_test, y_train, y_test = dataset_cache[sheet]

    rmse_tabpfn = evaluate_tabpfn(X_train, y_train, X_test, y_test, seed=42)
    rmse_plsr   = evaluate_plsr  (X_train, y_train, X_test, y_test, seed=42)
    rmse_svr    = evaluate_svr   (X_train, y_train, X_test, y_test, seed=42)

    if rmse_tabpfn < best_rmse["NIRSpecPFN"]:
        best_rmse["NIRSpecPFN"] = rmse_tabpfn
        best_sheet["NIRSpecPFN"] = sheet

    if rmse_plsr < best_rmse["PLSR"]:
        best_rmse["PLSR"] = rmse_plsr
        best_sheet["PLSR"] = sheet

    if rmse_svr < best_rmse["SVR"]:
        best_rmse["SVR"] = rmse_svr
        best_sheet["SVR"] = sheet

print("\n>>> Best preprocessing")
for model in best_sheet:
    print(model, ":", best_sheet[model])

# ==============================================================
# 6. Bootstrap significance analysis (paired resampling)
# ==============================================================
print("\n>>> Starting bootstrap analysis")

n_bootstrap = 100
rmse_tabpfn = []
rmse_plsr   = []
rmse_svr    = []

# Number of training samples (same for all sheets)
n_train = len(train_idx)

for i in range(n_bootstrap):
    # Generate one set of bootstrap indices (shared by all models)
    boot_idx = resample(np.arange(n_train), replace=True, random_state=i)

    # --- NIRSpecPFN ---
    sheet = best_sheet["NIRSpecPFN"]
    X_train, X_test, y_train, y_test = dataset_cache[sheet]
    X_boot = X_train[boot_idx]
    y_boot = y_train[boot_idx]
    rmse = evaluate_tabpfn(X_boot, y_boot, X_test, y_test, seed=i)
    rmse_tabpfn.append(rmse)

    # --- PLSR ---
    sheet = best_sheet["PLSR"]
    X_train, X_test, y_train, y_test = dataset_cache[sheet]
    X_boot = X_train[boot_idx]
    y_boot = y_train[boot_idx]
    rmse = evaluate_plsr(X_boot, y_boot, X_test, y_test, seed=i)
    rmse_plsr.append(rmse)

    # --- SVR ---
    sheet = best_sheet["SVR"]
    X_train, X_test, y_train, y_test = dataset_cache[sheet]
    X_boot = X_train[boot_idx]
    y_boot = y_train[boot_idx]
    rmse = evaluate_svr(X_boot, y_boot, X_test, y_test, seed=i)
    rmse_svr.append(rmse)

print("\n>>> Bootstrap completed")

# ==============================================================
# 7. Statistical significance tests (unchanged)
# ==============================================================
print("\n>>> Wilcoxon signed-rank tests")

stat1, p1 = wilcoxon(rmse_tabpfn, rmse_plsr)
stat2, p2 = wilcoxon(rmse_tabpfn, rmse_svr)

print(f"NIRSpecPFN vs PLSR p-value: {p1:.6f}")
print(f"NIRSpecPFN vs SVR  p-value: {p2:.6f}")

# ==============================================================
# 8. Save bootstrap results (unchanged)
# ==============================================================
results_df = pd.DataFrame({
    "NIRSpecPFN_RMSEP": rmse_tabpfn,
    "PLSR_RMSEP": rmse_plsr,
    "SVR_RMSEP": rmse_svr
})

excel_path = os.path.join(output_dir, f"wheat_{target_property}_bootstrap_rmse_results.xlsx")
results_df.to_excel(excel_path, index=False)

print("\nBootstrap RMSEP results saved to:", excel_path)

# ==============================================================
# 9. Plot RMSE distributions
# ==============================================================
def p_to_star(p):
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 5e-2:
        return "*"
    return "ns"

def add_sig_bar(ax, x1, x2, y, h, text, lw=1.2):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], c="black", lw=lw)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=11)

_, p_tab_vs_plsr = wilcoxon(rmse_tabpfn, rmse_plsr)
_, p_tab_vs_svr = wilcoxon(rmse_tabpfn, rmse_svr)

star_tab_vs_plsr = p_to_star(p_tab_vs_plsr)
star_tab_vs_svr = p_to_star(p_tab_vs_svr)

print(f"NIRSpecPFN vs PLSR: p={p_tab_vs_plsr:.6g}, sig={star_tab_vs_plsr}")
print(f"NIRSpecPFN vs SVR : p={p_tab_vs_svr:.6g}, sig={star_tab_vs_svr}")

# plot
fig, ax = plt.subplots(figsize=(6, 5))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

ax.boxplot(
    [rmse_plsr, rmse_svr, rmse_tabpfn],
    tick_labels=["PLSR", "SVR", "NIRSpecPFN"],
    whis=3.0,
    showfliers=False,
)

ax.set_ylabel("RMSEP",fontsize=18)
ax.set_title("Bootstrap RMSEP Distribution",fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)

all_vals = np.concatenate([rmse_plsr, rmse_svr, rmse_tabpfn])
y_max = np.max(all_vals)
y_min = np.min(all_vals)
yr = max(y_max - y_min, 1e-6)

h = 0.03 * yr
y1 = y_max + 0.05 * yr
y2 = y_max + 0.15 * yr

# position of significance bars：PLSR=1, SVR=2, NIRSpecPFN=3
add_sig_bar(ax, 1, 3, y1, h, star_tab_vs_plsr)  # NIRSpecPFN vs PLSR
add_sig_bar(ax, 2, 3, y2, h, star_tab_vs_svr)   # NIRSpecPFN vs SVR

ax.set_ylim(y_min - 0.05 * yr, y_max + 0.28 * yr)
plt.tight_layout()
plot_path = os.path.join(output_dir, f"bootstrap_rmsep_results_{target_property.split(' ')[0]}.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print("RMSEP distribution plot saved to:", plot_path)
print("\n>>> Significance analysis completed.")