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

file_path = r"D:\A\CSU\NIRdatasets\wheat\A2_processed.xlsx"
tabpfn_cache_dir = r"D:\workspace\TabPFN\tabpfn"

target_property = "Protein"
selection_mode = "per_model"  

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
# 2. Model evaluation functions
# ==============================================================

def evaluate_cv_rmsecv(model_name, X_tr, y_tr, seed):
    """Return RMSECV and best hyperparameters estimated on training set only."""
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=seed)

    if model_name == "NIRSpecPFN":
        fold_rmses = []
        for train_idx, valid_idx in cv_inner.split(X_tr):
            X_fold_train, X_fold_valid = X_tr[train_idx], X_tr[valid_idx]
            y_fold_train, y_fold_valid = y_tr[train_idx], y_tr[valid_idx]

            model = TabPFNRegressor(model_path=model_path, device=device, random_state=seed)
            setattr(model, "ignore_pretraining_limits", True)
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_valid)
            fold_rmses.append(np.sqrt(mean_squared_error(y_fold_valid, preds)))

        return {
            "RMSECV": float(np.mean(fold_rmses)),
            "Best_n": np.nan,
            "Best_C": np.nan,
            "Best_epsilon": np.nan,
        }

    if model_name == "PLSR":
        param_grid = {"n_components": list(range(5, 16))}
        grid = GridSearchCV(
            PLSRegression(),
            param_grid,
            cv=cv_inner,
            scoring="neg_mean_squared_error",
        )
        grid.fit(X_tr, y_tr)
        return {
            "RMSECV": np.sqrt(-grid.best_score_),
            "Best_n": grid.best_params_["n_components"],
            "Best_C": np.nan,
            "Best_epsilon": np.nan,
        }

    if model_name == "SVR":
        param_grid = {
            "C": [100, 200, 300, 400, 500],
            "epsilon": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
            "kernel": ["rbf"],
        }
        grid = GridSearchCV(
            SVR(),
            param_grid,
            cv=cv_inner,
            scoring="neg_mean_squared_error",
        )
        grid.fit(X_tr, y_tr)
        return {
            "RMSECV": np.sqrt(-grid.best_score_),
            "Best_n": np.nan,
            "Best_C": grid.best_params_["C"],
            "Best_epsilon": grid.best_params_["epsilon"],
        }

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_test_rmse(model_name, X_tr, y_tr, X_te, y_te, seed, best_n=np.nan, best_c=np.nan, best_epsilon=np.nan):
    """Fit on full training set and evaluate RMSE on fixed test set."""
    if model_name == "NIRSpecPFN":
        model = TabPFNRegressor(model_path=model_path, device=device, random_state=seed)
        setattr(model, "ignore_pretraining_limits", True)
    elif model_name == "PLSR":
        model = PLSRegression(n_components=int(best_n))
    elif model_name == "SVR":
        model = SVR(C=float(best_c), epsilon=float(best_epsilon), kernel="rbf")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    if hasattr(preds, "ravel"):
        preds = preds.ravel()
    return np.sqrt(mean_squared_error(y_te, preds))


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
y_all = y_ref
y_train_fixed = y_all[train_idx]
y_test_fixed = y_all[test_idx]

for sheet in all_preprocessing_sheets:
    df = pd.read_excel(file_path, sheet_name=sheet)
    X_all = df.iloc[:, 1:-2].to_numpy()

    # Use the same indices for every sheet
    X_train = X_all[train_idx]
    X_test  = X_all[test_idx]

    dataset_cache[sheet] = (X_train, X_test)

# ==============================================================
# 5. Select preprocessing using training-set RMSECV only
# ==============================================================
print("\n>>> Selecting preprocessing with RMSECV on training set")

model_names = ["NIRSpecPFN", "PLSR", "SVR"]
cv_records = []

for sheet in all_preprocessing_sheets:
    X_train, _ = dataset_cache[sheet]
    for model_name in model_names:
        cv_res = evaluate_cv_rmsecv(model_name, X_train, y_train_fixed, seed=42)
        cv_records.append(
            {
                "Preprocessing": sheet,
                "Model": model_name,
                "RMSECV": cv_res["RMSECV"],
                "Best_n": cv_res["Best_n"],
                "Best_C": cv_res["Best_C"],
                "Best_epsilon": cv_res["Best_epsilon"],
            }
        )
        print(
            f"{sheet} | {model_name}: RMSECV={cv_res['RMSECV']:.4f}, "
            f"Best_n={cv_res['Best_n']}, Best_C={cv_res['Best_C']}, Best_epsilon={cv_res['Best_epsilon']}"
        )

cv_df = pd.DataFrame(cv_records)

best_global_row = cv_df.loc[cv_df["RMSECV"].idxmin()]

best_per_model = {}
for model_name in model_names:
    model_rows = cv_df[cv_df["Model"] == model_name]
    best_per_model[model_name] = model_rows.loc[model_rows["RMSECV"].idxmin()]

if selection_mode == "global":
    best_preprocessing = best_global_row["Preprocessing"]
    selected_config = {}
    for model_name in model_names:
        X_train, _ = dataset_cache[best_preprocessing]
        cv_res = evaluate_cv_rmsecv(model_name, X_train, y_train_fixed, seed=42)
        selected_config[model_name] = pd.Series({
            "Preprocessing": best_preprocessing,
            "RMSECV": cv_res["RMSECV"],
            "Best_n": cv_res["Best_n"],
            "Best_C": cv_res["Best_C"],
            "Best_epsilon": cv_res["Best_epsilon"],
        })
else:
    selected_config = best_per_model
    print("\n>>> Selection mode: per_model")
    for model_name in model_names:
        row = selected_config[model_name]
        print(
            f"{model_name}: best preprocessing={row['Preprocessing']}, RMSECV={row['RMSECV']:.4f}"
        )

# Final one-time test evaluation on fixed test set
print("\n>>> Final test evaluation with selected preprocessing")
final_records = []
selected_sheet_by_model = {}

for model_name in model_names:
    selected_row = selected_config[model_name]
    sheet = selected_row["Preprocessing"]
    selected_sheet_by_model[model_name] = sheet

    X_train, X_test = dataset_cache[sheet]
    test_rmse = evaluate_test_rmse(
        model_name,
        X_train,
        y_train_fixed,
        X_test,
        y_test_fixed,
        seed=42,
        best_n=selected_row["Best_n"],
        best_c=selected_row["Best_C"],
        best_epsilon=selected_row["Best_epsilon"],
    )

    final_records.append(
        {
            "Model": model_name,
            "Selected_Preprocessing": sheet,
            "Train_RMSECV": selected_row["RMSECV"],
            "Best_n": selected_row["Best_n"],
            "Best_C": selected_row["Best_C"],
            "Best_epsilon": selected_row["Best_epsilon"],
            "Test_RMSEP": test_rmse,
        }
    )
    print(
        f"{model_name}: preprocessing={sheet}, RMSECV={selected_row['RMSECV']:.4f}, Test_RMSEP={test_rmse:.4f}"
    )

final_df = pd.DataFrame(final_records)


def evaluate_tabpfn(X_tr, y_tr, X_te, y_te, seed):
    return evaluate_test_rmse("NIRSpecPFN", X_tr, y_tr, X_te, y_te, seed=seed)


def evaluate_plsr(X_tr, y_tr, X_te, y_te, seed):
    cv_res = evaluate_cv_rmsecv("PLSR", X_tr, y_tr, seed=seed)
    return evaluate_test_rmse(
        "PLSR",
        X_tr,
        y_tr,
        X_te,
        y_te,
        seed=seed,
        best_n=cv_res["Best_n"],
        best_c=cv_res["Best_C"],
        best_epsilon=cv_res["Best_epsilon"],
    )


def evaluate_svr(X_tr, y_tr, X_te, y_te, seed):
    cv_res = evaluate_cv_rmsecv("SVR", X_tr, y_tr, seed=seed)
    return evaluate_test_rmse(
        "SVR",
        X_tr,
        y_tr,
        X_te,
        y_te,
        seed=seed,
        best_n=cv_res["Best_n"],
        best_c=cv_res["Best_C"],
        best_epsilon=cv_res["Best_epsilon"],
    )

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
    sheet = selected_sheet_by_model["NIRSpecPFN"]
    X_train, X_test = dataset_cache[sheet]
    X_boot = X_train[boot_idx]
    y_boot = y_train_fixed[boot_idx]
    rmse = evaluate_tabpfn(X_boot, y_boot, X_test, y_test_fixed, seed=i)
    rmse_tabpfn.append(rmse)

    # --- PLSR ---
    sheet = selected_sheet_by_model["PLSR"]
    X_train, X_test = dataset_cache[sheet]
    X_boot = X_train[boot_idx]
    y_boot = y_train_fixed[boot_idx]
    rmse = evaluate_plsr(X_boot, y_boot, X_test, y_test_fixed, seed=i)
    rmse_plsr.append(rmse)

    # --- SVR ---
    sheet = selected_sheet_by_model["SVR"]
    X_train, X_test = dataset_cache[sheet]
    X_boot = X_train[boot_idx]
    y_boot = y_train_fixed[boot_idx]
    rmse = evaluate_svr(X_boot, y_boot, X_test, y_test_fixed, seed=i)
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

ks_indices_df = pd.DataFrame(
    {
        "Train_Index": pd.Series(train_idx),
        "Test_Index": pd.Series(test_idx),
    }
)

excel_path = os.path.join(output_dir, f"wheat_{target_property}_bootstrap_rmse_results.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    cv_df.to_excel(writer, sheet_name="CV_Selection", index=False)
    final_df.to_excel(writer, sheet_name="Final_Test", index=False)
    results_df.to_excel(writer, sheet_name="Bootstrap_RMSEP", index=False)
    ks_indices_df.to_excel(writer, sheet_name="KS_Indices", index=False)

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
