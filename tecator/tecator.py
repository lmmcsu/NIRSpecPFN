import os
import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning, module="tabpfn.*")

# ==============================================================
# 1. Configuration & Path Settings
# ==============================================================
file_path = r"D:\A\CSU\NIRdatasets\tecator\preprocessed_tecator.xlsx"
tabpfn_cache_dir = r"D:\workspace\TabPFN\tabpfn"

# Available Sheets: "Raw", "MSC", "SNV", "SG-2D", "airPLS", "all"
selected_sheet = "all"
target_property = "moisture"  # moisture，fat，protein
selection_mode = "per_model"  
all_preprocessing_sheets = [
    "Raw",
    "MSC",
    "SNV",
    "SG-2D",
    "airPLS"
]

output_dir = os.path.join(os.path.dirname(__file__), f"tacator_{target_property}_results")
plot_dir = os.path.join(output_dir, f"tacator_{target_property}_plots")
excel_output_path = os.path.join(output_dir, f"tacator_{target_property}_metrics.xlsx")
os.makedirs(plot_dir, exist_ok=True)

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", tabpfn_cache_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(tabpfn_cache_dir, "tabpfn-v2.5-regressor-v2.5_real.ckpt")


def calculate_sep(y_true, y_pred):
    """Calculate Standard Error of Prediction (SEP)."""
    n = len(y_true)
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - 1))

def calculate_rpd(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sd = np.std(y_true, ddof=1)
    if rmse == 0:
        return np.nan
    return sd / rmse

def evaluate_cv_rmsecv(model_name, X_tr, y_tr, seed):
    """Cross-validated RMSE on training set for one (preprocessing, model) combination."""
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
        grid = GridSearchCV(PLSRegression(), param_grid, cv=cv_inner, scoring="neg_mean_squared_error")
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
        grid = GridSearchCV(SVR(), param_grid, cv=cv_inner, scoring="neg_mean_squared_error")
        grid.fit(X_tr, y_tr)
        return {
            "RMSECV": np.sqrt(-grid.best_score_),
            "Best_n": np.nan,
            "Best_C": grid.best_params_["C"],
            "Best_epsilon": grid.best_params_["epsilon"],
        }

    raise ValueError(f"Unsupported model: {model_name}")


def fit_and_evaluate_test(model_name, X_tr, y_tr, X_te, y_te, seed, best_n=np.nan, best_c=np.nan, best_epsilon=np.nan):
    """Fit model on full training set and evaluate on test set."""
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

    return {
        "R2": r2_score(y_te, preds),
        "RMSE": np.sqrt(mean_squared_error(y_te, preds)),
        "MAE": mean_absolute_error(y_te, preds),
        "SEP": calculate_sep(y_te, preds),
        "RPD": calculate_rpd(y_te, preds),
        "preds": preds,
    }

def safe_name(name):
    """Sanitize names for filesystem and Excel sheet usage."""
    return re.sub(r"[\\/:*?\"<>|]+", "_", name)


def plot_true_vs_pred(y_true, results_by_method, preprocessing_name, save_path):
    """Create and save scatter plots of true vs predicted values for each method."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    methods = ["NIRSpecPFN", "PLSR", "SVR"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    y_min = min(np.min(y_true), *(np.min(results_by_method[m]["preds"]) for m in methods))
    y_max = max(np.max(y_true), *(np.max(results_by_method[m]["preds"]) for m in methods))

    for ax, method in zip(axes, methods):
        y_pred = results_by_method[method]["preds"]
        ax.scatter(y_true, y_pred, alpha=0.75)
        ax.plot([y_min, y_max], [y_min, y_max], "r--", linewidth=1.2)
        ax.set_title(f"{method} | {preprocessing_name}", fontsize=24)
        ax.set_xlabel("True", fontsize=24)
        ax.set_ylabel("Predicted", fontsize=24)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.text(0.05, 0.95, f"R²: {results_by_method[method]['R2']:.4f}\nRMSEP: {results_by_method[method]['RMSE']:.4f}\n", transform=ax.transAxes, fontsize=20, verticalalignment='top')

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ==============================================================
# 2. Run all preprocessing settings
# ==============================================================
if selected_sheet == "all":
    sheets_to_run = all_preprocessing_sheets
else:
    sheets_to_run = [selected_sheet]

seed = 42
model_names = ["NIRSpecPFN", "PLSR", "SVR"]

print("\n>>> Start benchmarking across preprocessing sheets")
print(f"Target property: {target_property}")
print(f"Sheets to run: {sheets_to_run}")

dataset_cache = {}

for sheet_name in sheets_to_run:
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    subset = df.iloc[:, 1]
    train_mask = subset.isin(['C', 'M'])
    test_mask = subset == 'T'

    X = df.iloc[:, 2:102].to_numpy()
    y = df[target_property].to_numpy()

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"warning: sheet '{sheet_name}' is blank, skip this sheet.")
        continue

    dataset_cache[sheet_name] = (X_train, X_test, y_train, y_test)

all_records = []

for sheet_name, (X_train, X_test, y_train, y_test) in dataset_cache.items():
    print(f"\n--- Running sheet: {sheet_name} ---")

    results_by_method = {}

    for model_name in model_names:
        cv_info = evaluate_cv_rmsecv(model_name, X_train, y_train, seed=seed)
        test_info = fit_and_evaluate_test(
            model_name,
            X_train,
            y_train,
            X_test,
            y_test,
            seed=seed,
            best_n=cv_info["Best_n"],
            best_c=cv_info["Best_C"],
            best_epsilon=cv_info["Best_epsilon"],
        )

        results_by_method[model_name] = test_info
        all_records.append(
            {
                "Preprocessing": sheet_name,
                "Method": model_name,
                "Train_RMSECV": cv_info["RMSECV"],
                "Best_n": cv_info["Best_n"],
                "Best_C": cv_info["Best_C"],
                "Best_epsilon": cv_info["Best_epsilon"],
                "Test_R²": test_info["R2"],
                "Test_RMSEP": test_info["RMSE"],
                "Test_MAE": test_info["MAE"],
                "Test_SEP": test_info["SEP"],
                "Test_RPD": test_info["RPD"],
            }
        )

        print(
            f"{model_name}: RMSECV={cv_info['RMSECV']:.4f}, "
            f"R²={test_info['R2']:.4f}, RMSEP={test_info['RMSE']:.4f}, "
            f"MAE={test_info['MAE']:.4f}, SEP={test_info['SEP']:.4f}, RPD={test_info['RPD']:.4f}"
        )

    plot_path = os.path.join(plot_dir, f"{safe_name(sheet_name)}_true_vs_pred_{target_property.split(' ')[0]}.png")
    plot_true_vs_pred(y_test, results_by_method, sheet_name, plot_path)
    print(f"Saved scatter plot: {plot_path}")

all_metrics_df = pd.DataFrame(all_records)

# Select best preprocessing by training RMSECV
best_idx_global = all_metrics_df["Train_RMSECV"].idxmin()
best_global_df = all_metrics_df.loc[[best_idx_global]].reset_index(drop=True)

best_per_model_df = (
    all_metrics_df.sort_values("Train_RMSECV")
    .groupby("Method", as_index=False)
    .first()
    .reset_index(drop=True)
)

if selection_mode == "per_model":
    selected_for_comparison_df = best_per_model_df.copy()
    print("\n>>> Selection mode: per_model")
    for _, row in selected_for_comparison_df.iterrows():
        print(
            f"{row['Method']}: preprocessing={row['Preprocessing']}, "
            f"RMSECV={row['Train_RMSECV']:.4f}, Test_RMSEP={row['Test_RMSEP']:.4f}"
        )
else:
    selected_for_comparison_df = best_global_df.copy()
    print("\n>>> Selection mode: global")
    row = selected_for_comparison_df.iloc[0]
    print(
        f"Global best: {row['Method']} | {row['Preprocessing']}, "
        f"RMSECV={row['Train_RMSECV']:.4f}, Test_RMSEP={row['Test_RMSEP']:.4f}"
    )


# ==============================================================
# 3. Export all metrics to one Excel file
# ==============================================================
with pd.ExcelWriter(excel_output_path, engine="openpyxl") as writer:
    all_metrics_df.to_excel(writer, sheet_name="All_Model_Metrics", index=False)
    best_per_model_df.to_excel(writer, sheet_name="Best_Per_Model", index=False)
    best_global_df.to_excel(writer, sheet_name="Best_Global", index=False)
    selected_for_comparison_df.to_excel(writer, sheet_name="Selected_For_Compare", index=False)

print("\n>>> Completed all runs.")
print(f"Metrics Excel saved to: {excel_output_path}")
print(f"Scatter plots saved in: {plot_dir}")
