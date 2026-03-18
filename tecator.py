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
file_path = r"C:\Users\zmzhang\Desktop\tacator\preprocessed_tecator.xlsx"
tabpfn_cache_dir = r"D:\workspace\TabPFN\tabpfn"

# Available Sheets: "Raw", "MSC", "SNV", "First Derivative", "SG-2D", "airPLS", "all"
selected_sheet = "all"
target_property = "moisture"  # moisture，fat，protein
all_preprocessing_sheets = [
    "Raw",
    "MSC",
    "SNV",
    "First Derivative",
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

def get_pfn_metrics(X_tr, y_tr, X_te, y_te, seed):
    """Initialize, train, and evaluate TabPFN model."""
    model = TabPFNRegressor(model_path=model_path, device=device, random_state=seed)
    setattr(model, "ignore_pretraining_limits", True)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return {
        "R2": r2_score(y_te, preds),
        "RMSE": np.sqrt(mean_squared_error(y_te, preds)),
        "MAE": mean_absolute_error(y_te, preds),
        "SEP": calculate_sep(y_te, preds),
        "RPD": calculate_rpd(y_te, preds),
        "Best_n": np.nan,
        "Best_C": np.nan,
        "Best_epsilon": np.nan,
        "preds": preds,
    }


def get_plsr_metrics(X_tr, y_tr, X_te, y_te, seed):
    """Grid Search for PLSR components and evaluation."""
    param_grid = {"n_components": list(range(5, 16))}
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(PLSRegression(), param_grid, cv=cv_inner, scoring="neg_mean_squared_error")
    grid.fit(X_tr, y_tr)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_te).ravel()

    return {
        "R2": r2_score(y_te, preds),
        "RMSE": np.sqrt(mean_squared_error(y_te, preds)),
        "MAE": mean_absolute_error(y_te, preds),
        "SEP": calculate_sep(y_te, preds),
        "RPD": calculate_rpd(y_te, preds),
        "Best_n": grid.best_params_["n_components"],
        "Best_C": np.nan,
        "Best_epsilon": np.nan,
        "preds": preds,
    }


def get_svr_metrics(X_tr, y_tr, X_te, y_te, seed):
    """Grid Search for SVR parameters and evaluation."""
    param_grid = {
        "C": [100, 200, 300, 400, 500],
        "epsilon": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
        "kernel": ["rbf"],
    }
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(SVR(), param_grid, cv=cv_inner, scoring="neg_mean_squared_error")
    grid.fit(X_tr, y_tr)

    preds = grid.best_estimator_.predict(X_te)
    return {
        "R2": r2_score(y_te, preds),
        "RMSE": np.sqrt(mean_squared_error(y_te, preds)),
        "MAE": mean_absolute_error(y_te, preds),
        "SEP": calculate_sep(y_te, preds),
        "RPD": calculate_rpd(y_te, preds),
        "Best_n": np.nan,
        "Best_C": grid.best_params_["C"],
        "Best_epsilon": grid.best_params_["epsilon"],
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



def collect_metrics_dataframe(results_by_method):
    """Convert method results to a metrics DataFrame for Excel export."""
    rows = []
    for method_name, result in results_by_method.items():
        rows.append(
            {
                "Method": method_name,
                "R²": result["R2"],
                "RMSEP": result["RMSE"],
                "MAE": result["MAE"],
                "SEP": result["SEP"],
                "RPD": result["RPD"],
                "Best_n": result["Best_n"],
                "Best_C": result["Best_C"],
                "Best_epsilon": result["Best_epsilon"],
            }
        )
    return pd.DataFrame(rows)


# ==============================================================
# 2. Run all preprocessing settings
# ==============================================================
if selected_sheet == "all":
    sheets_to_run = all_preprocessing_sheets
else:
    sheets_to_run = [selected_sheet]

all_metrics_by_sheet = {}

print("\n>>> Start benchmarking across preprocessing sheets")
print(f"Target property: {target_property}")
print(f"Sheets to run: {sheets_to_run}")

for sheet_name in sheets_to_run:
    print(f"\n--- Running sheet: {sheet_name} ---")

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    subset = df.iloc[:, 1]                     # the second column indicates subset (C/M/T)
    train_mask = subset.isin(['C', 'M'])       
    test_mask  = subset == 'T'                  

    X = df.iloc[:, 2:102].to_numpy()
    y = df[target_property].to_numpy()

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"warning:sheet '{sheet_name}'is blank, skip the sheet。")
        continue

    pfn_results = get_pfn_metrics(X_train, y_train, X_test, y_test, seed=42)
    plsr_results = get_plsr_metrics(X_train, y_train, X_test, y_test, seed=42)
    svr_results = get_svr_metrics(X_train, y_train, X_test, y_test, seed=42)

    results_by_method = {
        "NIRSpecPFN": pfn_results,
        "PLSR": plsr_results,
        "SVR": svr_results,
    }

    for method_name, result in results_by_method.items():
        print(
            f"{method_name}: R²={result['R2']:.4f}, RMSEP={result['RMSE']:.4f}, "
            f"MAE={result['MAE']:.4f}, SEP={result['SEP']:.4f}, RPD={result['RPD']:.4f}"
        )

    plot_path = os.path.join(plot_dir, f"{safe_name(sheet_name)}_true_vs_pred_{target_property.split(' ')[0]}.png")
    plot_true_vs_pred(y_test, results_by_method, sheet_name, plot_path)
    print(f"Saved scatter plot: {plot_path}")

    metrics_df = collect_metrics_dataframe(results_by_method)
    all_metrics_by_sheet[sheet_name] = metrics_df


# ==============================================================
# 3. Export all metrics to one Excel file (one sheet per preprocessing)
# ==============================================================
with pd.ExcelWriter(excel_output_path, engine="openpyxl") as writer:
    for sheet_name, metrics_df in all_metrics_by_sheet.items():
        excel_sheet_name = safe_name(sheet_name)[:31]
        metrics_df.to_excel(writer, sheet_name=excel_sheet_name, index=False)

print("\n>>> Completed all runs.")
print(f"Metrics Excel saved to: {excel_output_path}")
print(f"Scatter plots saved in: {plot_dir}")