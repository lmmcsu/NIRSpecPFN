import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
import statsmodels.stats.stattools as sm_stats   # for durbin_watson
import statsmodels.api as sm                      # for OLS in APaRP
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams.update({'font.size': 15})
# ==============================================================
# 0. Dataset configuration
# ==============================================================

DATASET_CONFIG = {
    "wheat": {
        "path": r"C:\Users\zmzhang\Desktop\A2_processed.xlsx",
        "sheet": "Raw",
        "spectra_cols": (1, -2),
        "target_col": -1,
        "n_wavelengths": 741,
        "target_name": "Protein",
    },
    "soil": {
        "path": r"D:\A\CSU\NIRdatasets\soil\preprocessed_soil.xlsx",
        "sheet": "Raw spectral",
        "spectra_cols": (1, -3),
        "target_col": "TC (%)", #TC (%), OM (%)
        "n_wavelengths": 228,
        "target_name": "TC",
    },
    "tecator": {
        "path": r"D:\A\CSU\NIRdatasets\tecator\preprocessed_tecator.xlsx",
        "sheet": "Raw",
        "spectra_cols": (2, -4),
        "target_col": "fat", #moisture, fat, protein
        "n_wavelengths": 100,
        "target_name": "Fat",
    },
}

CURRENT_DATASET = "wheat"  # Switch: wheat / soil / tecator
config = DATASET_CONFIG[CURRENT_DATASET]

print(f"Current dataset: {CURRENT_DATASET.upper()}")
print(f"Configured wavelengths: {config['n_wavelengths']}")

# ==============================================================
# 1. Load data
# ==============================================================

df = pd.read_excel(config["path"], sheet_name=config["sheet"])
start_col, end_col = config["spectra_cols"]
spectra = df.iloc[:, start_col:end_col].values

target_col = config["target_col"]
y = df.iloc[:, target_col].values if isinstance(target_col, int) else df[target_col].values

n_samples = len(y)
print(f"Loaded samples: {n_samples}")

# ==============================================================
# 2. Durbin-Watson critical values (fixed n=200, alpha=0.05)
# ==============================================================

DW_N_USED = 200
DW_ALPHA = 0.05
DW_TABLE_N200_ALPHA_005 = {
    1: (1.66, 1.68),
    2: (1.65, 1.69),
    3: (1.64, 1.70),
    4: (1.63, 1.72),
    5: (1.62, 1.72),
    6: (1.61, 1.74),
    7: (1.60, 1.75),
    8: (1.59, 1.76),
    9: (1.58, 1.77),
    10: (1.57, 1.80),
    11: (1.65, 1.89),
    12: (1.64, 1.90),
    13: (1.63, 1.91),
    14: (1.62, 1.92),
    15: (1.61, 1.93),
    16: (1.51, 1.85),
    17: (1.50, 1.86),
    18: (1.48, 1.87),
    19: (1.47, 1.88),
    20: (1.46, 1.90),
}

def get_dw_critical_values_for_n200(k):
    """Return (dL, dU, k_used, note) for fixed n=200 and alpha=0.05."""
    k_min, k_max = min(DW_TABLE_N200_ALPHA_005), max(DW_TABLE_N200_ALPHA_005)
    if k < k_min:
        k_used = k_min
        note = f"k={k} is below table range, using k={k_used}."
    elif k > k_max:
        k_used = k_max
        note = f"k={k} is above table range, using k={k_used}."
    elif k in DW_TABLE_N200_ALPHA_005:
        k_used = k
        note = f"Exact k match: k={k_used}."
    else:
        k_used = min(DW_TABLE_N200_ALPHA_005.keys(), key=lambda x: abs(x - k))
        note = f"k={k} not in table, using nearest k={k_used}."

    dL, dU = DW_TABLE_N200_ALPHA_005[k_used]
    return dL, dU, k_used, note

# ==============================================================
# 3. Select optimal latent variables by CV
# ==============================================================

kf = KFold(n_splits=10, shuffle=True, random_state=42)
n_components_range = range(5, 21) if CURRENT_DATASET == "wheat" else range(3, 16)

rmse_cv = []
for n_comp in n_components_range:
    pls = PLSRegression(n_components=n_comp)
    y_pred_cv = cross_val_predict(pls, spectra, y, cv=kf)
    rmse_cv.append(np.sqrt(mean_squared_error(y, y_pred_cv)))

best_n = n_components_range[np.argmin(rmse_cv)]
print(f"Best PLS latent variables: {best_n} (min RMSECV = {np.min(rmse_cv):.4f})")

# ==============================================================
# 4. Fit PLS and compute sorted residuals for RP
# ==============================================================

pls_best = PLSRegression(n_components=best_n)
pls_best.fit(spectra, y)
y_pred = pls_best.predict(spectra).ravel()
residuals = y - y_pred

sort_idx = np.argsort(y)
residuals_sorted = residuals[sort_idx]
print(f"Residuals sorted by {config['target_name']} values.")

# ==============================================================
# 5. Durbin-Watson test on RP
# ==============================================================

print(f"\n{'=' * 70}")
print(f"Durbin-Watson test - {CURRENT_DATASET.upper()} dataset (RP)")
print(f"{'=' * 70}")

d = sm_stats.durbin_watson(residuals_sorted)   
print(f"DW statistic: d = {d:.4f}")
print(f"Samples loaded: n = {n_samples}")
print(f"PLS latent variables (k): {best_n}")
print(f"Fixed table setup: n = {DW_N_USED}, alpha = {DW_ALPHA}")

dL, dU, k_used, k_note = get_dw_critical_values_for_n200(best_n)

print("\nLookup details:")
print(f"  k used in table: {k_used}")
print(f"  note: {k_note}")
print(f"\nCritical values (alpha={DW_ALPHA}):")
print(f"  d_L = {dL:.3f}")
print(f"  d_U = {dU:.3f}")

print("\nDecision:")
print(f"DW statistic d = {d:.4f}")
print(f"Reject no-positive-correlation region: d < {dL:.3f}")
print(f"No-significant-correlation region: d > {dU:.3f}")

if d < dL:
    dw_conclusion = "Significant positive serial correlation"
    dw_interpretation = "Potential nonlinearity remains"
    dw_detail = f"d = {d:.4f} < d_L = {dL:.3f}"
    dw_result = "Nonliner"
elif d > dU:
    dw_conclusion = "No significant serial correlation"
    dw_interpretation = "Linear model assumption is acceptable"
    dw_detail = f"d = {d:.4f} > d_U = {dU:.3f}"
    dw_result = "Liner"
else:
    dw_conclusion = "Inconclusive zone"
    dw_interpretation = "Result is uncertain"
    dw_detail = f"{dL:.3f} ≤ {d:.4f} ≤ {dU:.3f}"
    dw_result = "Inconclusive"

print(f"\nConclusion: {dw_conclusion}")
print(f"Interpretation: {dw_interpretation}")
print(f"Basis: {dw_detail}")
print(f"DW result: {dw_result}")

# ==============================================================
# 6. Runs test on RP
# ==============================================================

def runs_test(residual_values):
    """Runs test to assess residual randomness."""
    signs = np.sign(residual_values)
    signs[signs == 0] = 1

    n1 = np.sum(signs == 1)
    n2 = np.sum(signs == -1)

    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            runs += 1

    mu = None
    sigma = None
    z = None
    p = None
    if n1 > 10 and n2 > 10:
        mu = 2 * n1 * n2 / (n1 + n2) + 1
        sigma = np.sqrt(2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1)))
        z = (runs - mu + 0.5) / sigma
        p = 2 * (1 - stats.norm.cdf(abs(z)))

    return runs, n1, n2, z, p, mu, sigma

print(f"\n{'=' * 70}")
print(f"Runs test - {CURRENT_DATASET.upper()} dataset (RP)")
print(f"{'=' * 70}")

runs, n1, n2, z, p, mu, sigma = runs_test(residuals_sorted)
print(f"Positive residuals n1 = {n1}")
print(f"Negative residuals n2 = {n2}")
print(f"Number of runs R = {runs}")

if z is not None:
    print(f"\nExpected mean mu = {mu:.2f}")
    print(f"Expected std sigma = {sigma:.4f}")
    print(f"Standardized z = {z:.4f}")
    print(f"Two-tailed p-value = {p:.4f}")

    z_crit = 1.96
    print(f"\nCritical threshold: |z| > {z_crit} (alpha=0.05)")

    if abs(z) > z_crit:
        runs_conclusion = "Significant non-random pattern"
        runs_detail = f"|z| = {abs(z):.4f} > {z_crit}"
        runs_pattern = "Too few runs (trend/positive serial structure)" if z < 0 else "Too many runs (oscillatory structure)"
        runs_result = "Nonliner"
    else:
        runs_conclusion = "Random pattern"
        runs_detail = f"|z| = {abs(z):.4f} <= {z_crit}"
        runs_pattern = "Residual ordering appears random"
        runs_result = "Liner"

    print(f"Conclusion: {runs_conclusion}")
    print(f"Pattern: {runs_pattern}")
    print(f"Basis: {runs_detail}")
    print(f"Runs result: {runs_result}")
else:
    runs_conclusion = "Insufficient counts for normal approximation"
    runs_result = "Inconclusive"
    print("Insufficient counts for normal approximation.")
    print(f"Runs result: {runs_result}")

# ==============================================================
# 7. drawing RP scatter plot
# ==============================================================
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
# LOWESS
sorted_idx = np.argsort(y_pred)
y_pred_sorted = y_pred[sorted_idx]
residuals_sorted_pred = residuals[sorted_idx]
lowess_smoothed = lowess(residuals_sorted_pred, y_pred_sorted, frac=0.4)
plt.plot(lowess_smoothed[:, 0], lowess_smoothed[:, 1], color='blue', linewidth=2, label='LOWESS')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title(f'RP')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(f'RP_{CURRENT_DATASET}.png', dpi=300)
plt.show()
# ==============================================================
# 8. APaRP for the first latent variable
# ==============================================================
print(f"\n{'=' * 70}")
print(f"APaRP (first latent variable) - {CURRENT_DATASET.upper()} dataset")
print(f"{'=' * 70}")

T = pls_best.x_scores_
t1 = T[:, 0]

X_extended = np.column_stack([T, t1**2])
X_extended_with_const = sm.add_constant(X_extended)
model_ext = sm.OLS(y, X_extended_with_const).fit()
e_aug = model_ext.resid

b1 = model_ext.params[1]
b11 = model_ext.params[-1]
aug_partial = b1 * t1 + b11 * (t1**2) + e_aug

aug_partial_sorted = aug_partial[sort_idx]

print(f"APaRP computed for first latent variable (t1).")
print(f"b1 = {b1:.4f}, b11 = {b11:.4f}")

# Durbin-Watson on APaRP
d_apa = sm_stats.durbin_watson(aug_partial_sorted)
print(f"\nDurbin-Watson statistic (APaRP): d = {d_apa:.4f}")

dL_apa, dU_apa, k_used_apa, k_note_apa = get_dw_critical_values_for_n200(best_n)
print(f"\nCritical values (alpha={DW_ALPHA}):")
print(f"  d_L = {dL_apa:.3f}, d_U = {dU_apa:.3f}")

if d_apa < dL_apa:
    dw_apa_conclusion = "Significant positive serial correlation"
    dw_apa_interpretation = "Potential nonlinearity (possibly quadratic) remains"
    dw_apa_detail = f"d = {d_apa:.4f} < d_L = {dL_apa:.3f}"
    dw_apa_result = "Nonliner"
elif d_apa > dU_apa:
    dw_apa_conclusion = "No significant serial correlation"
    dw_apa_interpretation = "Linear + quadratic term adequately modeled"
    dw_apa_detail = f"d = {d_apa:.4f} > d_U = {dU_apa:.3f}"
    dw_apa_result = "Liner"
else:
    dw_apa_conclusion = "Inconclusive zone"
    dw_apa_interpretation = "Result is uncertain"
    dw_apa_detail = f"{dL_apa:.3f} ≤ {d_apa:.4f} ≤ {dU_apa:.3f}"
    dw_apa_result = "Inconclusive"

print(f"\nDW Conclusion: {dw_apa_conclusion}")
print(f"Interpretation: {dw_apa_interpretation}")
print(f"Basis: {dw_apa_detail}")
print(f"DW result: {dw_apa_result}")

# Runs test on APaRP
runs_apa, n1_apa, n2_apa, z_apa, p_apa, mu_apa, sigma_apa = runs_test(aug_partial_sorted)
print(f"\nRuns test (APaRP):")
print(f"Positive residuals n1 = {n1_apa}")
print(f"Negative residuals n2 = {n2_apa}")
print(f"Number of runs R = {runs_apa}")

if z_apa is not None:
    print(f"Expected mean mu = {mu_apa:.2f}")
    print(f"Expected std sigma = {sigma_apa:.4f}")
    print(f"Standardized z = {z_apa:.4f}")
    print(f"Two-tailed p-value = {p_apa:.4f}")
    if abs(z_apa) > 1.96:
        runs_apa_result = "Nonliner"
        runs_apa_conclusion = "Significant non-random pattern"
    else:
        runs_apa_result = "Liner"
        runs_apa_conclusion = "Random pattern"
    print(f"Conclusion: {runs_apa_conclusion}")
    print(f"Runs result: {runs_apa_result}")
else:
    runs_apa_result = "Inconclusive"
    runs_apa_conclusion = "Insufficient counts for normal approximation"
    print(runs_apa_conclusion)
    print(f"Runs result: {runs_apa_result}")

# ==============================================================
# 9. drawing APaRP scatter plot
# ==============================================================
plt.figure(figsize=(8, 5))
plt.scatter(t1, aug_partial, alpha=0.6, edgecolors='k', s=50)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
# LOWESS
sorted_idx_t1 = np.argsort(t1)
t1_sorted = t1[sorted_idx_t1]
aug_partial_sorted_t1 = aug_partial[sorted_idx_t1]
lowess_smoothed_apa = lowess(aug_partial_sorted_t1, t1_sorted, frac=0.4)
plt.plot(lowess_smoothed_apa[:, 0], lowess_smoothed_apa[:, 1], color='blue', linewidth=2, label='LOWESS')
plt.xlabel('First latent variable (t1)')
plt.ylabel('Augmented partial residual')
plt.title(f'APaRP')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(f'APaRP_{CURRENT_DATASET}.png', dpi=300)
plt.show()
# ==============================================================
# 10. Combined summary and final interpretation
# ==============================================================
print(f"\n{'=' * 70}")
print(f"Combined summary with APaRP - {CURRENT_DATASET.upper()} dataset")
print(f"{'=' * 70}")

print(f"RP (Residual Plot):")
print(f"  Durbin-Watson: {dw_result} (d={d:.4f})")
runs_str_rp = f" (p={p:.4f})" if p is not None else ""
print(f"  Runs test: {runs_result} (z={z:.4f}{runs_str_rp})")

print(f"\nAPaRP (first latent variable):")
print(f"  Durbin-Watson: {dw_apa_result} (d={d_apa:.4f})")
runs_str_apa = f" (p={p_apa:.4f})" if p_apa is not None else ""
print(f"  Runs test: {runs_apa_result} (z={z_apa:.4f}{runs_str_apa})")

if dw_result == "Nonliner" or runs_result == "Nonliner" or dw_apa_result == "Nonliner" or runs_apa_result == "Nonliner":
    final_overall = "Nonliner"
elif dw_result == "Liner" and runs_result == "Liner" and dw_apa_result == "Liner" and runs_apa_result == "Liner":
    final_overall = "Liner"
else:
    final_overall = "Inconclusive"

print(f"\nOverall final result (combining RP and APaRP): {final_overall}")
if final_overall == "Inconclusive":
    print("\n Warning: Both RP and APaRP yielded inconclusive results.")
    print("   It is recommended to validate linearity through model performance (e.g., cross-validation, residual plots).")
    print("   Consider comparing linear PLS with a nonlinear method (e.g., LWR, SVM, ANN) for confirmation.")