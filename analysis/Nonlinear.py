import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
import statsmodels.stats.stattools as sm_stats   # for durbin_watson
import statsmodels.api as sm                      # for OLS in APaRP

# ==============================================================
# 0. Dataset configuration
# ==============================================================

DATASET_CONFIG = {
    "wheat": {
        "path": r"C:\Users\zmzhang\Desktop\A4_processed.xlsx",
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
        "path": r"C:\Users\zmzhang\Desktop\tacator\preprocessed_tecator.xlsx",
        "sheet": "Raw",
        "spectra_cols": (2, -4),
        "target_col": "protein", #moisture, fat, protein
        "n_wavelengths": 100,
        "target_name": "Protein",
    },
}

CURRENT_DATASET = "soil"  # Switch: wheat / soil / tecator
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
# 7. Combined summary (RP only)
# ==============================================================

print(f"\n{'=' * 70}")
print(f"Combined summary (RP) - {CURRENT_DATASET.upper()} dataset")
print(f"{'=' * 70}")

print(f"Durbin-Watson: {dw_result} (d={d:.4f})")
runs_stats_suffix = f", p={p:.4f})" if p is not None else ")"
z_display = f"{z:.4f}" if z is not None else "NA"
print(f"Runs test: {runs_result} (z={z_display}" + runs_stats_suffix)

if dw_result == runs_result:
    final_rp = dw_result
else:
    final_rp = "Inconclusive"

print(f"\nFinal RP-based result: {final_rp}")

# ==============================================================
# 8. APaRP for the first latent variable
# ==============================================================

print(f"\n{'=' * 70}")
print(f"APaRP (first latent variable) - {CURRENT_DATASET.upper()} dataset")
print(f"{'=' * 70}")

# Extract scores (latent variables) from the PLS model
T = pls_best.x_scores_  # shape (n_samples, best_n)

# Use the first latent variable (t1)
t1 = T[:, 0]

# Build extended model: y ~ all linear scores + t1^2
# Design matrix: include all scores (columns of T) and the squared term of t1
X_extended = np.column_stack([T, t1**2])
# Add constant (intercept) for OLS
X_extended_with_const = sm.add_constant(X_extended)

# Fit OLS
model_ext = sm.OLS(y, X_extended_with_const).fit()
e_aug = model_ext.resid  # residuals from extended model

# Extract coefficients: b1 for t1, b11 for t1^2
# Order: [const, score1, score2, ..., score_best_n, t1^2]
b1 = model_ext.params[1]               # coefficient of t1 (first score)
b11 = model_ext.params[-1]              # coefficient of t1^2 (last parameter)

# Compute augmented partial residual
aug_partial = b1 * t1 + b11 * (t1**2) + e_aug

# Sort augmented partial residual by y (same as for RP)
aug_partial_sorted = aug_partial[sort_idx]

print(f"APaRP computed for first latent variable (t1).")
print(f"b1 = {b1:.4f}, b11 = {b11:.4f}")

# Durbin-Watson test on sorted augmented partial residual
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

# Runs test on sorted augmented partial residual
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

    z_crit = 1.96
    if abs(z_apa) > z_crit:
        runs_apa_conclusion = "Significant non-random pattern"
        runs_apa_detail = f"|z| = {abs(z_apa):.4f} > {z_crit}"
        runs_apa_pattern = "Too few runs (trend/positive serial structure)" if z_apa < 0 else "Too many runs (oscillatory structure)"
        runs_apa_result = "Nonliner"
    else:
        runs_apa_conclusion = "Random pattern"
        runs_apa_detail = f"|z| = {abs(z_apa):.4f} <= {z_crit}"
        runs_apa_pattern = "Residual ordering appears random"
        runs_apa_result = "Liner"

    print(f"\nConclusion: {runs_apa_conclusion}")
    print(f"Pattern: {runs_apa_pattern}")
    print(f"Basis: {runs_apa_detail}")
    print(f"Runs result: {runs_apa_result}")
else:
    runs_apa_conclusion = "Insufficient counts for normal approximation"
    runs_apa_result = "Inconclusive"
    print("Insufficient counts for normal approximation.")
    print(f"Runs result: {runs_apa_result}")

# ==============================================================
# 9. Combined summary including APaRP
# ==============================================================

print(f"\n{'=' * 70}")
print(f"Combined summary with APaRP - {CURRENT_DATASET.upper()} dataset")
print(f"{'=' * 70}")

print(f"RP (Residual Plot):")
print(f"  Durbin-Watson: {dw_result} (d={d:.4f})")
runs_str_rp = f" (p={p:.4f})" if p is not None else ""
print(f"  Runs test: {runs_result} (z={z_display}{runs_str_rp})")

print(f"\nAPaRP (first latent variable):")
print(f"  Durbin-Watson: {dw_apa_result} (d={d_apa:.4f})")
runs_str_apa = f" (p={p_apa:.4f})" if p_apa is not None else ""
print(f"  Runs test: {runs_apa_result} (z={z_apa:.4f}{runs_str_apa})")

