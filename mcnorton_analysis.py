"""
McNorton et al. (2025) Dataset — Dissertation Analysis Workflow
----------------------------------------------------------------
Data (download from https://doi.org/10.5281/zenodo.16950724):
  - Fire_Info.csv        : metadata for 18 extreme fire events
  - ALL_FIRE_DATA.csv    : antecedent variable anomalies per fire (main file)
  - ALL_REGION_DATA.csv  : regional monthly time series 2003–2025
  - PAL_EAT_DATA.csv     : Palisades + Eaton fires specifically

Variables in the dataset (monthly, ~9 km resolution, ERA5-Land / MODIS):
  vpd       — Vapour Pressure Deficit (atmospheric dryness)
  spei      — Standardised Precipitation-Evapotranspiration Index (3-month)
  sm        — Soil Moisture
  lfmc      — Live Fuel Moisture Content
  dfmc      — Dead Fuel Moisture Content
  evi       — Enhanced Vegetation Index (MODIS)
  lai       — Leaf Area Index (MODIS)
  frp       — Fire Radiative Power (dependent variable proxy)
  biome     — Mediterranean / Desert / Forested Mountain
  lag       — months before fire (0 = fire month, negative = antecedent)
"""

# ============================================================
# 0. SETUP
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Download the files once ──────────────────────────────────
# import urllib.request
# BASE = "https://zenodo.org/records/16950724/files/"
# for f in ["Fire_Info.csv","ALL_FIRE_DATA.csv","ALL_REGION_DATA.csv","PAL_EAT_DATA.csv"]:
#     urllib.request.urlretrieve(BASE + f + "?download=1", f)


# ============================================================
# 1. LOAD & INSPECT
# ============================================================
fire_info   = pd.read_csv("Fire_Info.csv")
all_fire    = pd.read_csv("ALL_FIRE_DATA.csv")
all_region  = pd.read_csv("ALL_REGION_DATA.csv")
pal_eat     = pd.read_csv("PAL_EAT_DATA.csv")

# Convert dates
for df in [all_fire, all_region, pal_eat]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

print("=== Fire_Info ===")
print(fire_info.head(10))
print("\n=== ALL_FIRE_DATA columns ===")
print(all_fire.columns.tolist())
print(all_fire.head(3))
print("\n=== ALL_REGION_DATA columns ===")
print(all_region.columns.tolist())
print(all_region.head(3))


# ============================================================
# 2. FILTER TO LA FIRES (Palisades & Eaton)
# ============================================================
# Identify the fire names/IDs from Fire_Info
print("\n=== Fire names ===")
print(fire_info[["fire_name", "biome", "date"]].to_string())

# Use PAL_EAT_DATA for the LA-specific analysis
# If fire_id or fire_name column exists, filter ALL_FIRE_DATA similarly
la_fires = pal_eat.copy()

# Confirm what columns exist for filtering
print("\nPAL_EAT columns:", la_fires.columns.tolist())


# ============================================================
# 3. TEMPORAL WINDOW ANALYSIS
# ============================================================
# Replicate & extend McNorton's lagged Spearman approach,
# focused specifically on the LA fires and FWI-adjacent variables.

VARS = ["vpd", "spei", "sm", "lfmc", "dfmc", "evi", "lai"]

# -- 3a. Plot standardised anomaly time series for LA fires ------------------
# Each variable's anomaly from -30 months (antecedent) to 0 (fire month)

fig, axes = plt.subplots(len(VARS), 1, figsize=(12, 16), sharex=True)
fig.suptitle("Antecedent Variable Anomalies — Palisades & Eaton Fires\n(months before fire)", 
             fontsize=13, y=1.01)

for ax, var in zip(axes, VARS):
    if var not in la_fires.columns:
        ax.set_visible(False)
        continue
    # Group by lag (months before fire), average across both LA fires
    grouped = la_fires.groupby("lag")[var].mean()
    ax.bar(grouped.index, grouped.values, 
           color=["#d62728" if v > 0 else "#1f77b4" for v in grouped.values],
           alpha=0.8, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(-5, color="grey", linewidth=0.8, linestyle=":")   # ~onset of drying
    ax.axvline(-18, color="grey", linewidth=0.8, linestyle=":")  # ~peak of wet phase
    ax.set_ylabel(var.upper(), fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

axes[-1].set_xlabel("Months Before Fire (lag)", fontsize=10)
axes[-1].invert_xaxis()  # show most antecedent on the left
plt.tight_layout()
plt.savefig("fig1_antecedent_anomalies_LA.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: fig1_antecedent_anomalies_LA.png")


# ============================================================
# 4. LAGGED SPEARMAN CORRELATION ACROSS ALL 18 FIRES
# ============================================================
# Compute Spearman r with FRP for each variable and lag value.

if "frp" not in all_fire.columns:
    print("\n[Note] FRP column not found — adjust column name to match dataset.")
else:
    lags = sorted(all_fire["lag"].unique())
    corr_results = []

    for var in VARS:
        if var not in all_fire.columns:
            continue
        for lag in lags:
            subset = all_fire[all_fire["lag"] == lag][["frp", var]].dropna()
            if len(subset) < 5:
                continue
            r, p = stats.spearmanr(subset["frp"], subset[var])
            corr_results.append({"variable": var, "lag": lag, "r": r, "p": p,
                                 "significant": p < 0.05})

    corr_df = pd.DataFrame(corr_results)

    # -- 4a. Heatmap of Spearman r by variable × lag -------------------------
    pivot = corr_df.pivot(index="variable", columns="lag", values="r")
    sig_pivot = corr_df.pivot(index="variable", columns="lag", values="significant")

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(l) for l in pivot.columns], fontsize=7, rotation=90)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Lag (months before fire)", fontsize=10)
    ax.set_title("Spearman r with FRP — All 18 fires\n(* = p<0.05)", fontsize=11)

    # Mark significant cells
    for i in range(sig_pivot.shape[0]):
        for j in range(sig_pivot.shape[1]):
            if sig_pivot.iloc[i, j]:
                ax.text(j, i, "·", ha="center", va="center", fontsize=14, color="black")

    plt.colorbar(im, ax=ax, label="Spearman r")
    plt.tight_layout()
    plt.savefig("fig2_spearman_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig2_spearman_heatmap.png")

    # -- 4b. Save correlation table ------------------------------------------
    corr_df.to_csv("spearman_correlations_by_lag.csv", index=False)
    print("Saved: spearman_correlations_by_lag.csv")


# ============================================================
# 5. BIOME-STRATIFIED ANALYSIS
# ============================================================

if "biome" in all_fire.columns and "frp" in all_fire.columns:
    biomes = all_fire["biome"].dropna().unique()
    print(f"\nBiomes in dataset: {biomes}")

    fig, axes = plt.subplots(1, len(biomes), figsize=(14, 5), sharey=True)
    fig.suptitle("Spearman r (VPD vs FRP) by Biome — Temporal Window", fontsize=11)

    for ax, biome in zip(axes, biomes):
        subset = all_fire[all_fire["biome"] == biome]
        lags_b = sorted(subset["lag"].unique())
        rs = []
        for lag in lags_b:
            s = subset[subset["lag"] == lag][["frp", "vpd"]].dropna()
            if len(s) < 4:
                rs.append(np.nan)
                continue
            r, _ = stats.spearmanr(s["frp"], s["vpd"])
            rs.append(r)

        ax.plot(lags_b, rs, "o-", color="#d62728", markersize=4, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.axhline(0.3, color="grey", linewidth=0.7, linestyle=":")
        ax.axhline(-0.3, color="grey", linewidth=0.7, linestyle=":")
        ax.set_title(biome, fontsize=9)
        ax.set_xlabel("Lag (months)", fontsize=8)
        ax.invert_xaxis()

    axes[0].set_ylabel("Spearman r (VPD vs FRP)", fontsize=9)
    plt.tight_layout()
    plt.savefig("fig3_biome_vpd_temporal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig3_biome_vpd_temporal.png")


# ============================================================
# 6. CONSTRUCTING LAG FEATURES FOR YOUR OWN MODEL
# ============================================================

def build_feature_matrix(df, vars_of_interest, lag_windows):
    """
    Pivot long-format lag data into a wide feature matrix.

    Parameters
    ----------
    df              : ALL_FIRE_DATA or PAL_EAT_DATA (long format)
    vars_of_interest: list of variable names, e.g. ["vpd","spei","lfmc"]
    lag_windows     : list of lag values to include, e.g. range(-27, 1)

    Returns
    -------
    feature_df : wide DataFrame, one row per fire, one column per var×lag
    """
    subsets = []
    id_col = "fire_name" if "fire_name" in df.columns else "fire_id"

    for var in vars_of_interest:
        if var not in df.columns:
            continue
        pivot = (df[df["lag"].isin(lag_windows)]
                   .pivot_table(index=id_col, columns="lag", values=var))
        pivot.columns = [f"{var}_lag{abs(c)}m" for c in pivot.columns]
        subsets.append(pivot)

    if not subsets:
        return pd.DataFrame()

    feature_df = pd.concat(subsets, axis=1).reset_index()
    return feature_df


# Example: build features for lags 1–27 months prior
lag_windows = list(range(-27, 0))  # months -27 to -1
feature_matrix = build_feature_matrix(all_fire, VARS, lag_windows)
print(f"\nFeature matrix shape: {feature_matrix.shape}")
print(feature_matrix.head(3))
feature_matrix.to_csv("feature_matrix_lag27m.csv", index=False)
print("Saved: feature_matrix_lag27m.csv")


# ============================================================
# 7. CUMULATIVE / ROLLING WINDOW FEATURES
# ============================================================

def add_rolling_features(df, var, windows):
    """
    Given the regional time series (ALL_REGION_DATA), compute rolling means
    of 'var' over each window length and attach as new columns.
    
    Parameters
    ----------
    df      : ALL_REGION_DATA (dated time series)
    var     : variable name
    windows : dict of {label: n_months}, e.g. {"short":3,"medium":12,"long":24}
    """
    df = df.sort_values("date").copy()
    for label, n in windows.items():
        df[f"{var}_roll{n}m"] = df[var].rolling(window=n, min_periods=n).mean()
    return df


windows = {
    "short":  6,   # recent drying signal (~McNorton drying phase)
    "medium": 12,  # seasonal / annual memory
    "long":   24   # hydroclimatic rebound window
}

# Apply to VPD in the regional data (adapt to your biome of interest)
region_vpd = all_region.copy()
if "vpd" in region_vpd.columns:
    region_vpd = add_rolling_features(region_vpd, "vpd", windows)
    print("\nRolling VPD features added:")
    print(region_vpd[["date", "vpd", "vpd_roll6m", "vpd_roll12m", "vpd_roll24m"]].tail(10))


# ============================================================
# 8. SUMMARY STATISTICS BY BIOME
# ============================================================

if "biome" in all_fire.columns:
    summary = (all_fire.groupby("biome")[VARS + ["frp"]]
               .agg(["mean", "std"])
               .round(3))
    print("\n=== Summary statistics by biome ===")
    print(summary)
    summary.to_csv("summary_by_biome.csv")
    print("Saved: summary_by_biome.csv")


# ============================================================
# 9. SANITY CHECKS
# ============================================================
print("\n=== Missing values in ALL_FIRE_DATA ===")
print(all_fire[VARS + ["frp"]].isnull().sum())

print("\n=== Lag range in dataset ===")
print(f"Min lag: {all_fire['lag'].min()} | Max lag: {all_fire['lag'].max()}")

print("\n=== Number of fires per biome ===")
if "biome" in fire_info.columns:
    print(fire_info["biome"].value_counts())
