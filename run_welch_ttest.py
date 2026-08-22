"""
run_welch_ttest.py
------------------------------------------------------------------
Assigns EM-DAT wildfire loss events to CEC Level I ecoregions -> fire-regime
subgroups, then runs Welch's t-test (Test 2) comparing the per-event antecedent
lag between the fuel-limited and flammability-limited subgroups.

Run from the scripts/:cd ~/Desktop/Dissertation/scripts/run_welch_ttest.py
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

# ----------------------------- CONFIG ---------------------------------------
EMDAT_XLSX = "data/public_emdat_custom_request_2026.xlsx"   # <- your renamed file
SHEET      = "EM-DAT Data"                                  # NOT the 'Info' sheet
ECO_L1     = "data/NA_CEC_Eco_Level1.shp"                   # needs .dbf/.shx/.prj too
LAGS_CSV   = "data/per_event_lags.csv"                      # columns: DisNo, lag_months
EXCLUDE_HAWAII = True   # Hawaii is outside the CEC North American ecoregion scheme

# --------------------- ecoregion -> subgroup crosswalk ----------------------
FUEL = {"MEDITERRANEAN CALIFORNIA", "NORTH AMERICAN DESERTS",
        "SOUTHERN SEMIARID HIGHLANDS", "GREAT PLAINS"}
FLAMM = {"NORTHWESTERN FORESTED MOUNTAINS", "MARINE WEST COAST FOREST",
         "TEMPERATE SIERRAS", "TAIGA", "NORTHERN FORESTS", "EASTERN TEMPERATE FORESTS"}
MASK = {"ARCTIC CORDILLERA", "TUNDRA", "HUDSON PLAIN",
        "TROPICAL DRY FORESTS", "TROPICAL WET FORESTS"}

def subgroup(n):
    n = str(n).strip().upper()
    if n in FUEL:  return "fuel_limited"
    if n in FLAMM: return "flammability_limited"
    if n in MASK:  return "mask"
    return "other"

def subtype_class(s):
    s = str(s).strip().lower()
    if "forest" in s: return "forest"
    if "land fire" in s or "brush" in s or "bush" in s or "pasture" in s: return "land"
    return "mixed"   # 'Wildfire (General)' -> mixed

# ------------------------------- load ---------------------------------------
df = pd.read_excel(EMDAT_XLSX, sheet_name=SHEET)
print(f"Loaded {len(df)} events from {EMDAT_XLSX} [{SHEET}]")

# ------------------------- point-in-polygon ---------------------------------
ecoP = gpd.read_file(ECO_L1)                 # projected (Lambert Azimuthal)
ecoP = ecoP[ecoP["NA_L1NAME"] != "WATER"].copy()
eco  = ecoP.to_crs("EPSG:4326")

pts = gpd.GeoDataFrame(
    df.copy(),
    geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),  # raw -180/180 lon
    crs="EPSG:4326")

j = gpd.sjoin(pts, eco[["NA_L1NAME", "geometry"]], how="left", predicate="within")
j = j[~j.index.duplicated(keep="first")]

miss = j["NA_L1NAME"].isna()                 # coastal points -> nearest region
if miss.any():
    near = gpd.sjoin_nearest(pts[miss.values].to_crs(ecoP.crs),
                             ecoP[["NA_L1NAME", "geometry"]], how="left")
    near = near[~near.index.duplicated(keep="first")]
    j.loc[miss, "NA_L1NAME"] = near["NA_L1NAME"].values

j["ecoregion"]     = j["NA_L1NAME"]
j["subgroup"]      = j["ecoregion"].map(subgroup)
j["subtype_class"] = j["Disaster Subtype"].map(subtype_class)

table = pd.DataFrame({
    "DisNo": j["DisNo."], "year": j["Start Year"].astype("Int64"),
    "country": j["Country"], "location": j["Location"].astype(str).str.slice(0, 40),
    "lat": j["Latitude"].round(3), "lon": j["Longitude"].round(3),
    "ecoregion": j["ecoregion"], "subgroup": j["subgroup"],
    "subtype": j["Disaster Subtype"], "subtype_class": j["subtype_class"],
    "total_dmg_k": j["Total Damage ('000 US$)"],
    "insured_k": j["Insured Damage ('000 US$)"],
})

if EXCLUDE_HAWAII:
    hi = table["location"].str.contains("Hawai", case=False, na=False)
    if hi.any():
        print(f"Excluding {int(hi.sum())} Hawaii event(s) (outside CEC NA scheme)")
        table = table[~hi]

table.sort_values(["subgroup", "year"]).to_csv("event_subgroup_assignments.csv", index=False)
loss = table[table["total_dmg_k"].notna()].copy()
loss.to_csv("event_subgroup_assignments_loss.csv", index=False)

print(f"\nWrote event_subgroup_assignments.csv ({len(table)} rows)")
print(f"Wrote event_subgroup_assignments_loss.csv ({len(loss)} loss-bearing rows)")
print("\nLoss-bearing subgroup counts:")
print(loss["subgroup"].value_counts().to_string())

# ===================== TEST 2 - Welch's t-test ==============================
import os
if not os.path.exists(LAGS_CSV):
    print(f"\n[Welch test NOT run] Missing {LAGS_CSV}.")
    print("  Provide a CSV with columns: DisNo, lag_months (one antecedent lag per")
    print("  event, from your Test-1 anomaly / lagged-correlation step). Then re-run.")
    raise SystemExit(0)

lags = pd.read_csv(LAGS_CSV)
m = loss.merge(lags[["DisNo", "lag_months"]], on="DisNo", how="inner")
fl = m.loc[m["subgroup"] == "fuel_limited", "lag_months"].dropna()
fr = m.loc[m["subgroup"] == "flammability_limited", "lag_months"].dropna()
print(f"\nEvents with lags -> fuel_limited: {len(fl)}, flammability_limited: {len(fr)}")

t, p   = stats.ttest_ind(fl, fr, equal_var=False)              # Welch
u, pu  = stats.mannwhitneyu(fl, fr, alternative="two-sided")   # rank companion
nx, ny = len(fl), len(fr)
sp = np.sqrt(((nx-1)*fl.var(ddof=1) + (ny-1)*fr.var(ddof=1)) / (nx+ny-2))
d  = (fl.mean() - fr.mean()) / sp
se = np.sqrt(fl.var(ddof=1)/nx + fr.var(ddof=1)/ny)
dfw = se**4 / ((fl.var(ddof=1)/nx)**2/(nx-1) + (fr.var(ddof=1)/ny)**2/(ny-1))
diff = fl.mean() - fr.mean()
ci = (diff - stats.t.ppf(0.975, dfw)*se, diff + stats.t.ppf(0.975, dfw)*se)

print("\n================ TEST 2: Welch's t-test =================")
print(f"fuel_limited         : n={nx}, mean={fl.mean():.2f} mo, sd={fl.std(ddof=1):.2f}")
print(f"flammability_limited : n={ny}, mean={fr.mean():.2f} mo, sd={fr.std(ddof=1):.2f}")
print(f"Welch t = {t:.3f},  df = {dfw:.1f},  p = {p:.4f}")
print(f"mean difference = {diff:.2f} mo,  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
print(f"Cohen's d = {d:.3f}")
print(f"Mann-Whitney U = {u:.1f},  p = {pu:.4f}  (rank-based companion)")
