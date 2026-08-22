"""
assign_subgroups_and_welch.py
------------------------------
Enumerate the EM-DAT wildfire loss events, assign each to a CEC/EPA Level I
ecoregion (point-in-polygon), map to a fire-regime subgroup, attach the EM-DAT
subtype (general -> 'mixed'), write the event table, then run Welch's t-test
(Test 2) comparing antecedent-lag distributions between the two subgroups.

DESIGN NOTE: grouping uses ONLY the physical ecoregion classification and the
EM-DAT subtype field. The Australian FDRS is NOT used to reclassify anything -
it is purely an interpretive comparison in the dissertation write-up.

Inputs (edit CONFIG):
  EMDAT_XLSX : the EM-DAT export (.xlsx)
  ECO_L1     : CEC/EPA Level I ecoregion polygons (.shp or .geojson)
  LAGS_CSV   : per-event antecedent lag Li (columns: event_id, lag_months),
               produced by your Test-1 anomaly / lagged-correlation step.

Outputs:
  event_subgroup_assignments.csv   (the full event table)
  Welch + Mann-Whitney results printed to console.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

# ------------------------------- CONFIG -------------------------------------
EMDAT_XLSX = "data/public_emdat_custom_request_2026.xlsx"
ECO_L1     = "data/na_cec_eco_l1/NA_CEC_Eco_Level1.shp"   # CEC Level I polygons
LAGS_CSV   = "data/per_event_lags.csv"      # optional (event_id, lag_months)
OUT_TABLE  = "event_subgroup_assignments.csv"


def apply_delong_filter(df):
    """Replicate the fwi_delong_roc.py event filter so the count matches (55).
    EDIT this to mirror your script exactly - placeholder keeps loss-bearing rows."""
    df = df[df.get("insured_loss").notna() | df.get("total_damage").notna()]
    # e.g. df = df[df["year"] >= 2001]        # drop year-2000 (only 2 events)
    # e.g. df = df[df["cems_covered"] == True] # keep only events inside CEMS grid
    return df


# --------------------- ecoregion -> subgroup crosswalk ----------------------
FUEL_LIMITED = {"MEDITERRANEAN CALIFORNIA", "NORTH AMERICAN DESERTS",
                "SOUTHERN SEMI-ARID HIGHLANDS", "GREAT PLAINS"}
FLAMM_LIMITED = {"NORTHWESTERN FORESTED MOUNTAINS", "MARINE WEST COAST FOREST",
                 "TEMPERATE SIERRAS", "TAIGA", "NORTHERN FORESTS",
                 "EASTERN TEMPERATE FORESTS"}
MASK = {"ARCTIC CORDILLERA", "TUNDRA", "HUDSON PLAIN",
        "TROPICAL DRY FORESTS", "TROPICAL WET FORESTS"}


def region_to_subgroup(name):
    n = str(name).strip().upper()
    if n in FUEL_LIMITED:
        return "fuel_limited"
    if n in FLAMM_LIMITED:
        return "flammability_limited"
    if n in MASK:
        return "mask"
    return "unassigned"


def subtype_to_class(subtype):
    """EM-DAT wildfire subtype -> descriptive class. General -> 'mixed'."""
    s = str(subtype).strip().lower()
    if "forest" in s:
        return "forest"
    if "land fire" in s or "brush" in s or "bush" in s or "pasture" in s:
        return "land"
    return "mixed"   # 'Wildfire (General)' and anything else


# ------------------------------- load ---------------------------------------
emdat = pd.read_excel(EMDAT_XLSX)
emdat.columns = [c.strip() for c in emdat.columns]

# Map EM-DAT export headers -> canonical names (EDIT if your export differs).
COLMAP = {
    "DisNo.": "event_id", "Disaster Subtype": "subtype",
    "Country": "country", "Location": "location", "Start Year": "year",
    "Latitude": "lat", "Longitude": "lon",
    "Insured Damage ('000 US$)": "insured_loss",
    "Total Damage ('000 US$)": "total_damage",
}
emdat = emdat.rename(columns={k: v for k, v in COLMAP.items() if k in emdat.columns})

emdat = apply_delong_filter(emdat)
print(f"Events after filter: {len(emdat)}   (expecting 55)")

missing_xy = emdat["lat"].isna() | emdat["lon"].isna()
if missing_xy.any():
    print(f"\nWARNING: {missing_xy.sum()} event(s) lack lat/lon - assign these "
          f"manually, point-in-polygon will skip them:")
    print(emdat.loc[missing_xy, ["event_id", "country", "location", "year"]]
          .to_string(index=False))

# ------------------------- point-in-polygon ---------------------------------
eco = gpd.read_file(ECO_L1).to_crs("EPSG:4326")
name_field = next((c for c in eco.columns
                   if "L1" in c.upper() and "NAME" in c.upper()), eco.columns[0])

geo = emdat.dropna(subset=["lat", "lon"]).copy()
pts = gpd.GeoDataFrame(
    geo, geometry=gpd.points_from_xy(geo["lon"], geo["lat"]), crs="EPSG:4326")
joined = gpd.sjoin(pts, eco[[name_field, "geometry"]], how="left", predicate="within")
joined = joined.rename(columns={name_field: "ecoregion_L1"})

joined["subgroup"] = joined["ecoregion_L1"].map(region_to_subgroup)
joined["subtype_class"] = joined["subtype"].map(subtype_to_class)

# ----------------------------- write table ----------------------------------
cols = ["event_id", "year", "country", "location", "lat", "lon",
        "ecoregion_L1", "subgroup", "subtype", "subtype_class",
        "insured_loss", "total_damage"]
table = (joined[[c for c in cols if c in joined.columns]]
         .sort_values(["subgroup", "year"]))
table.to_csv(OUT_TABLE, index=False)
print(f"\nWrote {OUT_TABLE} ({len(table)} rows)\n")
print(table.to_string(index=False))
print("\nSubgroup counts:\n", table["subgroup"].value_counts().to_string())

# ===================== TEST 2 - Welch's t-test ==============================
# Grouping is the PHYSICAL ecoregion subgroup. Requires per-event lag Li.
try:
    lags = pd.read_csv(LAGS_CSV)             # columns: event_id, lag_months
    m = table.merge(lags, on="event_id", how="inner")

    fl = m.loc[m["subgroup"] == "fuel_limited", "lag_months"].dropna()
    fr = m.loc[m["subgroup"] == "flammability_limited", "lag_months"].dropna()

    t, p = stats.ttest_ind(fl, fr, equal_var=False)               # Welch
    u, pu = stats.mannwhitneyu(fl, fr, alternative="two-sided")   # rank companion

    nx, ny = len(fl), len(fr)
    sp = np.sqrt(((nx - 1) * fl.var(ddof=1) + (ny - 1) * fr.var(ddof=1)) / (nx + ny - 2))
    d = (fl.mean() - fr.mean()) / sp
    se = np.sqrt(fl.var(ddof=1) / nx + fr.var(ddof=1) / ny)
    dfw = se**4 / ((fl.var(ddof=1) / nx)**2 / (nx - 1)
                   + (fr.var(ddof=1) / ny)**2 / (ny - 1))
    diff = fl.mean() - fr.mean()
    tc = stats.t.ppf(0.975, dfw)
    ci = (diff - tc * se, diff + tc * se)

    print("\n================ TEST 2: Welch's t-test =================")
    print(f"fuel_limited         : n={nx}, mean={fl.mean():.2f} mo, sd={fl.std(ddof=1):.2f}")
    print(f"flammability_limited : n={ny}, mean={fr.mean():.2f} mo, sd={fr.std(ddof=1):.2f}")
    print(f"Welch t = {t:.3f},  df = {dfw:.1f},  p = {p:.4f}")
    print(f"mean difference = {diff:.2f} mo,  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    print(f"Cohen's d = {d:.3f}")
    print(f"Mann-Whitney U = {u:.1f},  p = {pu:.4f}  (rank-based companion)")
    print("mixed / mask / unassigned events are excluded from the test")
except FileNotFoundError:
    print(f"\n[Welch test skipped] Add {LAGS_CSV} (event_id, lag_months) from "
          f"your Test-1 anomaly step, then re-run to get Test 2.")
