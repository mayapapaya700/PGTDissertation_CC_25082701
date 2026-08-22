"""
extract_vpd_lags.py
------------------------------------------------------------------
For each EM-DAT loss event, find the antecedent lag = the timing (months before
the fire) of the LARGEST DRYING ANOMALY in VPD.

VPD sign convention:
  VPD = saturation vapour pressure(T) - actual vapour pressure(Td), in kPa.
  Higher VPD  = drier / more atmospheric moisture demand.

Data:
  ERA5-Land monthly averaged reanalysis, variables
    2m_temperature (t2m) and 2m_dewpoint_temperature (d2m), covering North America

Output: data/per_event_lags.csv  (columns: DisNo, lag_months, + diagnostics)

Run from the Dissertation root:
    cd ~/Desktop/Dissertation
    python3 scripts/extract_vpd_lags.py
"""
import glob
import numpy as np
import pandas as pd
import xarray as xr

# ----------------------------- CONFIG ---------------------------------------
EMDAT_XLSX = "data/public_emdat_custom_request_2026.xlsx"
SHEET      = "EM-DAT Data"
ERA5_GLOB  = "data/era5/*.nc"      # ERA5-Land monthly t2m + d2m
OUT_CSV    = "data/per_event_lags.csv"
MIN_LAG    = 1     # months before fire (start of antecedent window)
MAX_LAG    = 24    # months before fire (end of window); widen toward 27 if wanted
LOSS_ONLY  = True  # process only events with a Total Damage value (the 63)

# --------------------------- VPD from ERA5 ----------------------------------
def vpd_kpa(t2m_K, d2m_K):
    """Vapour pressure deficit in kPa. Higher = drier (Tetens formula)."""
    Tc, Tdc = t2m_K - 273.15, d2m_K - 273.15
    es = 0.6108 * np.exp(17.27 * Tc  / (Tc  + 237.3))   # sat. vp at air temp
    ea = 0.6108 * np.exp(17.27 * Tdc / (Tdc + 237.3))   # actual vp (at dewpoint)
    return es - ea

def pick(ds, candidates):
    for c in candidates:
        if c in ds.variables:
            return c
    raise KeyError(f"None of {candidates} found in dataset. Have: {list(ds.variables)}")

# ------------------------------- load ERA5 ----------------------------------
files = sorted(glob.glob(ERA5_GLOB))
if not files:
    raise SystemExit(f"No ERA5 files at {ERA5_GLOB}. Download 2m temp + 2m dewpoint "
                     f"(ERA5-Land monthly) and save the .nc there.")
ds = xr.open_mfdataset(files, combine="by_coords")

t2m = pick(ds, ["t2m", "2t", "var167"])
d2m = pick(ds, ["d2m", "2d", "var168"])
tname = pick(ds, ["valid_time", "time"])
latn = pick(ds, ["latitude", "lat"])
lonn = pick(ds, ["longitude", "lon"])

vpd = vpd_kpa(ds[t2m], ds[d2m]).rename("vpd")
vpd = vpd.rename({tname: "time", latn: "lat", lonn: "lon"})

# ERA5 longitudes may be 0-360; note the convention so we can convert event lon.
lon_is_0_360 = float(vpd["lon"].min()) >= 0 and float(vpd["lon"].max()) > 180

# ------------------------------- events -------------------------------------
df = pd.read_excel(EMDAT_XLSX, sheet_name=SHEET)
if LOSS_ONLY:
    df = df[df["Total Damage ('000 US$)"].notna()].copy()
print(f"Processing {len(df)} events")

def peak_drying_lag(lat, lon, fyear, fmonth):
    """Return (lag_months, peak_sigma, n_available) of the largest positive
    (drying) VPD anomaly within the antecedent window."""
    lon_q = lon + 360 if (lon_is_0_360 and lon < 0) else lon
    cell = vpd.sel(lat=lat, lon=lon_q, method="nearest").to_series().dropna()
    if cell.empty:
        return np.nan, np.nan, 0
    cell.index = pd.PeriodIndex(cell.index, freq="M")
    # standardise per calendar month -> sigma anomalies
    grp = cell.groupby(cell.index.month)
    anom = (cell - grp.transform("mean")) / grp.transform("std")
    fire = pd.Period(f"{int(fyear)}-{int(fmonth):02d}", freq="M")
    rows = []
    for lag in range(MIN_LAG, MAX_LAG + 1):
        tgt = fire - lag
        if tgt in anom.index:
            rows.append((lag, float(anom.loc[tgt])))
    if not rows:
        return np.nan, np.nan, 0
    lags, vals = zip(*rows)
    i = int(np.nanargmax(vals))           # largest POSITIVE anomaly = driest month
    return lags[i], vals[i], len(rows)

recs = []
for _, r in df.iterrows():
    if pd.isna(r["Latitude"]) or pd.isna(r["Longitude"]) or pd.isna(r["Start Month"]):
        recs.append((r["DisNo."], np.nan, np.nan, r["Start Year"], r["Start Month"], 0))
        continue
    lag, sig, n = peak_drying_lag(r["Latitude"], r["Longitude"],
                                  r["Start Year"], r["Start Month"])
    recs.append((r["DisNo."], lag, sig, r["Start Year"], r["Start Month"], n))

out = pd.DataFrame(recs, columns=["DisNo", "lag_months", "peak_anom_sigma",
                                  "fire_year", "fire_month", "n_months_available"])
out.to_csv(OUT_CSV, index=False)

miss = out["lag_months"].isna().sum()
print(f"\nWrote {OUT_CSV} ({len(out)} rows; {miss} without a lag - check coords/coverage)")
print(out.head(12).to_string(index=False))
print(f"\nmedian lag: {out['lag_months'].median():.1f} months   "
      f"(window {MIN_LAG}-{MAX_LAG} mo before fire)")
