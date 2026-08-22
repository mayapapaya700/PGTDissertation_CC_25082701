"""
ROC/AUC Analysis: FWI Variables as Predictors of Wildfire-Induced Loss
Datasets: EM-DAT North America Wildfires (USA + Canada, 2000-2026); CEMS 
Fire Danger Reanalysis (Canadian FWI System components, ECMWF/Copernicus)
Author: Maya Lopansri - MSc Climate Change, King's College London

Builds on fwi_roc_auc_emdat.py, replacing the synthetic FWI stub with a real
merge against the CEMS NetCDF archive under data/cems_fwi_downloads/.

"""

# -- 0. Imports --------------------------------------------------------------
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import PchipInterpolator

try:
    import xarray as xr
except ImportError as e:
    raise ImportError(
        "This script requires xarray + netCDF4 to read the CEMS .nc files.\n"
        "Install with: pip install xarray netCDF4"
    ) from e

warnings.filterwarnings("ignore")

# -- 1. Paths & constants ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _find_emdat_path() -> Path:
    """
    Locate the EM-DAT export under data/. Resolved by glob (not a hardcoded
    filename) because the source file has already been renamed once by hand
    (public_emdat_custom_request_2026-06-28_....xlsx -> ...2026.xlsx); a
    fixed name silently breaks the whole pipeline the next time that happens.
    Picks the most recently modified match and skips Excel's "~$" lock files.
    """
    candidates = [
        p for p in (BASE_DIR / "data" / "raw").glob("public_emdat_custom_request*.xlsx")
        if not p.name.startswith("~$")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No public_emdat_custom_request*.xlsx found under {BASE_DIR / 'data' / 'raw'}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


EMDAT_PATH = _find_emdat_path()
CEMS_DIR   = BASE_DIR / "data" / "raw" / "cems_fwi_downloads"

LOSS_COL     = "Total Damage, Adjusted ('000 US$)"
INSURED_COL  = "Insured Damage, Adjusted ('000 US$)"

CEMS_VAR_MAP = {
    "FFMC": "ffmcode",   # Fine Fuel Moisture Code
    "DMC":  "dufmcode",  # Duff Moisture Code
    "DC":   "drtcode",   # Drought Code
    "ISI":  "infsinx",   # Initial Spread Index
    "BUI":  "fbupinx",   # Buildup Index
    "FWI":  "fwinx",     # Fire Weather Index
    "DSR":  "fdsrte",    # Daily Severity Rating
}

SINGLE_VARS    = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "DSR"]
COMPOSITE_VARS = ["FWI_x_ISI", "BUI_x_FFMC", "DC_x_DMC", "ISI_sq", "FWI_event_mean", "FWI_event_max"]
ALL_VARS       = SINGLE_VARS + COMPOSITE_VARS

TIME_TOLERANCE_DAYS = 45  # allow nearest-available CEMS date if an event falls just outside a file's window

_FULL_DF     = None  # cached by build_feature_matrix() for reuse by this script's own analysis section
_EXCLUDED_DF = None  # cached by build_feature_matrix(): every event dropped, and why


def get_excluded_events():
    """
    Return the DataFrame of EM-DAT events excluded from the modelling set,
    with a reason per event. Populated by the most recent build_feature_matrix()
    call; None if that hasn't run yet.
    """
    return _EXCLUDED_DF


def build_feature_matrix():
    """
    Load EM-DAT + CEMS, match each event to its FWI grid data, engineer
    composite predictors, and return (X, y) for the high-loss classification
    task.

    X columns are the raw (unscaled, unfitted) FWI variable values — exactly
    SINGLE_VARS + COMPOSITE_VARS — so callers can standardise/fit however
    they like. y is the binary high_loss label (top quartile of Total
    Damage, Adjusted).

    The full merged DataFrame (original EM-DAT columns + engineered FWI
    variables, restricted to the events actually used for modelling) is
    cached in the module-level `_FULL_DF` afterwards. Every event dropped
    along the way (no costed damage figure, or no trustworthy CEMS match)
    is recorded in `_EXCLUDED_DF` — see get_excluded_events().
    """
    global _FULL_DF, _EXCLUDED_DF
    excluded_rows = []

    # -- 2. Load EM-DAT Excel export -------------------------------------------
    df = pd.read_excel(EMDAT_PATH, sheet_name="EM-DAT Data")
    print(f"Loaded {len(df)} rows x {df.shape[1]} columns from EM-DAT")

    # -- 3. Clean & filter ------------------------------------------------------
    no_damage = df[df[LOSS_COL].isna()]
    for _, row in no_damage.iterrows():
        excluded_rows.append({
            "DisNo.": row["DisNo."], "Country": row["Country"],
            "Start Year": row["Start Year"], "Disaster Subtype": row["Disaster Subtype"],
            "stage": "damage_filter",
            "reason": "No costed Total Damage, Adjusted figure in EM-DAT",
            "detail": "",
        })

    df = df[df[LOSS_COL].notna()].copy()
    print(f"After filtering to events with Total Damage (adjusted): {len(df)} rows")

    df["start_year"]  = df["Start Year"].astype(int)
    df["start_month"] = pd.to_numeric(df["Start Month"], errors="coerce").fillna(1).astype(int)
    df["start_day"]   = pd.to_numeric(df["Start Day"],   errors="coerce").fillna(1).astype(int)
    df["event_date"]  = pd.to_datetime(
        dict(year=df["start_year"], month=df["start_month"], day=df["start_day"]),
        errors="coerce",
    )

    df["end_year"]  = pd.to_numeric(df["End Year"],  errors="coerce").fillna(df["start_year"]).astype(int)
    df["end_month"] = pd.to_numeric(df["End Month"], errors="coerce").fillna(df["start_month"]).astype(int)
    df["end_day"]   = pd.to_numeric(df["End Day"],   errors="coerce").fillna(df["start_day"]).astype(int)
    df["event_end_date"] = pd.to_datetime(
        dict(year=df["end_year"], month=df["end_month"], day=df["end_day"]),
        errors="coerce",
    )
    # guard against malformed end < start
    df.loc[df["event_end_date"] < df["event_date"], "event_end_date"] = df["event_date"]

    df["event_label"] = df["DisNo."].astype(str) + " - " + df["Event Name"].fillna("").str[:30]

    # CEMS longitude convention is 0-360 east
    df["lon360"] = df["Longitude"].apply(lambda x: x if x >= 0 else x + 360)

    print("\nLoss column summary (adjusted, '000 US$):")
    print(df[LOSS_COL].describe().round(0))
    print(f"\nInsured Damage available for {df[INSURED_COL].notna().sum()} events "
          f"({df[INSURED_COL].notna().mean()*100:.0f}%)")

    # -- 4. Binary outcome: top-quartile total damage = "high-loss event" -----
    q75 = df[LOSS_COL].quantile(0.75)
    df["high_loss"] = (df[LOSS_COL] >= q75).astype(int)
    print(f"\nHigh-loss threshold (75th pctile): ${q75/1e3:,.0f}M")
    print(f"High-loss events: {df['high_loss'].sum()} / {len(df)}")

    # -- 5. Index CEMS NetCDF files ---------------------------------------------
    nc_files = sorted(CEMS_DIR.glob("*/*.nc"))
    print(f"\nIndexing {len(nc_files)} CEMS NetCDF files under {CEMS_DIR.name}/ ...")

    index_rows = []
    for f in nc_files:
        with xr.open_dataset(f) as ds:
            index_rows.append({
                "file": f,
                "lat0": float(ds.latitude.min()),  "lat1": float(ds.latitude.max()),
                "lon0": float(ds.longitude.min()), "lon1": float(ds.longitude.max()),
                "t0":   pd.Timestamp(ds.valid_time.min().values),
                "t1":   pd.Timestamp(ds.valid_time.max().values),
            })
    cems_index = pd.DataFrame(index_rows)
    cems_index["area"] = (cems_index["lat1"] - cems_index["lat0"]) * (cems_index["lon1"] - cems_index["lon0"])
    print(f"Indexed {len(cems_index)} files spanning "
          f"{cems_index['t0'].min().date()} to {cems_index['t1'].max().date()}")

    # -- 6. Match each event to a CEMS file & extract FWI variables -----------
    NEAR_FALLBACK_DEG = 2.0  # ~220km, matching the largest buffer tier already used for these downloads

    def _bbox_distance(lat, lon360, r):
        """0 if (lat, lon360) is inside r's bbox, else distance in degrees to its nearest edge."""
        dlat = max(r.lat0 - lat, 0.0, lat - r.lat1)
        dlon = max(r.lon0 - lon360, 0.0, lon360 - r.lon1)
        return (dlat ** 2 + dlon ** 2) ** 0.5

    def match_cems_file(lat, lon360, date):
        """
        Return (best-matching CEMS index row, tier) or (None, None).
        Tier "exact": the file's bbox contains the point (original behaviour).
        Tier "nearby": no downloaded file's bbox contains the point, but one
        within NEAR_FALLBACK_DEG does exist within the time tolerance - used
        for multi-state/pan-provincial events whose EM-DAT centroid falls in
        a gap between the named sub-location boxes that were downloaded for
        them (e.g. a province not explicitly listed in the EM-DAT Location text).
        """
        cand = cems_index[
            (cems_index.lat0 <= lat) & (cems_index.lat1 >= lat) &
            (cems_index.lon0 <= lon360) & (cems_index.lon1 >= lon360)
        ].copy()
        if not cand.empty:
            cand["gap_days"] = cand.apply(
                lambda r: 0 if r.t0 <= date <= r.t1
                else min(abs((date - r.t0).days), abs((date - r.t1).days)),
                axis=1,
            )
            cand = cand[cand["gap_days"] <= TIME_TOLERANCE_DAYS]
            if not cand.empty:
                return cand.sort_values(["gap_days", "area"]).iloc[0], "exact"

        near = cems_index.copy()
        near["gap_days"] = near.apply(
            lambda r: 0 if r.t0 <= date <= r.t1
            else min(abs((date - r.t0).days), abs((date - r.t1).days)),
            axis=1,
        )
        near = near[near["gap_days"] <= TIME_TOLERANCE_DAYS]
        if near.empty:
            return None, None
        near["dist"] = near.apply(lambda r: _bbox_distance(lat, lon360, r), axis=1)
        near = near[(near["dist"] > 0) & (near["dist"] <= NEAR_FALLBACK_DEG)]
        if near.empty:
            return None, None
        return near.sort_values(["dist", "gap_days", "area"]).iloc[0], "nearby"

    def describe_no_match(lat, lon360, date):
        """
        For an event with no accepted CEMS match (neither an "exact" bbox-
        containment match nor a "nearby" fallback within NEAR_FALLBACK_DEG),
        explain why: no extract spatially close enough exists at all, or the
        closest one is too far away in time to trust.
        """
        near = cems_index.copy()
        near["dist"] = near.apply(lambda r: _bbox_distance(lat, lon360, r), axis=1)
        near = near[near["dist"] <= NEAR_FALLBACK_DEG]
        if near.empty:
            return (
                f"No downloaded CEMS extract is within {NEAR_FALLBACK_DEG} deg of this "
                f"event's coordinates at all."
            )
        near["gap_days"] = near.apply(
            lambda r: 0 if r.t0 <= date <= r.t1
            else min(abs((date - r.t0).days), abs((date - r.t1).days)),
            axis=1,
        )
        best = near.sort_values("gap_days").iloc[0]
        years = best["gap_days"] / 365.25
        return (
            f"Nearest spatially-close ({best['dist']:.2f} deg) CEMS extract is "
            f"{best['gap_days']} days (~{years:.1f} yr) away in time (covers "
            f"{best['t0'].date()} to {best['t1'].date()}) - too far from the event "
            f"date to represent its fire-weather conditions (tolerance is "
            f"{TIME_TOLERANCE_DAYS} days)."
        )

    def extract_fwi(file_row, lat, lon360, start_date, end_date):
        """Extract ignition-day FWI components + event-window FWI mean/max."""
        with xr.open_dataset(file_row["file"]) as ds:
            used_ensemble_mean = False
            if "number" in ds.dims:
                # A handful of downloads were pulled from the ensemble product
                # (10 members) instead of the deterministic "Consolidated"
                # reanalysis used for every other file. Collapse to the
                # ensemble mean so extraction still works; flagged in the
                # returned dict so it's visible downstream, not silent.
                ds = ds.mean(dim="number", keep_attrs=True)
                used_ensemble_mean = True

            point = ds.sel(latitude=lat, longitude=lon360, method="nearest")
            t0, t1 = pd.Timestamp(ds.valid_time.min().values), pd.Timestamp(ds.valid_time.max().values)

            ignition_date = min(max(start_date, t0), t1)
            ignition = point.sel(valid_time=ignition_date, method="nearest")

            result = {label: float(ignition[code].values) for label, code in CEMS_VAR_MAP.items()}

            w0 = max(start_date, t0)
            w1 = min(end_date, t1) if pd.notna(end_date) else w0
            if w1 < w0:
                w1 = w0
            window = point.sel(valid_time=slice(w0, w1))["fwinx"].values
            result["FWI_event_mean"] = float(np.nanmean(window)) if window.size else result["FWI"]
            result["FWI_event_max"]  = float(np.nanmax(window))  if window.size else result["FWI"]
            result["_used_ensemble_mean"] = used_ensemble_mean
        return result

    extracted, unmatched, fallback_matched = [], [], []
    for _, row in df.iterrows():
        match, tier = match_cems_file(row["Latitude"], row["lon360"], row["event_date"])
        if match is None:
            unmatched.append(row["DisNo."])
            excluded_rows.append({
                "DisNo.": row["DisNo."], "Country": row["Country"],
                "Start Year": row["Start Year"], "Disaster Subtype": row["Disaster Subtype"],
                "stage": "cems_match",
                "reason": "No CEMS FWI grid data within the accepted spatial/temporal window",
                "detail": describe_no_match(row["Latitude"], row["lon360"], row["event_date"]),
            })
            extracted.append({})
            continue
        if tier == "nearby":
            fallback_matched.append(row["DisNo."])
        result = extract_fwi(match, row["Latitude"], row["lon360"],
                              row["event_date"], row["event_end_date"])
        result["cems_match_tier"] = tier
        extracted.append(result)

    fwi_df = pd.DataFrame(extracted, index=df.index)
    df = pd.concat([df, fwi_df], axis=1)

    n_matched = len(df) - len(unmatched)
    print(f"\nMatched {n_matched} / {len(df)} events to CEMS FWI grid data "
          f"(spatial containment + {TIME_TOLERANCE_DAYS}-day time tolerance)")
    if unmatched:
        print(f"Unmatched events (no CEMS coverage for this location/date): {unmatched}")
    if fallback_matched:
        print(f"Matched via nearby-box fallback (bbox didn't contain the centroid, but a "
              f"CEMS extract within {NEAR_FALLBACK_DEG} deg did): {fallback_matched}")
    if "_used_ensemble_mean" in df.columns and df["_used_ensemble_mean"].any():
        ens = df.loc[df["_used_ensemble_mean"] == True, "DisNo."].tolist()
        print(f"Used ensemble-mean (not deterministic reanalysis) for: {ens} - "
              f"that download was pulled from the 10-member ensemble product instead of "
              f"'Consolidated'; values are the ensemble mean, not the deterministic run "
              f"used for every other event.")

    # -- 7. Engineer composite FWI predictors ----------------------------------
    # Theoretically grounded in Canadian FWI System fire-behaviour interactions:
    #   FWI x ISI  - overall fire danger combined with instantaneous spread rate
    #   BUI x FFMC - drought-accumulated fuels x surface-moisture depletion
    #   DC x DMC   - deep-layer x mid-layer drying interaction
    #   ISI^2      - non-linear acceleration in spread rate
    # Plus two temporal composites unlocked by real (non-synthetic) daily data:
    #   FWI_event_mean / FWI_event_max - average / peak fire danger across the
    #   event's full active window, vs. the single ignition-day FWI value.
    df["FWI_x_ISI"]  = df["FWI"] * df["ISI"]
    df["BUI_x_FFMC"] = df["BUI"] * df["FFMC"]
    df["DC_x_DMC"]   = df["DC"]  * df["DMC"]
    df["ISI_sq"]     = df["ISI"] ** 2

    # Drop events without usable FWI data before modelling. Normally this is
    # exactly the `unmatched` set above; anything extra here means a matched
    # file produced a partial/NaN extraction (e.g. a variable missing from
    # that particular grid cell) rather than a clean no-match.
    already_logged = {r["DisNo."] for r in excluded_rows if r["stage"] == "cems_match"}
    still_missing = df[df[ALL_VARS].isna().any(axis=1) & ~df["DisNo."].isin(already_logged)]
    for _, row in still_missing.iterrows():
        excluded_rows.append({
            "DisNo.": row["DisNo."], "Country": row["Country"],
            "Start Year": row["Start Year"], "Disaster Subtype": row["Disaster Subtype"],
            "stage": "cems_extract",
            "reason": "CEMS file matched but produced NaN for one or more FWI variables",
            "detail": "",
        })

    n_before = len(df)
    df = df.dropna(subset=ALL_VARS).copy()
    print(f"\nDropped {n_before - len(df)} events with missing FWI data "
          f"-> {len(df)} events used for modelling "
          f"({df['high_loss'].sum()} high-loss / {len(df) - df['high_loss'].sum()} lower-loss)")

    if df["high_loss"].nunique() < 2:
        raise RuntimeError(
            "high_loss outcome is not both classes after dropping unmatched events - "
            "cannot fit a classifier. Widen TIME_TOLERANCE_DAYS or check CEMS coverage."
        )

    _EXCLUDED_DF = pd.DataFrame(excluded_rows)
    print(f"\n-- Excluded events ({len(_EXCLUDED_DF)} total) -----------------------------")
    for stage, label in [("damage_filter", "No costed damage figure"),
                          ("cems_match", "No CEMS coverage close enough"),
                          ("cems_extract", "CEMS match but NaN extraction")]:
        stage_rows = _EXCLUDED_DF[_EXCLUDED_DF["stage"] == stage] if len(_EXCLUDED_DF) else _EXCLUDED_DF
        if len(stage_rows):
            print(f"  {label} ({len(stage_rows)}): {stage_rows['DisNo.'].tolist()}")

    _FULL_DF = df
    X = df[ALL_VARS].copy()
    y = df["high_loss"].values
    return X, y


def smooth_roc(fpr, tpr, n=300):
    """Monotone-interpolate a step-function ROC curve into a smooth, rounded curve."""
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    uniq_fpr, first_idx = np.unique(fpr, return_index=True)
    # at repeated fpr values (vertical jumps) keep the highest tpr reached
    uniq_tpr = np.array([tpr[fpr == f].max() for f in uniq_fpr])
    if len(uniq_fpr) < 3:
        return fpr, tpr
    interpolator = PchipInterpolator(uniq_fpr, uniq_tpr)
    fpr_smooth = np.linspace(0, 1, n)
    tpr_smooth = np.clip(interpolator(fpr_smooth), 0, 1)
    tpr_smooth = np.maximum.accumulate(tpr_smooth)  # guard against interpolation dips
    return fpr_smooth, tpr_smooth


def _run_analysis():
    """Sections 8-11: CV AUC, ROC/ranking plots, Insured Damage secondary outcome."""
    X, y = build_feature_matrix()
    df = _FULL_DF

    # -- 8. Cross-validated AUC per variable -------------------------------------
    n_splits = min(5, df["high_loss"].value_counts().min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}
    for var in ALL_VARS:
        Xv = df[[var]].values
        model_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, solver="lbfgs")),
        ])
        scores = cross_val_score(model_pipe, Xv, y, cv=skf, scoring="roc_auc")
        results[var] = {"mean_auc": scores.mean(), "std_auc": scores.std(), "scores": scores}

    summary = (
        pd.DataFrame({v: {"AUC (mean)": r["mean_auc"], "AUC (std)": r["std_auc"]}
                      for v, r in results.items()})
        .T
        .sort_values("AUC (mean)", ascending=False)
    )
    print(f"\n-- Cross-Validated AUC by Variable ({n_splits}-fold) ------------------------")
    print(summary.round(3).to_string())

    # -- 9. Plot A: ROC curves (single vs composite) -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    palette   = cm.tab10(np.linspace(0, 1, max(len(SINGLE_VARS), len(COMPOSITE_VARS))))

    for ax, var_group, title in zip(
        axes,
        [SINGLE_VARS, COMPOSITE_VARS],
        ["Single FWI Components", "Composite FWI Variables"],
    ):
        for var, color in zip(var_group, palette):
            X_full  = StandardScaler().fit_transform(df[[var]])
            clf     = LogisticRegression(max_iter=1000).fit(X_full, y)
            y_prob  = clf.predict_proba(X_full)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            fpr_s, tpr_s = smooth_roc(fpr, tpr)
            cv_auc  = results[var]["mean_auc"]
            ax.plot(fpr_s, tpr_s, color=color, lw=2.2,
                    solid_capstyle="round", solid_joinstyle="round",
                    label=f"{var}  (CV AUC = {cv_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.45,
                solid_capstyle="round", label="Random (AUC = 0.50)")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.5, loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"ROC Curves: FWI Variables as Predictors of High Wildfire Loss\n"
        f"EM-DAT North America Wildfires x CEMS Fire Danger Reanalysis (n={len(df)})",
        fontsize=12, y=1.03,
    )
    plt.tight_layout()
    plt.savefig("roc_curves_fwi_cems.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved -> roc_curves_fwi_cems.png")

    # -- 10. Plot B: AUC ranking bar chart ----------------------------------------
    vars_sorted = summary.index.tolist()
    auc_vals    = summary["AUC (mean)"].values
    auc_std     = summary["AUC (std)"].values
    bar_colors  = ["#2166ac" if v in SINGLE_VARS else "#d6604d" for v in vars_sorted]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.barh(
        vars_sorted[::-1], auc_vals[::-1], xerr=auc_std[::-1],
        color=bar_colors[::-1], edgecolor="white", height=0.6,
        error_kw={"elinewidth": 1.6, "capsize": 4},
    )
    ax.axvline(0.5, color="grey", linestyle="--", lw=1.3, label="Random classifier (AUC = 0.50)")
    ax.set_xlabel(f"Mean Cross-Validated AUC ({n_splits}-fold stratified)", fontsize=11)
    ax.set_title(
        f"FWI Variable Ranking - Predictive Power for High Wildfire Loss\n"
        f"EM-DAT x CEMS Fire Danger Reanalysis (n={len(df)})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(
        handles=[
            Patch(facecolor="#2166ac", label="Single FWI component"),
            Patch(facecolor="#d6604d", label="Composite variable"),
            plt.Line2D([0], [0], color="grey", linestyle="--", lw=1.3,
                       label="Random classifier (AUC = 0.50)"),
        ],
        fontsize=9,
        loc="lower right",
    )
    ax.set_xlim([0.3, 1.0])
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig("auc_ranking_fwi_cems.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved -> auc_ranking_fwi_cems.png")

    # -- 11. Optional: second outcome using Insured Damage ------------------------
    insured_df = df[df[INSURED_COL].notna()].copy()
    if len(insured_df) >= 10 and insured_df[LOSS_COL].notna().sum() >= 10:
        i_threshold = insured_df[INSURED_COL].quantile(0.75)
        insured_df["high_insured_loss"] = (insured_df[INSURED_COL] >= i_threshold).astype(int)
        if insured_df["high_insured_loss"].nunique() == 2:
            y_ins = insured_df["high_insured_loss"].values
            n_splits_ins = min(5, insured_df["high_insured_loss"].value_counts().min())
            skf_ins = StratifiedKFold(n_splits=n_splits_ins, shuffle=True, random_state=42)
            ins_results = {}
            for var in ALL_VARS:
                Xv = insured_df[[var]].values
                model_pipe = Pipeline([("scaler", StandardScaler()),
                                        ("clf", LogisticRegression(max_iter=1000))])
                scores = cross_val_score(model_pipe, Xv, y_ins, cv=skf_ins, scoring="roc_auc")
                ins_results[var] = scores.mean()
            ins_summary = pd.Series(ins_results, name="AUC (insured loss)").sort_values(ascending=False)
            print(f"\n-- AUC using Insured Damage outcome (n={len(insured_df)}) --")
            print(ins_summary.round(3).to_string())
        else:
            print(f"\nInsured Damage outcome skipped - only one class present after thresholding.")
    else:
        print(f"\nInsured Damage outcome skipped - only {len(insured_df)} matched events "
              f"with insured figures (need >=10 for cross-validation).")

    print("\nDone.")


if __name__ == "__main__":
    _run_analysis()
