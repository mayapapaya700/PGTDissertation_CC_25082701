"""
fwi_delong_roc.py
=================
Companion module to `fwi_roc_auc_emdat_cems.py`.
Author: Maya Lopansri

"""

from __future__ import annotations

import  warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------
# CONFIG — match these to fwi_roc_auc_emdat_cems.py
# --------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5

SINGLE_COMPONENTS = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "DSR"]
COMPOSITE_VARS = [
    "FWI_x_ISI", "BUI_x_FFMC", "DC_x_DMC",
    "ISI_sq", "FWI_event_mean", "FWI_event_max",
]


# ==========================================================================
# PART 1 — FAST DELONG
# ==========================================================================
# Reference:
#   Sun, X. & Xu, W. (2014). Fast implementation of DeLong's algorithm for
#   comparing the areas under correlated receiver operating characteristic
#   curves. IEEE Signal Processing Letters, 21(11), 1389-1393.

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks, handling ties by averaging (required for tied FWI values)."""
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)

    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(sorted_scores: np.ndarray, n_pos: int):
    """
    Core DeLong computation.

    Parameters
    ----------
    sorted_scores : (k, N) array
        k predictors x N observations, columns ordered so that the first
        `n_pos` columns are the positive class.
    n_pos : int
        Number of positive-class observations.

    Returns
    -------
    aucs : (k,) array
    cov  : (k, k) array
        DeLong covariance matrix of the AUC estimates.
    """
    m = n_pos
    n = sorted_scores.shape[1] - m
    k = sorted_scores.shape[0]

    if m == 0 or n == 0:
        raise ValueError("DeLong requires at least one positive and one negative case.")

    positive = sorted_scores[:, :m]
    negative = sorted_scores[:, m:]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(sorted_scores[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n

    # Structural components (placement values)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    # np.cov collapses to a scalar for k == 1; force 2-D
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n

    return aucs, cov


def _structure(y_true: np.ndarray, scores: np.ndarray):
    """Reorder columns so positives come first. `scores` is (k, N)."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="mergesort")
    return scores[:, order], int(y_true.sum())


def delong_auc_variance(y_true, y_score):
    """AUC and its DeLong variance for a single predictor."""
    scores = np.asarray(y_score, dtype=float).reshape(1, -1)
    sorted_scores, n_pos = _structure(y_true, scores)
    aucs, cov = _fast_delong(sorted_scores, n_pos)
    return float(aucs[0]), float(cov[0, 0])


def delong_auc_ci(y_true, y_score, alpha: float = 0.05):
    """
    Two-sided DeLong confidence interval for a single AUC.
    """
    auc, var = delong_auc_variance(y_true, y_score)
    se = np.sqrt(var)
    z = stats.norm.ppf(1 - alpha / 2)
    lo, hi = auc - z * se, auc + z * se
    return auc, float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1)), float(se)


def delong_test(y_true, score_a, score_b):
    """
    Paired DeLong test: H0 = AUC(a) == AUC(b).

    Returns (auc_a, auc_b, z, p_two_sided).
    """
    scores = np.vstack([
        np.asarray(score_a, dtype=float),
        np.asarray(score_b, dtype=float),
    ])
    sorted_scores, n_pos = _structure(y_true, scores)
    aucs, cov = _fast_delong(sorted_scores, n_pos)

    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]

    if var_diff <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0

    z = (aucs[0] - aucs[1]) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def holm_bonferroni(pvals, alpha: float = 0.05):
    """
    Holm-Bonferroni step-down correction, controlling family wise error rate.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)

    return adj, adj <= alpha


# ==========================================================================
# PART 2 — OUT-OF-FOLD PREDICTION HARVESTING
# ==========================================================================

@dataclass
class OOFResult:
    """Out-of-fold predictions and per-fold ROC data for one predictor."""
    name: str
    oof_pred: np.ndarray                     # (n,) pooled out-of-fold scores
    fold_ids: np.ndarray                     # (n,) which fold each obs was held out in
    fold_aucs: list = field(default_factory=list)
    per_fold_roc: list = field(default_factory=list)  # list of (fpr, tpr)

    @property
    def mean_fold_auc(self) -> float:
        return float(np.mean(self.fold_aucs))

    @property
    def std_fold_auc(self) -> float:
        return float(np.std(self.fold_aucs, ddof=1))


def _make_estimator():
    """Univariate logistic regression, scaled. Matches fwi_roc_auc_emdat_cems.py."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(solver="lbfgs", max_iter=1000)),
    ])


def collect_oof_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    predictors: list[str],
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    score_mode: str = "raw",
) -> dict[str, OOFResult]:
    """
    Run stratified k-fold once, univariately per predictor, and retain the
    out-of-fold scores.

    score_mode
    ----------
    "raw"   : score = the standardised variable value itself.

    "model" : score = logistic-regression predicted probability, fitted on
              the training folds. The model ABSORBS THE SIGN: a variable that
              anti-predicts gets a negative coefficient and its AUC flips
              above 0.5. 
    """
    if score_mode not in {"raw", "model"}:
        raise ValueError("score_mode must be 'raw' or 'model'")

    y = np.asarray(y).astype(int)
    n = len(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(np.zeros(n), y))

    results: dict[str, OOFResult] = {}

    for name in predictors:
        if name not in X.columns:
            warnings.warn(f"Predictor '{name}' not in X; skipping.")
            continue

        col = X[[name]].to_numpy(dtype=float)
        oof = np.full(n, np.nan, dtype=float)
        fold_ids = np.full(n, -1, dtype=int)
        fold_aucs, per_fold_roc = [], []

        for fold_i, (tr, te) in enumerate(splits):
            if score_mode == "model":
                est = _make_estimator()
                est.fit(col[tr], y[tr])
                p = est.predict_proba(col[te])[:, 1]
            else:
                p = col[te].ravel()

            oof[te] = p
            fold_ids[te] = fold_i

            # Guard: a fold with one class present has undefined AUC
            if len(np.unique(y[te])) < 2:
                warnings.warn(f"Fold {fold_i} for '{name}' has a single class; AUC skipped.")
                continue

            fold_aucs.append(roc_auc_score(y[te], p))
            fpr, tpr, _ = roc_curve(y[te], p)
            per_fold_roc.append((fpr, tpr))

        results[name] = OOFResult(name, oof, fold_ids, fold_aucs, per_fold_roc)

    return results


# ==========================================================================
# PART 3 — PAIRWISE DELONG MATRIX
# ==========================================================================

def pairwise_delong_matrix(y, oof: dict[str, OOFResult], alpha: float = 0.05):
    """
    All-pairs DeLong tests on pooled out-of-fold predictions, with
    Holm-Bonferroni FWER correction.
    """
    names = list(oof.keys())
    k = len(names)

    z_mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    p_mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)

    rows, raw_p = [], []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = names[i], names[j]
            auc_a, auc_b, z, p = delong_test(y, oof[a].oof_pred, oof[b].oof_pred)

            z_mat.loc[a, b] = z
            z_mat.loc[b, a] = -z
            p_mat.loc[a, b] = p
            p_mat.loc[b, a] = p

            rows.append({
                "predictor_a": a, "predictor_b": b,
                "auc_a": auc_a, "auc_b": auc_b,
                "auc_diff": auc_a - auc_b,
                "z": z, "p_raw": p,
            })
            raw_p.append(p)

    long = pd.DataFrame(rows)
    if len(long):
        adj, rej = holm_bonferroni(long["p_raw"].to_numpy(), alpha=alpha)
        long["p_holm"] = adj
        long["significant"] = rej
        long = long.sort_values("p_raw").reset_index(drop=True)

    return long, z_mat, p_mat


def auc_summary_table(y, oof: dict[str, OOFResult], alpha: float = 0.05) -> pd.DataFrame:
    """Per-predictor pooled AUC with DeLong CI, alongside the CV fold spread."""
    rows = []
    for name, r in oof.items():
        auc, lo, hi, se = delong_auc_ci(y, r.oof_pred, alpha=alpha)
        rows.append({
            "variable": name,
            "auc_pooled_oof": auc,
            "delong_se": se,
            "ci_low": lo,
            "ci_high": hi,
            "auc_mean_fold": r.mean_fold_auc,
            "auc_std_fold": r.std_fold_auc,
            "beats_chance": lo > 0.5,
            "worse_than_chance": hi < 0.5,   # <- flags DC / DMC / BUI
        })
    return (pd.DataFrame(rows)
            .sort_values("auc_pooled_oof", ascending=False)
            .reset_index(drop=True))


# ==========================================================================
# PART 4 — UN-SMOOTHED STEP ROC PLOTS
# ==========================================================================
# A ROC curve from n observations is a step function with at most n+1
# vertices. It MUST begin at (0,0) and end at (1,1). Any curve that is
# continuous, or that leaves the y-axis above zero, has been interpolated.
# `drawstyle="steps-post"` renders the estimator as it actually is.

def plot_step_roc_panel(
    y,
    oof: dict[str, OOFResult],
    predictors: list[str],
    ax,
    title: str,
):
    cmap = plt.get_cmap("tab10")

    for i, name in enumerate(predictors):
        if name not in oof:
            continue
        r = oof[name]
        colour = cmap(i % 10)

        # Pooled out-of-fold curve, plotted as a true step function.
        fpr, tpr, _ = roc_curve(y, r.oof_pred)
        auc, lo, hi, _ = delong_auc_ci(y, r.oof_pred)
        ax.plot(fpr, tpr, drawstyle="steps-post", color=colour, lw=1.9, zorder=3,
                label=f"{name}  AUC={auc:.3f} [{lo:.3f}, {hi:.3f}]")

    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.0, zorder=0, label="Random (AUC = 0.50)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_aspect("equal")


def plot_roc_figure(y, oof, n_obs: int, outpath: str = "roc_curves_fwi_cems_step.png"):
    with plt.rc_context({"font.family": "Arial"}):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))

        singles = [v for v in SINGLE_COMPONENTS if v in oof]
        composites = [v for v in COMPOSITE_VARS if v in oof]

        plot_step_roc_panel(y, oof, singles, axes[0], "Single FWI Components")
        plot_step_roc_panel(y, oof, composites, axes[1], "Composite FWI Variables")

        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.text(0.01, 0.005,
                  f"Source: EM-DAT (CRED/UCLouvain) x CEMS Fire Danger Reanalysis (Copernicus), n={n_obs}",
                  fontsize=8, color="grey", ha="left", va="bottom")
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    return fig


def plot_auc_ranking(summary: pd.DataFrame, n_obs: int,
                     outpath: str = "auc_ranking_fwi_cems_delong.png"):
    """Ranking plot with DeLong CIs (not fold SDs), and chance line at 0.5."""
    with plt.rc_context({"font.family": "Arial"}):
        df = summary.sort_values("auc_pooled_oof")
        colours = ["#d62728" if v in COMPOSITE_VARS else "#1f77b4" for v in df["variable"]]

        err_lo = df["auc_pooled_oof"] - df["ci_low"]
        err_hi = df["ci_high"] - df["auc_pooled_oof"]

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(df["variable"], df["auc_pooled_oof"], color=colours,
                xerr=[err_lo, err_hi], capsize=3.5,
                error_kw={"ecolor": "black", "lw": 1.1})

        ax.axvline(0.5, ls="--", color="grey", lw=1.2)
        ax.text(0.5, len(df) - 0.3, " Random classifier", color="grey", fontsize=8, va="top")
        ax.set_xlim(0.25, 1.0)
        ax.set_xlabel("Pooled out-of-fold AUC (95% DeLong CI)")

        handles = [
            plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
            plt.Rectangle((0, 0), 1, 1, color="#d62728"),
        ]
        ax.legend(handles, ["Single FWI component", "Composite variable"], loc="lower right")
        ax.grid(axis="x", alpha=0.3, lw=0.5)

        fig.tight_layout()
        fig.text(0.01, 0.01,
                  f"Source: EM-DAT (CRED/UCLouvain) x CEMS Fire Danger Reanalysis (Copernicus), n={n_obs}",
                  fontsize=8, color="grey", ha="left", va="bottom")
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    return fig


# ==========================================================================
# PART 5 — ORIENTATION DIAGNOSTIC (the DC / DMC / BUI result)
# ==========================================================================

def orientation_report(summary: pd.DataFrame) -> pd.DataFrame:
    """
    AUC is orientation-dependent: AUC(-s) = 1 - AUC(s). 
    This table reports the *magnitude* of discrimination, |AUC - 0.5|, and the
    implied direction. 
    """
    df = summary.copy()
    df["auc_inverted"] = 1 - df["auc_pooled_oof"]
    df["discrimination"] = (df["auc_pooled_oof"] - 0.5).abs()
    df["direction"] = np.where(
        df["auc_pooled_oof"] >= 0.5,
        "higher -> more loss",
        "higher -> LESS loss (inverted)",
    )
    return (df[["variable", "auc_pooled_oof", "auc_inverted", "discrimination",
                "direction", "ci_low", "ci_high"]]
            .sort_values("discrimination", ascending=False)
            .reset_index(drop=True))


# ==========================================================================
# MAIN
# ==========================================================================

def run(X: pd.DataFrame, y: np.ndarray, outdir: str = ".", score_mode: str = "raw"):
    """
    Entry point. Get (X, y) from fwi_roc_auc_emdat_cems.py's build_feature_matrix():

        from fwi_roc_auc_emdat_cems import build_feature_matrix
        from fwi_delong_roc import run
        X, y = build_feature_matrix()
        run(X, y, outdir="data", score_mode="raw")

    """
    import os
    os.makedirs(outdir, exist_ok=True)

    y = np.asarray(y).astype(int)
    n_obs = len(y)
    predictors = [v for v in SINGLE_COMPONENTS + COMPOSITE_VARS if v in X.columns]

    if not predictors:
        raise ValueError(
            "No recognised FWI predictors found in X. Expected some of:\n"
            f"  {SINGLE_COMPONENTS + COMPOSITE_VARS}\n"
            f"  got: {list(X.columns)}"
        )

    print(f"n = {n_obs}   positives = {y.sum()}   negatives = {(1-y).sum()}")
    print(f"predictors = {len(predictors)}   score_mode = '{score_mode}'\n")

    oof = collect_oof_predictions(X, y, predictors, score_mode=score_mode)

    summary = auc_summary_table(y, oof)
    long, z_mat, p_mat = pairwise_delong_matrix(y, oof)
    orient = orientation_report(summary)

    summary.to_csv(os.path.join(outdir, "auc_summary_delong.csv"), index=False)
    long.to_csv(os.path.join(outdir, "delong_pairwise_tests.csv"), index=False)
    p_mat.to_csv(os.path.join(outdir, "delong_pvalue_matrix.csv"))
    orient.to_csv(os.path.join(outdir, "auc_orientation_report.csv"), index=False)

    plot_roc_figure(y, oof, n_obs, os.path.join(outdir, "roc_curves_fwi_cems_step.png"))
    plot_auc_ranking(summary, n_obs, os.path.join(outdir, "auc_ranking_fwi_cems_delong.png"))

    print("=== AUC summary (pooled OOF, DeLong CI) ===")
    print(summary.to_string(index=False), "\n")

    print("=== Significant pairwise differences after Holm correction ===")
    sig = long[long["significant"]] if "significant" in long else long.iloc[:0]
    print(sig.to_string(index=False) if len(sig) else "  None survive FWER correction.\n")

    return {"oof": oof, "summary": summary, "pairwise": long, "orientation": orient}


def _synthetic_demo(n: int = 82, seed: int = 0):
    """Fabricated data with the real dimensions. Verifies the install; proves nothing."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.5, n)
    isi = np.abs(rng.gamma(2.0, 2.0, n) + 1.2 * y)     # non-negative, like real ISI
    fwi = np.abs(rng.gamma(2.0, 2.5, n) + 1.0 * y)
    X = pd.DataFrame({
        "FFMC": rng.normal(85, 5, n) + 1.5 * y,
        "DMC":  rng.normal(40, 15, n) - 5.0 * y,        # deliberately inverted
        "DC":   rng.normal(300, 80, n) - 30.0 * y,      # deliberately inverted
        "ISI":  isi,
        "BUI":  rng.normal(60, 20, n) - 7.0 * y,        # deliberately inverted
        "FWI":  fwi,
        "DSR":  0.0272 * fwi ** 1.77,
        "ISI_sq": isi ** 2,                             # exercises the var<=0 branch
        "FWI_x_ISI": fwi * isi,
        "FWI_event_mean": fwi + rng.normal(0, 2.0, n),
        "FWI_event_max": fwi + np.abs(rng.normal(0, 3.0, n)),
    })
    return X, y


_LAST_PIPELINE_MODULE = None  # stashed by _load_from_pipeline() so __main__ can pull the exclusion log afterward

MANUAL_EXCLUDE_DISNO = {"2023-0524-USA"}


def _load_from_pipeline():
    
    global _LAST_PIPELINE_MODULE
    import fwi_roc_auc_emdat_cems as pipe
    _LAST_PIPELINE_MODULE = pipe

    if hasattr(pipe, "build_feature_matrix"):
        X, y = pipe.build_feature_matrix()
    elif hasattr(pipe, "X") and hasattr(pipe, "y"):
        X, y = pipe.X, pipe.y
    else:
        raise AttributeError(
            "fwi_roc_auc_emdat_cems.py exposes neither `build_feature_matrix()` nor module-level `X`/`y`."
        )

    full_df = getattr(pipe, "_FULL_DF", None)
    if full_df is not None and "DisNo." in full_df.columns and MANUAL_EXCLUDE_DISNO:
        keep = ~full_df["DisNo."].isin(MANUAL_EXCLUDE_DISNO).values
        if not keep.all():
            dropped = full_df.loc[~keep, "DisNo."].tolist()
            print(f"Manually excluding {dropped} from this DeLong analysis "
                  f"(see MANUAL_EXCLUDE_DISNO).")
            X = X.loc[keep]
            y = y[keep]

    return X, y


if __name__ == "__main__":
    import argparse
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    default_outdir = os.path.normpath(os.path.join(here, "..", "data"))

    ap = argparse.ArgumentParser(
        description="DeLong tests and un-smoothed step ROC curves for FWI vs. wildfire loss."
    )
    ap.add_argument("--outdir", default=default_outdir,
                    help=f"Where to write CSVs and PNGs. Default: {default_outdir}")
    ap.add_argument("--score-mode", choices=["raw", "model"], default="raw",
                    help="'raw' preserves predictor direction (sub-chance AUCs); "
                         "'model' lets logistic regression absorb the sign.")
    ap.add_argument("--demo", action="store_true",
                    help="Run on synthetic data instead of importing fwi_roc_auc_emdat_cems.")
    args = ap.parse_args()

    if args.demo:
        print("[demo] synthetic data — results are meaningless, this only checks the install\n")
        X, y = _synthetic_demo()
    else:
        sys.path.insert(0, here)
        try:
            X, y = _load_from_pipeline()
        except Exception as exc:
            print(f"Could not load data from fwi_roc_auc_emdat_cems.py:\n  {exc}\n")
            print("Run with --demo to verify the install, or import `run` directly:\n")
            print("    from fwi_roc_auc_emdat_cems import build_feature_matrix")
            print("    from fwi_delong_roc import run")
            print("    X, y = build_feature_matrix()")
            print(f"    run(X, y, outdir={default_outdir!r}, score_mode='raw')")
            sys.exit(1)

    run(X, y, outdir=args.outdir, score_mode=args.score_mode)

    if _LAST_PIPELINE_MODULE is not None and hasattr(_LAST_PIPELINE_MODULE, "get_excluded_events"):
        excluded = _LAST_PIPELINE_MODULE.get_excluded_events()
        if excluded is not None and len(excluded):
            excluded_path = os.path.join(args.outdir, "excluded_events.csv")
            excluded.to_csv(excluded_path, index=False)
            print(f"\n=== Events excluded from the ROC analysis ({len(excluded)}) ===")
            print(excluded[["DisNo.", "Country", "Start Year", "stage", "reason"]].to_string(index=False))
            print(f"Full detail (incl. nearest-CEMS-data explanation) -> {excluded_path}")

    print(f"\nOutputs written to: {os.path.abspath(args.outdir)}")
