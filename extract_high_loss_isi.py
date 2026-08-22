"""
extract_high_loss_isi.py

Run this from the same folder as fwi_roc_auc_emdat_cems.py to pull real
per-event FWI-system values for the 15 high-loss events.

Author: Maya Lopansri, King's College London
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fwi_roc_auc_emdat_cems as pipe

# The 15 high-loss DisNo.s, confirmed against master_event_table.xlsx
# (Maui / 2023-0524-USA excluded — outside CEC North American scheme, §3.1/§4.2.2)
HIGH_LOSS_DISNO = [
    "2025-0008-USA", "2018-0409-USA", "2017-0434-USA", "2020-0441-USA",
    "2018-0468-USA", "2003-0786-USA", "2016-0172-CAN", "2021-0832-USA",
    "2007-0519-USA", "2021-0433-USA", "2008-0524-USA", "2017-0511-USA",
    "2000-0240-USA", "2011-0630-CAN", "2015-0421-USA",
]

X, y = pipe.build_feature_matrix()
full = pipe._FULL_DF.copy()

print(f"\nFull matched model set: {len(full)} events")
print(f"high_loss label distribution (NOTE: computed pre-exclusion in this "
      f"script's current code — see flagged issue): "
      f"{full['high_loss'].value_counts().to_dict()}")

missing = set(HIGH_LOSS_DISNO) - set(full["DisNo."])
if missing:
    print(f"\nWARNING: {len(missing)} of the 15 target events are NOT in the "
          f"matched model set (no CEMS coverage close enough, or excluded "
          f"upstream): {sorted(missing)}")
    print("Check get_excluded_events() below for the reason.")

subset = full[full["DisNo."].isin(HIGH_LOSS_DISNO)].copy()

out_cols = ["DisNo.", "event_date", "event_end_date", pipe.LOSS_COL,
            "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "DSR",
            "FWI_x_ISI", "BUI_x_FFMC", "DC_x_DMC", "ISI_sq",
            "FWI_event_mean", "FWI_event_max"]
out_cols = [c for c in out_cols if c in subset.columns]

subset = subset[out_cols].sort_values(pipe.LOSS_COL, ascending=False)
out_path = Path(__file__).resolve().parent / "high_loss_15_ISI_extracted.csv"
subset.to_csv(out_path, index=False)

print(f"\nExtracted {len(subset)} of 15 target events.")
print(subset.to_string(index=False))
print(f"\nSaved -> {out_path}")
print("\nUpload this CSV back — it's small (15 rows) and contains no raw CEMS data,")
print("just the matched per-event index values.")

excluded = pipe.get_excluded_events()
if excluded is not None and len(excluded):
    relevant_excluded = excluded[excluded["DisNo."].isin(HIGH_LOSS_DISNO)]
    if len(relevant_excluded):
        print(f"\n{len(relevant_excluded)} of the 15 target events were excluded upstream:")
        print(relevant_excluded.to_string(index=False))
