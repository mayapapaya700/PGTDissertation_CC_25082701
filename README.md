This repository is submitted as part of a MSc degree in Climate Change: Environment, Science and Policy at King’s College London. 

# How Well Does the FWI Discriminate Historical Wildfire-Induced Insured Losses — Fire-Regime Subgroups, Lag Test & Pipeline Reference

*Consolidated working reference. Ecoregion grouping is based on physical CEC Level I
classification plus the EM-DAT subtype field. The Australian Fire Danger Rating System
is used only as an interpretive comparison — it is NOT used to reclassify any data.*

---

## 1. Fire-regime subgroup scheme (CEC Level I → subgroup)

The split that carries the argument is **fuel-limited vs. flammability-limited**, the
same axis as McNorton's hydroclimatic-rebound mechanism. In fuel-limited systems the
drought codes (DC/DMC/BUI) are expected to under-perform or anti-correlate with insured
loss; in flammability-limited (fuel-rich) systems they should behave normally.

| CEC Level I region | Subgroup | Mechanism |
|---|---|---|
| 11 Mediterranean California | Fuel-limited | Rebound; long antecedent memory |
| 10 North American Deserts | Fuel-limited | Rebound |
| 12 Southern Semiarid Highlands | Fuel-limited | Rebound (thin) |
| 9 Great Plains | Fuel-limited | Fine-fuel, wind/ISI-driven |
| 6 NW Forested Mountains | Flammability-limited | Drought-driven |
| 7 Marine West Coast Forest | Flammability-limited | Drought-driven |
| 13 Temperate Sierras | Flammability-limited | Drought-driven |
| 3 Taiga | Flammability-limited | Boreal crown fire |
| 5 Northern Forests | Flammability-limited | Boreal/mixed |
| 8 Eastern Temperate Forests | Flammability-limited* | Suppression-dominated (*distinct regime) |
| 1 Arctic Cordillera, 2 Tundra, 4 Hudson Plain, 14 Tropical Dry, 15 Tropical Wet | Masked | Negligible fire loss |

---

## 2. The lag test (Test 2 — Welch's t-test)

**Grouping variable:** physical ecoregion subgroup (fuel-limited vs. flammability-limited).
**Measured variable:** per-event antecedent lag = **timing of the largest VPD drying anomaly**
before the fire.

- **VPD sign convention:** VPD = e_s(T) − e_s(Td), kPa; higher VPD = drier, so a *drying*
  anomaly is a *positive* standardised VPD anomaly. The lag is the month of the maximum
  positive VPD sigma anomaly in the 1–24 month pre-fire window.
- **Test:** `scipy.stats.ttest_ind(fuel, flamm, equal_var=False)` (Welch, unequal variance),
  reported alongside **Mann–Whitney U** (the more trustworthy statistic at this n),
  Cohen's d, and a 95% CI on the mean difference.
- **Data note:** VPD is NOT in the CEMS Fire Danger NetCDFs (FWI codes only). It must be
  derived from ERA5-Land 2 m temperature + 2 m dewpoint, as in McNorton.

*Related framings for the write-up:* Test 1 (one-sample, one-sided) asks whether the
empirical lag exceeds FWI's deepest built-in memory (Drought Code ≈ 52 days); Test 3
(paired) compares empirical vs. FWI-implied lag per event.

---

## 3. Australian FDRS — interpretive comparison only

The Australian system independently splits fire danger by fuel type: classically FFDI
(forest, KBDI-type drought memory) vs. GFDI (grassland, explicit *curing* term encoding
the wet-grow-then-cure rebound). The modern AFDRS (2022) went further — eight vegetation-
specific models factoring fuel load, time since last fire, and curing. This illustrates
why letting vegetation type govern which weather/fuel signals matter is defensible; it is
cited as an example, not used to relabel any events.

*EM-DAT subtype ↔ fuel axis (descriptive cross-check only, general → mixed):*
Forest fire → forest/flammability-limited · Land fire (brush/bush/pasture) →
grass-shrub/fuel-limited · Wildfire (General) → mixed.

---

## 4. Pipeline & run order

Run everything from the Dissertation root (`cd ~/Desktop/Dissertation`).

1. **`run_welch_ttest.py`** — assigns events to Level I region → subgroup (point-in-polygon
   with projected-CRS nearest fallback), attaches subtype (general→mixed), writes the two
   assignment CSVs, and runs the Welch test *if* the lag file exists.
   Needs: `data/public_emdat_custom_request_2026.xlsx`,
   `data/NA_CEC_Eco_Level1/NA_CEC_Eco_Level1.shp` (+ .dbf/.shx/.prj).
2. **`extract_vpd_lags.py`** — computes VPD from ERA5-Land t2m+d2m, standardises to monthly
   sigma anomalies, finds each event's largest-drying-anomaly lag, writes
   `data/per_event_lags.csv`. Needs ERA5-Land monthly 2 m temp + 2 m dewpoint in `data/era5/`.
3. Re-run **`run_welch_ttest.py`** — now finds `per_event_lags.csv` and prints the Welch +
   Mann–Whitney result.

**Outputs:** `event_subgroup_assignments.csv` (all events) vs.
`event_subgroup_assignments_loss.csv` (63 with a Total Damage value). Of these, **60 are
matched to a CEMS Fire Danger grid bounding box** — that CEMS-matched subset is the working
set for DeLong/ROC and the lag test.

---

## 5. Loss-bearing event assignments (63 total; 60 CEMS-matched)

Rows marked **†** are in the loss-bearing universe but were not matched to a CEMS Fire
Danger grid bounding box, so they sit outside the 60-event analysis working set.

### Flammability-limited (17 total; 15 CEMS-matched)

| DisNo | Yr | Location | Level I region | Subtype |
|---|---|---|---|---|
| 2002-0184-USA | 02 | Lincoln district (New Mexico provi | Temperate Sierras | forest |
| 2002-0351-USA | 02 | Colorado province | NW Forested Mtns | forest |
| 2002-0379-USA | 02 | Arizona province | Temperate Sierras | forest |
| 2011-0349-USA | 11 | Bastrop district (Texas province) | E Temperate Forests | forest |
| 2011-0630-CAN | 11 | Slave Lake area (Division No. 17 d | Northern Forests | forest |
| 2013-0315-USA | 13 | Yosemite valley (Mariposa district | NW Forested Mtns | mixed |
| 2015-0401-USA | 15 | California, Idaho, Montana, Oregon | NW Forested Mtns | mixed |
| 2015-0613-USA | 15 | Amador, Calaveras (California) | NW Forested Mtns | land |
| 2016-0172-CAN | 16 | Fort McMurray city (Wood Buffalo a | Northern Forests | forest |
| 2016-0463-USA | 16 | Gatlinburg city (Sevier district,  | E Temperate Forests | forest |
| 2021-0433-USA | 21 | Plumas County (California) | NW Forested Mtns | forest |
| 2021-0526-USA | 21 | El Dorado County (California) | NW Forested Mtns | forest |
| 2022-0486-USA | 22 | Siskiyou County (California) | NW Forested Mtns | mixed |
| 2023-0280-CAN † | 23 | Alberta, British Columbia, Nova Sc | Northern Forests | forest |
| 2023-0524-USA † | 23 | Maui and Kula City (Hawai) | Marine W Coast | mixed |
| 2024-0403-USA | 24 | Otero and Lincoln Counties (New Me | Temperate Sierras | forest |
| 2024-0524-CAN | 24 | National park Jasper (Alberta prov | NW Forested Mtns | mixed |

### Fuel-limited (46 total; 45 CEMS-matched)

| DisNo | Yr | Location | Level I region | Subtype |
|---|---|---|---|---|
| 2000-0240-USA | 00 | Los Alamos, Rio Arriba, Sandoval,  | NA Deserts | forest |
| 2000-0465-USA | 00 | Arizona, California, Colorado, Ida | NA Deserts | forest |
| 2002-0092-USA | 02 | Fallbrok area (San Diego district, | Mediterranean CA | forest |
| 2003-0369-CAN | 03 | British Columbia, Alberta, Saskatc | Great Plains | forest |
| 2003-0786-USA | 03 | San Bernardino district (Californi | Mediterranean CA | forest |
| 2005-0724-USA | 05 | Texas, Oklahoma provinces | Great Plains | forest |
| 2006-0374-USA | 06 | California province | Mediterranean CA | mixed |
| 2006-0576-USA | 06 | Riverside, Palm Springs cities (Ri | Mediterranean CA | land |
| 2006-0662-USA | 06 | Ventura district (California provi | Mediterranean CA | land |
| 2006-0759-USA | 06 | Texas | Great Plains | mixed |
| 2007-0519-USA | 07 | Los Angeles, Orange, Riverside, Sa | Mediterranean CA | land |
| 2007-0579-USA † | 07 | Malibu area (Los Angeles district, | Mediterranean CA | forest |
| 2008-0253-USA | 08 | California province | Mediterranean CA | forest |
| 2008-0524-USA | 08 | Los Angeles, Orange, Riverside, Sa | Mediterranean CA | mixed |
| 2009-0178-USA | 09 | Santa Barbara district (California | Mediterranean CA | land |
| 2011-0140-USA | 11 | Texas province | Great Plains | land |
| 2011-0639-USA | 11 | Arizona, Minnesota, Texas, Florida | Great Plains | forest |
| 2012-0232-USA | 12 | Colorado Springs city (El Paso dis | Great Plains | mixed |
| 2012-0542-USA | 12 | Cleveland, Creek, Oklahoma, Payne  | Great Plains | mixed |
| 2013-0209-USA | 13 | Los Angeles district (California p | Mediterranean CA | forest |
| 2013-0210-USA | 13 | Black Forest area (El Paso distric | Great Plains | forest |
| 2013-0212-USA | 13 | Yarnell area (Yavapai district, Ar | NA Deserts | forest |
| 2013-0404-USA | 13 | Shasta district (California provin | Mediterranean CA | mixed |
| 2014-0165-USA | 14 | Hutchinson district (Texas provinc | Great Plains | forest |
| 2015-0312-USA | 15 | Lake, Napa, Solano, Yolo, Modoc, C | Mediterranean CA | mixed |
| 2015-0421-USA | 15 | Lake, Napa, Sonoma, Butte district | Mediterranean CA | land |
| 2016-0206-USA | 16 | California province | Mediterranean CA | forest |
| 2016-0264-USA | 16 | Los Angeles, San Francisco distric | Mediterranean CA | forest |
| 2016-0297-USA | 16 | Lower Lake, Clearlake (Lake distri | Mediterranean CA | forest |
| 2017-0434-USA | 17 | Napa, Sonoma, Mendocino, Lake, Sol | Mediterranean CA | land |
| 2017-0511-USA | 17 | Ventura, Santa Barbara, Los Angele | Mediterranean CA | mixed |
| 2018-0258-USA | 18 | Shasta, Trinity, Mendocino, Lake,  | Mediterranean CA | forest |
| 2018-0409-USA | 18 | Butte county (North California) | Mediterranean CA | forest |
| 2018-0468-USA | 18 | Thousand Oaks, Oak Park, Westlake  | Mediterranean CA | mixed |
| 2019-0513-USA | 19 | Los Angeles and Riverside counties | Mediterranean CA | forest |
| 2019-0517-USA | 19 | Los Angeles, San Bernardino, Ventu | NA Deserts | forest |
| 2020-0441-USA | 20 | California, Washington, Oregon, Co | NA Deserts | forest |
| 2021-0357-USA | 21 | Pinas and Gila counties (Arizona); | NA Deserts | mixed |
| 2021-0367-CAN | 21 | Lytton (British Columbia) | NA Deserts | mixed |
| 2021-0410-USA | 21 | Arizona, California, Oregon | NA Deserts | mixed |
| 2021-0832-USA | 21 | Louisville, Superior (Boulder coun | Great Plains | mixed |
| 2022-0215-USA | 22 | Lincoln, San Miguel Counties (New  | Great Plains | mixed |
| 2024-0126-USA | 24 | Texas, Oklahoma | Great Plains | mixed |
| 2024-0827-USA | 24 | Ventura county (California) | Mediterranean CA | mixed |
| 2025-0008-USA | 25 | Los Angeles County (northern of Lo | Mediterranean CA | mixed |
| 2025-0477-USA | 25 | Wasco County (Oregon) | NA Deserts | mixed |

---

## 6. Data-quality flags (carry into methods)

- **63 loss-bearing / 60 CEMS-matched:** the full loss-bearing universe (has a Total Damage
  value) is 63 events. Three of them — `2023-0280-CAN`, `2007-0579-USA`, and `2023-0524-USA`
  (marked **†** in Section 5) — are not matched to a CEMS Fire Danger grid bounding box, so
  they fall outside the 60-event CEMS-matched analysis working set (15 flammability-limited,
  45 fuel-limited) used for DeLong/ROC and the lag test. They remain part of the 63-event
  universe for any analysis that doesn't require CEMS coverage.
- **Hawaii (`2023-0524`, Maui/Lahaina) — additional flag:** on top of not being CEMS-matched,
  it also sits outside the CEC North American ecoregion scheme, where the nearest-polygon
  fallback mis-snaps it to Marine West Coast Forest. If it's ever pulled into a CEMS-matched
  extension, hand-flag or exclude it rather than trusting that ecoregion tag.
- **~20 multi-jurisdiction centroids:** EM-DAT gives one coordinate for multi-state events,
  so region tags for those are only as good as the centroid. Hand-refine or exclude the
  pan-regional ones.
- **Subtype ↔ ecoregion discordance = 35% (22/63):** EM-DAT subtype is a poor proxy for fuel
  regime — this *justifies* grouping on physical ecoregion rather than the fuel label.
- **60 vs. 55:** the 60-event CEMS-matched working set still doesn't match the 55 used in
  `fwi_delong_roc.py`; that's a further script-internal filter (year-2000 accounts for 2;
  the rest is likely additional CEMS grid coverage gaps). Reconcile against the script to
  label the exact 55.
- **High/Lower EM-DAT loss quartile (separate from fire-regime grouping):** for the
  DeLong/ROC-AUC discrimination analysis, the 60 CEMS-matched events are also split by EM-DAT
  loss quartile into **High (15)** and **Lower (45)** groups. This is a distinct axis from
  the Flammability-limited/Fuel-limited split in Section 5 — that split is the physical
  ecoregion grouping used for the lag test (Section 2); High/Lower is the loss-severity
  grouping used for the ROC/AUC work. Per-event High/Lower labels TBD — reconcile against
  `fwi_delong_roc.py`.

---

## 7. Next steps

1. Download ERA5-Land monthly 2 m temp + 2 m dewpoint (North America, 1998–2026) → `data/era5/`.
2. Run `extract_vpd_lags.py` → `per_event_lags.csv`.
3. Re-run `run_welch_ttest.py` → Welch + Mann–Whitney result.
4. Reconcile the 63→60 (CEMS match) →55 (`fwi_delong_roc.py`) filter chain, and confirm the
   per-event High/Lower EM-DAT loss-quartile assignments used for the ROC/AUC analysis.
5. Decide whether to widen the lag window to 27 months to probe the fuel-accumulation phase.

---

## 8. Reference dataset — McNorton et al., 2025 (Global Change Biology)

This repository also contains the data used in the publication *"Hydroclimatic Rebound
Drives Extreme Fire in California's Non-Forested Ecosystems"* (McNorton et al., 2025,
*Global Change Biology*), which underpins the hydroclimatic-rebound mechanism referenced
in Section 1 above.

The primary dataset used in the McNorton reanalysis is hosted on Zenodo:
["Hydroclimatic Rebound Drives Extreme Fire in California's Non-Forested
  Ecosystems"](https://zenodo.org/records/16950724)

To run the pipeline locally, download the CSV files from the Zenodo record and place
it inside the `data/raw/` directory.
**Plotting code for the figures:**

- `PLOT_FIG2.py`
- `PLOT_FIG3.py`
- `PLOT_FIG4.py`

**Data for figures:**

- `Fire_Info.csv`
- `PAL_EAT_DATA.csv`
- `sig_results.csv`
- `ALL_FIRE_DATA.csv`
- `ALL_REGION_DATA.csv`

The data used in this study are primarily derived from publicly available datasets, as
described in the main manuscript. Some operational components, however, originate from
ECMWF data products generated through the Integrated Forecasting System (IFS), which are
accessible to registered ECMWF users. The processed data are available through the CSVs
listed above.
