# UrbanWell Analytics — Scripts Documentation

Twelve scripts across three languages, covering the full pipeline from satellite export through statistical modeling. Each is documented below with purpose, inputs, outputs, and dependencies. Run order matters — follow §1 for a first-time setup.

---

## 1. Run Order (first time)

```
GEE exports (run in the GEE Code Editor, in any order):
  gc_unweighted_export.js
  tcc_reference_export.js
  lst_summer_export.js

Python, in this order:
  gs_hls_scale_unify_2015_2018.py     (production GS/BS exports, 2015-2018)
  gs_s2_scale_unify_2019_2020.py      (production GS exports, 2019-2020+)
  kappa_validation.py                  (bluespace classifier validation)
  greenex_naip_validation.py           (greenspace Level 1 validation)
  verify_naip_dates.py                 (confirms NAIP reference year assumptions)
  compute_proximity_metrics.py         (optional — Phase 2, park/resource proximity)

R:
  data_prep.R                          (merges everything into the master panel)

Python (after data_prep.R and the GEE exports above have all run):
  mgwr_analysis.py                     (fits OLS/GWR/MGWR per city)
  validation_report.py                 (all validation tables and correlations)
```

---

## 2. Google Earth Engine exports (JavaScript, run in the GEE Code Editor)

### `gc_unweighted_export.js`
**Purpose:** exports raw, unweighted, unbuffered greenspace coverage per census tract — the construct Wu et al. (2023) validate in their Supplementary Fig. 17, distinct from the project's primary exposure metric (which is population-weighted and passed through a 500m smoothing kernel).
**Method:** Sentinel-2 surface reflectance, three-endmember linear spectral unmixing (vegetation/impervious/soil), same band set, cloud mask, and endmember values as the production pipeline.
**Output:** `greenex_gc_unweighted_{CITY}.csv` (GEOID, city, mean, count) to Google Drive.
**Used by:** `validation_report.py --greenex-correlate` (Level 2).

### `tcc_reference_export.js`
**Purpose:** exports tract-level Tree Canopy Cover from the USDA Forest Service / USGS NLCD TCC product — an independent, federally produced validation reference, distinct from the project's own classifiers.
**Dataset:** `USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5`.
**Scope note:** tree canopy only, not all vegetation — expect systematically lower values than the project's own vegetation-fraction metric, by construction.
**Output:** `greenex_tcc_reference_{CITY}.csv` (GEOID, city, tcc_pct_mean, tcc_frac_mean, count, tcc_year, product).
**Used by:** `validation_report.py --tcc-correlate` (Level 3).

### `lst_summer_export.js`
**Purpose:** exports mean summer land surface temperature per tract for an exploratory correlation check against greenspace/bluespace exposure. Not a model covariate — a mechanism check against the literature's atmospheric-cooling framing.
**Method:** Landsat 8/9 Collection 2 Level 2 surface temperature band (ST_B10), USGS scale factors applied, converted to Celsius, merged across both satellites for denser summer coverage.
**Output:** `lst_summer_{CITY}.csv` (GEOID, city, year, mean, count).
**Used by:** `validation_report.py --lst-correlate`.

---

## 3. Production exposure exports (Python, calls the Earth Engine API)

### `gs_hls_scale_unify_2015_2018.py`
**Purpose:** production greenspace and bluespace exposure export for the early panel years (2015–2018), using NASA/USGS Harmonized Landsat Sentinel-2 (HLS30) imagery for greenspace.
**Output:** county/state-tiled CSVs feeding the master panel via `data_prep.R`.

### `gs_s2_scale_unify_2019_2020.py`
**Purpose:** production greenspace export for 2019 onward, using Sentinel-2 imagery. Contains the actual production spectral-unmixing endmember values (vegetation/impervious/soil) that `gc_unweighted_export.js` deliberately mirrors for its Level 2 validation comparison.
**Output:** county/state-tiled CSVs feeding the master panel via `data_prep.R`.

---

## 4. Validation scripts (Python)

### `kappa_validation.py`
**Purpose:** validates dual-sensor (Sentinel-2 MNDWI + Sentinel-1 SAR) bluespace classification against NAIP aerial reference imagery, reporting Cohen's kappa and a full confusion matrix per county.
**Modes:** default dual-sensor AND gate, plus MNDWI-only, relaxed SAR threshold, and OR-logic alternatives — useful because inland, tree-canopied water features can cause SAR double-bounce backscatter, making the default gate under-detect water in heavily forested counties.
**Output:** `kappa_results.json`, printed confusion matrices.

### `greenex_naip_validation.py`
**Purpose:** validates satellite-derived greenspace exposure against an independent NAIP aerial-imagery reference (Level 1 validation — a different construct comparison than `gc_unweighted_export.js`'s Level 2, see `validation_report.py`).
**Method:** trains a random forest vegetation classifier on NDVI-threshold pseudo-labels from NAIP, compares to the production exposure metric.
**Limitation, worth stating in any Methods section:** the training labels are NDVI-threshold pseudo-labels on NAIP imagery, not independently verified ground truth.
**Output:** `greenex_naip_reference_{CITY}.csv` to Google Drive.

### `verify_naip_dates.py`
**Purpose:** confirms whether NAIP imagery returned by a given year filter is genuinely acquired in that year, rather than assuming the filter is accurate. Pulls actual acquisition timestamps and flags any mismatch.
**Why it matters:** NAIP flight coverage is not annual for every county; a year filter can silently return the nearest available year rather than the requested one.

### `compute_proximity_metrics.py`
**Purpose:** computes park and resource proximity metrics (NLCD-based distance, TPL ParkServe walk-access, NWI-based water-body distance) as a secondary, distance-based complement to the project's primary percentage-exposure metrics.
**Status:** a later-phase addition, not part of the primary panel's confirmatory analysis — kept as a separate, optional script rather than merged into the main pipeline.

---

## 5. Modeling (Python and R)

### `data_prep.R`
**Purpose:** the central merge script. Loads all GEE exports (greenspace, bluespace), joins CDC PLACES (mental health outcome), ACS covariates, SVI (social vulnerability moderator), applies boundary-crosswalk and outlier-removal rules, projects to a metric CRS, builds spatial weights, and runs an OLS baseline with a Moran's I diagnostic to confirm spatial autocorrelation before proceeding to spatial models.
**Output:** `ugbs_panel_2015_2023.csv` (flat panel), `ugbs_panel_sf_5070.rds` (spatial object), `spatial_weights_k8.rds` (k-nearest-neighbor spatial weights).
**Everything downstream depends on this script's output.**

### `mgwr_analysis.py`
**Purpose:** fits OLS, Geographically Weighted Regression (GWR), and Multiscale GWR (MGWR) per city, comparing model fit via corrected AIC (AICc). MGWR is the paper's central spatial-heterogeneity result — different covariates are allowed their own, independently optimized spatial bandwidth.
**Implementation notes:** continuous covariates are standardized before GWR/MGWR (not OLS) to avoid a numerically unstable local design matrix at small candidate bandwidths; a minimum-bandwidth floor and retry logic guard against this same failure mode; MGWR non-convergence for a given city is reported as a named limitation rather than silently dropped, preserving that city's valid OLS/GWR results.
**Output:** JSON/CSV results per city, feeding `validation_report.py --tables`.

### `validation_report.py`
**Purpose:** the single reporting script that ties everything together — auto-fills the OLS/GWR/MGWR comparison tables, runs all three levels of greenspace validation (NAIP, coverage-to-coverage, independent federal product), the LST mechanism check, and produces coefficient maps.
**Modes:** `--tables`, `--greenex-correlate`, `--tcc-correlate`, `--lst-correlate`, `--maps` — run any subset independently.
**Depends on:** outputs from `data_prep.R`, `mgwr_analysis.py`, and all three GEE export scripts having already run and been downloaded to `data/processed/`.

---

## 6. What's referenced but not included in this release

A few scripts mentioned in the project's own documentation are not part of this release, either because they weren't shared in the sessions this documentation was built from, or because they're not yet finalized:
- The NHGIS boundary-crosswalk script and the CDC PLACES/ACS/SVI download scripts referenced by `data_prep.R`'s error messages.
- Any diagnostics-guide or Methods-note markdown files referenced in code comments (e.g. a scale-methods citation note).

If you're assembling a full repository from this documentation, treat those as gaps to fill from your own copies, not as evidence they don't exist.
