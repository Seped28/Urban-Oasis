# UrbanWell Analytics

Satellite-derived urban green/blue space (UGBS) exposure and mental health
across 85,395 US census tracts, 2015–2023.

## Papers
- Paper 1: Ipede, O., Lin, M., Hladik, C., & Tu, W. (2026). Urban Oases: The Critical Role of Green and Blue Spaces in Mental Well-Being. Sustainability, 18(2), 642. https://doi.org/10.3390/su18020642.
- Paper 2a: [ECAS-8 proceedings citation, once assigned]
- Paper 2b: in preparation

## Pre-registration
- Registration (H1–H6): [[OSF link](https://osf.io/dx6ng/)]

## What's here
Twelve scripts covering the full pipeline — see [SCRIPTS.md](SCRIPTS.md) for
full documentation of each one (purpose, inputs, outputs, run order).

## Quick start
1. Google Earth Engine account with access to your own GEE project
   (`PROJECT = 'urbangreenblue'` in the scripts — change to your own project ID).
2. Run the three `gee/*.js` scripts in the GEE Code Editor.
3. Run the `python/` scripts in the
   order given in SCRIPTS.md §1.
4. `Rscript r/data_prep.R` to build the master panel.
5. `python python/mgwr_analysis.py` then `python python/validation_report.py`.

## Data sources
- CDC PLACES (mental health outcome): [[link](https://data.cdc.gov/browse?category=500+Cities+%26+Places&sortBy=relevance&tags=places&pageSize=20&q=PLACES%3A+Local+Data+for+Better+Health%2C+Census+Tract)]
- ACS covariates: [[link](https://api.census.gov/data/key_signup.html)]
- SVI: [[link](https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html)]
- CDC PLACES/ACS/SVI download scripts are not included in this release —
  see SCRIPTS.md §6 for what's referenced but not (yet) included.

## Citation
If you use this code, please cite [[Paper 1 citation](https://doi.org/10.3390/su18020642)] and, once available,
[Paper 2a citation].

## License
MIT — see LICENSE.
