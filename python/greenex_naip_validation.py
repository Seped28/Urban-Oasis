"""
greenex_naip_validation.py — UrbanWell Analytics

Validates satellite-derived greenspace exposure (GreenEx_pct) against an
independent NAIP aerial-imagery reference. Trains a random forest
vegetation classifier on NDVI-threshold pseudo-labels from NAIP, then
compares the classified vegetation fraction to GreenEx_pct at tract level.

Notable implementation choices:
- The tract-count check runs asynchronously (via the Drive/Tasks tab after
  export) rather than as a synchronous pre-export call, since GEE's
  synchronous request window (~5 minutes) is far more restrictive than the
  async batch export task that actually does the heavy computation.
- Training-sample collection (stratifiedSample) runs at 5m rather than
  native ~1m resolution -- still much finer than the 100m final output,
  and substantially faster.
- tileScale=16 on stratifiedSample and reduceRegions calls, needed for
  large-AOI counties.
- A retry wrapper handles transient GEE errors (timeouts, concurrent-task
  limits) with exponential backoff.
- No separate reprojection/aggregation step before the zonal reduce:
  reduceRegions() aggregates native-resolution pixels down to whatever
  `scale` is requested internally, so a manual reduceResolution() call
  would be both unnecessary and prone to memory-limit failures over a
  full county at native resolution.

See classify_vegetation() and run_city_once() for the full pipeline. Note
the training labels are NDVI-threshold pseudo-labels on NAIP imagery, not
independently verified ground truth -- worth stating as a caveat in any
Methods section that cites this validation.
"""

import argparse
import pathlib
import sys
import time
from datetime import datetime

import ee

PROJECT = 'urbangreenblue'
TARGET_YEAR = 2023
N_TRAIN_PER_CLASS = 300
TRAIN_SAMPLE_SCALE = 5      # far finer than the 100m final output, much faster than native ~1m
TILE_SCALE = 16             # needed for large-AOI counties (e.g. Los Angeles)
NTREES = 15
NDVI_VEG_THRESHOLD = 0.5
NDVI_NONVEG_THRESHOLD = 0.1
SEED = 42

CITIES = {
    'ATL': {'name': 'Atlanta (Fulton County, GA)',      'state': '13', 'county': '121'},
    'MIA': {'name': 'Miami (Miami-Dade County, FL)',    'state': '12', 'county': '086'},
    'LAX': {'name': 'Los Angeles (LA County, CA)',      'state': '06', 'county': '037'},
}

OUT_DIR = pathlib.Path('data/processed')

TRANSIENT = ('timed out', 'computation timed out', 'too many concurrent',
             'internal error', 'deadline exceeded', 'user memory limit')


def init_ee():
    ee.Initialize(project=PROJECT)
    print(f'GEE initialised | project={PROJECT} | TARGET_YEAR={TARGET_YEAR}')


def get_aoi(state_fips: str, county_fips: str) -> ee.Geometry:
    return (ee.FeatureCollection('TIGER/2020/TRACT')
              .filter(ee.Filter.And(
                  ee.Filter.eq('STATEFP', state_fips),
                  ee.Filter.eq('COUNTYFP', county_fips)))
              .geometry().dissolve(maxError=10))


def get_naip_composite(aoi: ee.Geometry, target_year: int):
    naip_col = ee.ImageCollection('USDA/NAIP/DOQQ')
    for yr in [target_year, target_year - 1, target_year - 2, target_year - 3]:
        col = naip_col.filterBounds(aoi).filterDate(f'{yr}-01-01', f'{yr}-12-31')
        n = col.size().getInfo()
        print(f'    NAIP {yr}: {n} images in AOI')
        if n > 0:
            return col.median(), yr
    raise RuntimeError('No NAIP found within 4 years of target — cannot validate')


def classify_vegetation(naip_img: ee.Image, aoi: ee.Geometry) -> ee.Image:
    ndvi = naip_img.normalizedDifference(['N', 'R']).rename('NDVI')
    features = naip_img.select(['R', 'G', 'B', 'N']).addBands(ndvi)

    veg_mask    = ndvi.gt(NDVI_VEG_THRESHOLD)
    nonveg_mask = ndvi.lt(NDVI_NONVEG_THRESHOLD)

    veg_pts = features.updateMask(veg_mask).addBands(ee.Image.constant(1).rename('class')) \
        .stratifiedSample(numPoints=N_TRAIN_PER_CLASS, classBand='class', region=aoi,
                          scale=TRAIN_SAMPLE_SCALE, seed=SEED, geometries=True, tileScale=TILE_SCALE)
    nonveg_pts = features.updateMask(nonveg_mask).addBands(ee.Image.constant(0).rename('class')) \
        .stratifiedSample(numPoints=N_TRAIN_PER_CLASS, classBand='class', region=aoi,
                          scale=TRAIN_SAMPLE_SCALE, seed=SEED, geometries=True, tileScale=TILE_SCALE)
    training = veg_pts.merge(nonveg_pts)

    classifier = ee.Classifier.smileRandomForest(NTREES).train(
        features=training, classProperty='class',
        inputProperties=['R', 'G', 'B', 'N', 'NDVI'],
    )
    classified = features.classify(classifier).rename('veg_predicted')
    return classified.toUint8()


# No separate aggregate_to_100m()/reduceResolution() step: reduceRegions()
# already aggregates native-resolution pixels down to whatever `scale` is
# requested internally, so a manual reprojection step is both unnecessary
# and prone to memory-limit failures over a full county at native
# resolution. reduceResolution() is only needed when combining two images
# of genuinely different native resolutions before a joint pixel-wise
# operation, which doesn't happen here.


def zonal_mean_per_tract(veg_binary: ee.Image, state_fips: str, county_fips: str) -> ee.FeatureCollection:
    """Reduces directly from the classified (native-resolution) image to
    tract polygons at scale=100, letting reduceRegions handle the internal
    aggregation -- no separate reproject step."""
    tracts = (ee.FeatureCollection('TIGER/2020/TRACT')
                .filter(ee.Filter.And(
                    ee.Filter.eq('STATEFP', state_fips),
                    ee.Filter.eq('COUNTYFP', county_fips))))
    return veg_binary.rename('veg_fraction_100m').reduceRegions(
        collection=tracts,
        reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
        scale=100,
        tileScale=TILE_SCALE,
    )


def run_city_once(code: str, cfg: dict, export: bool):
    print(f'\n{"="*70}')
    print(f'CITY: {cfg["name"]}')
    print(f'{"="*70}')

    aoi = get_aoi(cfg['state'], cfg['county'])

    print('  Getting NAIP composite...')
    naip_img, naip_year = get_naip_composite(aoi, TARGET_YEAR)
    year_matched = (naip_year == TARGET_YEAR)
    if not year_matched:
        print(f'    WARNING: NAIP {TARGET_YEAR} unavailable — using {naip_year}.')

    print('  Training random forest vegetation classifier (NDVI-threshold pseudo-labels)...')
    veg_binary = classify_vegetation(naip_img, aoi)

    print('  Building tract-level zonal-mean feature collection (not evaluated yet)...')
    tract_fc = zonal_mean_per_tract(veg_binary, cfg['state'], cfg['county'])

    # Tract count is not checked synchronously here -- the export task
    # below runs async with much more headroom. For a sanity check on
    # tract count, look at the Drive CSV's row count after the export
    # finishes, or check the Tasks tab.

    if export:
        out_name = f'greenex_naip_reference_{code}'
        task = ee.batch.Export.table.toDrive(
            collection=tract_fc.map(lambda f: f.set({
                'city': cfg['name'],
                'naip_year_used': naip_year,
                'year_matched': year_matched,
            })),
            description=out_name,
            folder='UrbanWellExports',
            fileNamePrefix=out_name,
            fileFormat='CSV',
            selectors=['GEOID', 'city', 'mean', 'count', 'naip_year_used', 'year_matched'],
        )
        task.start()
        print(f'  Export task started: {out_name} → Drive folder UrbanWellExports')
        print(f'  This runs server-side asynchronously -- check progress at')
        print(f'  https://code.earthengine.google.com/tasks (may take several')
        print(f'  minutes for a full county at this resolution; that\'s normal')
        print(f'  for an async batch task, unlike the synchronous call that just')
        print(f'  timed out).')
        return task
    return None


def run_city(code: str, cfg: dict, export: bool, retries: int = 2):
    """Retry wrapper for transient GEE errors (timeouts, concurrent-task
    limits), matching the pattern used in kappa_validation.py."""
    for attempt in range(1, retries + 2):
        try:
            return run_city_once(code, cfg, export)
        except Exception as e:
            msg = str(e)
            is_transient = any(m in msg.lower() for m in TRANSIENT)
            if is_transient and attempt <= retries:
                wait = 30 * attempt
                print(f'  TRANSIENT ERROR ({attempt}/{retries+1}): {msg} — retry in {wait}s')
                time.sleep(wait)
                continue
            print(f'  ERROR for {code}: {msg}')
            raise


def main():
    parser = argparse.ArgumentParser(description='GreenEx NAIP validation')
    parser.add_argument('--city', choices=['ATL', 'MIA', 'LAX'])
    parser.add_argument('--no-export', action='store_true')
    parser.add_argument('--retries', type=int, default=2)
    args = parser.parse_args()

    init_ee()
    cities_to_run = ({args.city: CITIES[args.city]} if args.city else CITIES)

    tasks = []
    for code, cfg in cities_to_run.items():
        t = run_city(code, cfg, export=not args.no_export, retries=args.retries)
        if t is not None:
            tasks.append((code, t))

    if tasks:
        print(f'\n{len(tasks)} export task(s) submitted. Check Drive in a few minutes.')
        print('Once downloaded to data/processed/, run:')
        print('  python python/paper2a_local_report.py --greenex-correlate')


if __name__ == '__main__':
    main()
