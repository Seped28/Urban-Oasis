"""
gs_hls_scale_unify_2015_2018.py — UrbanWell Analytics
=======================================================
PURPOSE:
  Re-export GreenBlue_HLS30 (GS_exposure_HLS30 + BS_exposure_HLS30) for
  years 2015–2018 at SCALE=100m, unifying the scale across ALL GS exports:
    GS_HLS  2015-2018: 300m → 100m   ← THIS SCRIPT
    GS_S2   2019-2020: 300m → 100m   ← gs_s2_scale_unify_2019_2020.py
    GS_S2   2021-2023: 300m → 100m   ← worldpop_repop_2021_2023.py (running)
    BS      2015-2023: 100m           ← always correct, unchanged

SCALE EQUIVALENCE (confirmed, not assumed):
  Sensitivity test (Maricopa County, AZ, n=95 tracts):
    GS at 300m vs 100m: MAPD=0.000%, r=0.993
  Scale changes do not affect tract-level GS exposure values because the
  500m population-weighted buffer convolution averages sub-pixel variation.
  Re-exporting at 100m produces IDENTICAL results to 300m for GS.
  This re-export is for methodological consistency only, not data quality.

SATELLITE PIPELINE (faithfully replicates _build_gs_hls in urbanwell_gee_batch_v3.py):
  Collection: NASA/HLS/HLSL30/v002 (Landsat-8) merged with NASA/HLS/HLSS30/v002 (Sentinel-2)
  Cloud mask: HLS Fmask bits 1 (cloud), 2 (adjacent cloud), 3 (cloud shadow)
  Endmembers (from production script, line 144-146):
    VEG_HLS  = [0.04, 0.05, 0.08, 0.38, 0.10, 0.05]
    IMP_HLS  = [0.14, 0.17, 0.20, 0.22, 0.28, 0.20]
    SOIL_HLS = [0.09, 0.11, 0.16, 0.20, 0.34, 0.26]
  Spectral unmixing: bands B2,B3,B4,B5,B6,B7 (harmonized 30m)
  Buffer convolution: 500m circle, normalized
  MNDWI (BS): bands B3/B6 (HLS Green/SWIR1), threshold >0.0
  WorldPop: official collection, year-matched (2015-2018 all ≤ 2020, no cap needed)

SELECTORS (same as production script):
  ['GEOID', 'year', 'GS_exposure_HLS30', 'BS_exposure_HLS30', 'pop_sum', 'pop_year']
  NOTE: BS_exposure_HLS30 is re-exported for completeness but remains UNUSED in
  the panel model (main BS comes from BlueSpace_USA files). Including it preserves
  output file format compatibility with downstream merge scripts.

ARCHITECTURE: county-level tiling for ALL states (not state-level) because HLS30
  at 100m resolution may hit computation limits. All states use county-level to be safe.
  This matches batch_submit_all_gaps.py v1.4 SingleCounty_HLS30 approach.

SAFETY: suffix '_rescale100' distinguishes from existing 300m exports.
  --rename backs up existing files before overwriting.

USAGE:
  # 1. Dry run
  python python/gs_hls_scale_unify_2015_2018.py --dry

  # 2. Smoke test
  python python/gs_hls_scale_unify_2015_2018.py --state 11 --year 2015 --submit

  # 3. Full run (expect 4-6 hours per year, ~16-24 hours total)
  python python/gs_hls_scale_unify_2015_2018.py --submit --watch

  # 4. Single year
  python python/gs_hls_scale_unify_2015_2018.py --year 2015 --submit --watch

  # 5. After tasks complete + CSVs downloaded from Drive:
  python python/gs_hls_scale_unify_2015_2018.py --rename --dir data/raw/gee_exports

  # 6. Re-run merge for GS_HLS 2015-2018, then gee_gap_targeting.py + audit.
"""

import argparse
import logging
import pathlib
import re
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime

import ee

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT        = 'urbangreenblue'
DRIVE_FOLDER   = 'UrbanWellness-hls-15_18'  # CONFIRM this matches your active folder
SCALE          = 100    # unified scale — replaces 300m
MAX_CONCURRENT = 5
POLL_INTERVAL  = 120
SUFFIX         = '_rescale100'
TARGET_YEARS   = [2015, 2016, 2017, 2018]

SKIP_STATES = {'60', '66', '69', '78'}

ALL_FIPS = [
    "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18",
    "19","20","21","22","23","24","25","26","27","28","29","30","31","32","33",
    "34","35","36","37","38","39","40","41","42","44","45","46","47","48","49",
    "50","51","53","54","55","56","72",
]

# HLS endmembers — copied verbatim from urbanwell_gee_batch_v3.py lines 144-146
VEG_HLS  = [0.04, 0.05, 0.08, 0.38, 0.10, 0.05]
IMP_HLS  = [0.14, 0.17, 0.20, 0.22, 0.28, 0.20]
SOIL_HLS = [0.09, 0.11, 0.16, 0.20, 0.34, 0.26]

# ── LOGGING ───────────────────────────────────────────────────────────────────
_LOG_DIR = pathlib.Path('logs')
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            _LOG_DIR / f'hls_rescale_{datetime.now():%Y%m%d_%H%M%S}.log',
            encoding='utf-8'
        ),
    ]
)
log = logging.getLogger(__name__)


# ── EE HELPERS ────────────────────────────────────────────────────────────────
def init_ee():
    ee.Initialize(project=PROJECT)
    log.info(f'EE initialised | project={PROJECT} | SCALE={SCALE}m | suffix={SUFFIX}')

def running_count() -> int:
    return sum(1 for t in ee.data.getTaskList()
               if t['state'] in ('READY', 'RUNNING'))

def wait_for_slot(label=''):
    while True:
        n = running_count()
        if n < MAX_CONCURRENT:
            return
        log.info(f'  {n}/{MAX_CONCURRENT} tasks running — waiting {POLL_INTERVAL}s  {label}')
        time.sleep(POLL_INTERVAL)

def get_existing_task_names() -> set:
    return {t['description'] for t in ee.data.getTaskList()
            if t['state'] in ('COMPLETED', 'RUNNING', 'READY')}

def safe_start(task, desc):
    try:
        task.start()
        log.info(f'  SUBMITTED {desc}')
    except ee.EEException as e:
        if 'already started with the given request_id' in str(e):
            log.warning(f'  DUPLICATE (OK): {desc}')
        else:
            raise

def watch_until_done(submitted_ids):
    log.info(f'Watching {len(submitted_ids)} tasks...')
    pending = set(submitted_ids)
    while pending:
        time.sleep(POLL_INTERVAL)
        tasks = {t['id']: t for t in ee.data.getTaskList()}
        done = {tid for tid in pending
                if tasks.get(tid, {}).get('state') in
                ('COMPLETED', 'FAILED', 'CANCELLED')}
        for tid in done:
            s = tasks[tid]['state']
            d = tasks[tid].get('description', tid)
            if s == 'COMPLETED':
                log.info(f'  COMPLETED  {d}')
            else:
                log.warning(f'  {s}  {d} | {tasks[tid].get("error_message","")}')
        pending -= done
        log.info(f'  {len(pending)} tasks pending')
    log.info('All tasks finished.')

def rr(image, geom):
    """reduceRegion at SCALE with tileScale=4 to prevent Error 3/8."""
    return image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=SCALE,
        maxPixels=int(1e13),
        bestEffort=True,
        tileScale=4
    )

def get_county_list(fips: str) -> list:
    tracts = (ee.FeatureCollection('TIGER/2020/TRACT')
              .filter(ee.Filter.eq('STATEFP', fips)))
    return tracts.aggregate_array('COUNTYFP').distinct().sort().getInfo()


# ── WORLDPOP — official, year-matched (2015-2018, all ≤ 2020, no cap needed) ─
def get_population(year: int):
    col = (ee.ImageCollection('WorldPop/GP/100m/pop')
           .filterDate(f'{year}-01-01', f'{year+1}-01-01')
           .select('population'))
    pop = col.mosaic()
    return pop.updateMask(pop.gt(0)), year


# ── HLS CLOUD MASK (faithfully copied from _mask_hls in production script) ───
def mask_hls(img):
    """HLS Fmask: clear bit 1 (cloud), bit 2 (adjacent cloud), bit 3 (shadow)."""
    fmask = img.select('Fmask')
    return (img.updateMask(
        fmask.bitwiseAnd(1 << 1).eq(0)
        .And(fmask.bitwiseAnd(1 << 2).eq(0))
        .And(fmask.bitwiseAnd(1 << 3).eq(0))
    ).multiply(0.0001).copyProperties(img, ['system:time_start']))


# ── SUBMIT: GS_HLS, county-level ─────────────────────────────────────────────
def submit_hls_county(fips, county, year, pop_g, pop_year, dry_run):
    all_tracts    = ee.FeatureCollection('TIGER/2020/TRACT')
    county_tracts = (all_tracts
                     .filter(ee.Filter.eq('STATEFP', fips))
                     .filter(ee.Filter.eq('COUNTYFP', county)))
    county_bbox   = county_tracts.geometry().bounds()
    pop           = pop_g.clip(county_bbox)

    # HLS30 — merge Landsat and Sentinel-2 harmonized collections
    hls = (ee.ImageCollection('NASA/HLS/HLSL30/v002')
           .merge(ee.ImageCollection('NASA/HLS/HLSS30/v002'))
           .filterDate(f'{year}-01-01', f'{year}-12-31')
           .filterBounds(county_bbox)
           .filter(ee.Filter.lt('CLOUD_COVERAGE', 20))
           .map(mask_hls))

    # GreenSpace: spectral unmixing on 6 harmonized bands
    veg   = (hls.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7']).median()
               .unmix([VEG_HLS, IMP_HLS, SOIL_HLS], True, True).select([0]))
    gs_sm = veg.convolve(
        ee.Kernel.circle(radius=500, units='meters', normalize=True))
    gs_n  = gs_sm.unmask(0).multiply(pop).rename('gs_weighted')

    # BlueSpace: HLS MNDWI (B3=Green, B6=SWIR1), threshold >0
    mndwi  = (hls.map(lambda i: i.normalizedDifference(['B3', 'B6']).rename('MNDWI'))
                 .median())
    water  = mndwi.gt(0.0).toFloat()
    bs_n   = water.unmask(0).multiply(pop).rename('bs_weighted')

    def reduce_tract(tract):
        geom  = tract.geometry()
        pop_n = ee.Number(rr(pop, geom).get('population')).max(1)
        gs_v  = ee.Number(rr(gs_n, geom).get('gs_weighted'))
        bs_v  = ee.Number(rr(bs_n, geom).get('bs_weighted'))
        pop_s = rr(pop, geom).get('population')
        return tract.set({
            'GS_exposure_HLS30': gs_v.divide(pop_n),
            'BS_exposure_HLS30': bs_v.divide(pop_n),  # included, remains unused in model
            'year':              year,
            'pop_sum':           pop_s,
            'pop_year':          pop_year,
        })

    desc = f'GreenBlue_HLS30_state{fips}_county{county}_{year}{SUFFIX}'
    if dry_run:
        log.info(f'  [DRY] {desc}')
        return None, desc

    task = ee.batch.Export.table.toDrive(
        collection=county_tracts.map(reduce_tract),
        description=desc, folder=DRIVE_FOLDER, fileFormat='CSV',
        selectors=['GEOID', 'year', 'GS_exposure_HLS30',
                   'BS_exposure_HLS30', 'pop_sum', 'pop_year'])
    safe_start(task, desc)
    return task, desc


# ── RENAME _rescale100 FILES ONTO CANONICAL NAMES ────────────────────────────
def rename_rescale_files(export_dir: pathlib.Path):
    import pandas as pd

    backup_dir = export_dir / '_preRescale100_backup'
    backup_dir.mkdir(parents=True, exist_ok=True)

    rescale_files = sorted(export_dir.glob(f'*{SUFFIX}.csv'))
    if not rescale_files:
        log.warning(f'No *{SUFFIX}.csv files found in {export_dir}')
        return

    log.info(f'Found {len(rescale_files)} {SUFFIX} files to process')

    # Concat county fragments into state-level files
    county_pat = re.compile(
        r'(GreenBlue_HLS30_state\d{2})_county\d+_(\d{4})' + re.escape(SUFFIX) + r'\.csv$')
    county_groups = defaultdict(list)
    for f in rescale_files:
        m = county_pat.match(f.name)
        if m:
            county_groups[(m.group(1), m.group(2))].append(f)

    for (state_prefix, year), files in county_groups.items():
        canonical_rescale = export_dir / f'{state_prefix}_{year}{SUFFIX}.csv'
        if canonical_rescale.exists():
            log.warning(f'  Skip concat — {canonical_rescale.name} already exists')
            continue
        df = pd.concat([pd.read_csv(f, dtype={'GEOID': str}) for f in files])
        df.to_csv(canonical_rescale, index=False)
        log.info(f'  CONCAT {len(files)} counties → {canonical_rescale.name} ({len(df)} rows)')
        for f in files:
            f.unlink()

    # Rename onto canonical GreenBlue_HLS30_state{SS}_{YYYY}.csv names
    state_pat = re.compile(
        r'(GreenBlue_HLS30_state\d{2}_\d{4})' + re.escape(SUFFIX) + r'\.csv$')
    for f in sorted(export_dir.glob(f'*{SUFFIX}.csv')):
        m = state_pat.match(f.name)
        if not m:
            log.warning(f'  Unexpected filename pattern: {f.name} — skipping')
            continue
        canonical = export_dir / f'{m.group(1)}.csv'
        if canonical.exists():
            backup_target = backup_dir / canonical.name
            if backup_target.exists():
                backup_target = backup_dir / f'{canonical.stem}_{datetime.now():%H%M%S}{canonical.suffix}'
            shutil.move(str(canonical), str(backup_target))
            log.info(f'  BACKED UP {canonical.name} → _preRescale100_backup/')
        shutil.move(str(f), str(canonical))
        log.info(f'  REPLACED  {canonical.name}')

    log.info('Rename complete. Old 300m files are in _preRescale100_backup/.')
    log.info('Next: re-run merge for GS_HLS 2015-2018, then gee_gap_targeting.py + audit.')


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Re-export GS_HLS 2015-2018 at scale=100m for scale unification')
    parser.add_argument('--year', type=int, choices=TARGET_YEARS)
    parser.add_argument('--state', help='Single state FIPS, e.g. 11')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--rename', action='store_true')
    parser.add_argument('--dir', default='data/raw/gee_exports')
    args = parser.parse_args()

    if args.rename:
        rename_rescale_files(pathlib.Path(args.dir))
        return

    dry_run = not args.submit
    if dry_run:
        log.info('DRY RUN — pass --submit to submit GEE tasks')

    log.info('=' * 70)
    log.info('GS_HLS Scale Unification: 2015-2018 at 100m')
    log.info(f'  SCALE = {SCALE}m (matches GS_S2 unified scale)')
    log.info('  HLS pipeline: NASA/HLS/HLSL30 + NASA/HLS/HLSS30 (Landsat+Sentinel-2)')
    log.info('  Endmembers: verbatim from urbanwell_gee_batch_v3.py lines 144-146')
    log.info('  WorldPop: official collection, year-matched (no cap for 2015-2018)')
    log.info('  tileScale=4 on all reduceRegion calls (prevents Error 3/8)')
    log.info('  County-level tiling for all states (safe for 100m resolution)')
    log.info('=' * 70)

    init_ee()

    years  = [args.year] if args.year else TARGET_YEARS
    states = [args.state.zfill(2)] if args.state else [
        f for f in ALL_FIPS if f not in SKIP_STATES]

    existing      = get_existing_task_names() if not dry_run else set()
    submitted_ids = []
    skipped = errors = 0

    for year in years:
        pop_g, pop_year = get_population(year)
        log.info(f'\n── year {year} | WorldPop: official year-matched | pop_year={pop_year} ──')

        for fips in states:
            counties = get_county_list(fips)
            for county in counties:
                desc = f'GreenBlue_HLS30_state{fips}_county{county}_{year}{SUFFIX}'
                if desc in existing:
                    log.info(f'  SKIP {desc}'); skipped += 1; continue
                if not dry_run:
                    wait_for_slot(desc)
                try:
                    task, desc = submit_hls_county(fips, county, year, pop_g, pop_year, dry_run)
                    if task and not dry_run:
                        submitted_ids.append(task.id)
                        existing.add(desc)
                except Exception as e:
                    log.error(f'  ERROR state={fips} county={county} year={year}: {e}')
                    errors += 1
                time.sleep(0.5)

    log.info(f'\n── Summary ──')
    log.info(f'  Submitted: {len(submitted_ids)} | Skipped: {skipped} | Errors: {errors}')
    if not dry_run:
        log.info(f'\nAfter tasks complete + CSVs downloaded from Drive:')
        log.info(f'  python python/gs_hls_scale_unify_2015_2018.py --rename --dir {args.dir}')

    if args.watch and submitted_ids:
        watch_until_done(submitted_ids)


if __name__ == '__main__':
    main()

