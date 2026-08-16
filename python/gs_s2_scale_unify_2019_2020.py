"""
gs_s2_scale_unify_2019_2020.py — UrbanWell Analytics
======================================================
PURPOSE:
  Re-export GS_S2 for years 2019 and 2020 at SCALE_GS_S2=100m, using the
  official WorldPop/GP/100m/pop collection (which is year-matched and
  available for both years).

WHY THIS IS NEEDED (June 2026):
  The worldpop_repop_2021_2023.py script (currently running) exports 2021-2023
  GS_S2 at SCALE_GS_S2=100m using the Community Catalog WorldPop. To have
  a uniform scale across ALL GS_S2 years (2019-2023), the existing 2019-2020
  exports (produced at SCALE_GS=300m by urbanwell_gee_batch_v3.py) need to
  be re-run at 100m. This script does exactly that.

WHY SCALE=100 IS NOW THE STANDARD FOR GS_S2:
  The project originally used SCALE_GS=300 across all GS exports. When Error
  3/8 (EECU timeout) appeared for certain large states in GS_S2, the team ran
  a sensitivity analysis and found MAPD=0.000%, r=0.993 between 100m and 300m
  for GS_S2. Scale=100m was used as a workaround for gap-fill exports in
  batch_submit_all_gaps.py v1.4. The worldpop_repop_2021_2023.py ran at 100m.
  Since Error 3/8 has NOT reappeared with scale=100, and the sensitivity
  analysis confirms both scales produce identical results, scale=100 is now
  the standard for ALL GS_S2 exports. HLS remains at 300m (different pipeline).

OFFICIAL WORLDPOP for 2019 and 2020 (confirmed available):
  WorldPop/GP/100m/pop: 2019 → ✓ | 2020 → ✓ (249 images)
  No fallback needed. These years are fully covered by the official collection.

SCALE NOTE:
  GS_HLS (2015-2018) remains at 300m — this is correct and deliberate.
  Sensitivity analysis confirmed MAPD=0.000% between 100m and 300m for GS,
  so HLS at 300m and GS_S2 at 100m produce numerically identical exposures.
  No re-export of HLS is needed or planned.

SAFETY:
  Same _rescale suffix pattern as worldpop_repop_2021_2023.py. Files land
  with suffix _rescale.csv; --rename step backs up old 300m exports before
  overwriting. Old files go to _preRescale_backup/, never deleted.

STATES NEEDING COUNTY-LEVEL TILING:
  GS_S2_COUNTY_STATES = {02, 06, 08, 20, 32, 35, 48}
  Same set as batch_submit_all_gaps.py v1.4. Error 3/8 was resolved at
  scale=100 but county-level tiling is retained for these states as insurance.

USAGE:
  # 1. Dry run first — confirm task list before spending quota
  python python/gs_s2_scale_unify_2019_2020.py --dry

  # 2. Smoke test on one cheap state+year
  python python/gs_s2_scale_unify_2019_2020.py --state 11 --year 2019 --submit

  # 3. Full run (both years, all states)
  python python/gs_s2_scale_unify_2019_2020.py --submit --watch

  # 4. Single year only
  python python/gs_s2_scale_unify_2019_2020.py --year 2019 --submit --watch

  # 5. After all tasks complete and CSVs downloaded from Drive:
  python python/gs_s2_scale_unify_2019_2020.py --rename --dir data/raw/gee_exports

  # 6. Re-run merge + gee_gap_targeting.py + audit_gee_exports.py to confirm clean landing.
  # 7. Re-run 00_data_prep_v3.R and confirm GS_S2 pop_year and scale diagnostics.

DRIVE_FOLDER note: defaults to 'UrbanWellExports'. Change if your production
  exports go to 'UrbanWellness'.
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
DRIVE_FOLDER   = 'UrbanWellness-hls-19_20'
SCALE_GS_S2    = 100    # unified scale for all GS_S2 exports going forward
MAX_CONCURRENT = 5
POLL_INTERVAL  = 120
SUFFIX         = '_rescale'
TARGET_YEARS   = [2019, 2020]

SKIP_STATES = {'60', '66', '69', '78'}   # territories — no PLACES data

ALL_FIPS = [
    "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18",
    "19","20","21","22","23","24","25","26","27","28","29","30","31","32","33",
    "34","35","36","37","38","39","40","41","42","44","45","46","47","48","49",
    "50","51","53","54","55","56","72",
]

# States requiring county-level tiling — same as batch_submit_all_gaps.py v1.4
GS_S2_COUNTY_STATES = {'02', '06', '08', '20', '32', '35', '48'}

# ── LOGGING ───────────────────────────────────────────────────────────────────
_LOG_DIR = pathlib.Path('logs')
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            _LOG_DIR / f'gs_s2_rescale_{datetime.now():%Y%m%d_%H%M%S}.log',
            encoding='utf-8'
        ),
    ]
)
log = logging.getLogger(__name__)


# ── EE HELPERS ────────────────────────────────────────────────────────────────
def init_ee():
    ee.Initialize(project=PROJECT)
    log.info(f'EE initialised | project={PROJECT} | SCALE_GS_S2={SCALE_GS_S2}m | suffix={SUFFIX}')

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
            log.warning(f'  DUPLICATE request_id (OK): {desc}')
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

def rr(image, geom, scale):
    """Safe reduceRegion — explicit kwargs, tileScale=4 (prevents Error 3/8)."""
    return image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=scale,
        maxPixels=int(1e13),
        bestEffort=True,
        tileScale=4
    )

def get_county_list(fips: str) -> list:
    tracts = (ee.FeatureCollection('TIGER/2020/TRACT')
              .filter(ee.Filter.eq('STATEFP', fips)))
    return tracts.aggregate_array('COUNTYFP').distinct().sort().getInfo()


# ── WORLDPOP — official collection for 2019 and 2020 ─────────────────────────
def get_population_official(year: int):
    """Official WorldPop/GP/100m/pop — year-matched for 2019 and 2020."""
    col = (ee.ImageCollection('WorldPop/GP/100m/pop')
           .filterDate(f'{year}-01-01', f'{year+1}-01-01')
           .select('population'))
    pop = col.mosaic()
    return pop.updateMask(pop.gt(0)), year


# ── S2 CLOUD MASK ─────────────────────────────────────────────────────────────
VEG_S2  = [0.05, 0.04, 0.09, 0.40, 0.08, 0.03]
IMP_S2  = [0.15, 0.18, 0.22, 0.24, 0.30, 0.21]
SOIL_S2 = [0.10, 0.12, 0.18, 0.22, 0.36, 0.28]

def mask_s2(img):
    qa = img.select('QA60')
    return (img.updateMask(
        qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    ).divide(10000)
     .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
     .copyProperties(img, ['system:time_start']))


# ── SUBMIT: GS_S2, state-level ────────────────────────────────────────────────
def submit_gs_s2_state(fips, year, pop_g, pop_year, dry_run):
    all_tracts   = ee.FeatureCollection('TIGER/2020/TRACT')
    state_tracts = all_tracts.filter(ee.Filter.eq('STATEFP', fips))
    state_bbox   = state_tracts.geometry().bounds()
    pop          = pop_g.clip(state_bbox)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filterBounds(state_bbox)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .map(mask_s2).median())

    veg    = s2.unmix([VEG_S2, IMP_S2, SOIL_S2], True, True).select([0]).rename('veg_frac')
    gs     = veg.convolve(ee.Kernel.circle(radius=500, units='meters', normalize=True))
    gs_num = gs.unmask(0).multiply(pop).rename('gs_weighted')

    def reduce_tract(tract):
        geom  = tract.geometry()
        pop_n = ee.Number(rr(pop, geom, SCALE_GS_S2).get('population')).max(1)
        gs_n  = ee.Number(rr(gs_num, geom, SCALE_GS_S2).get('gs_weighted'))
        pop_s = rr(pop, geom, SCALE_GS_S2).get('population')
        return tract.set({
            'GS_exposure': gs_n.divide(pop_n),
            'GS_year':     year,
            'pop_sum':     pop_s,
            'pop_year':    pop_year,
        })

    desc = f'GreenSpace_S2_state{fips}_{year}{SUFFIX}'
    if dry_run:
        log.info(f'  [DRY] {desc}')
        return None, desc
    task = ee.batch.Export.table.toDrive(
        collection=state_tracts.map(reduce_tract),
        description=desc, folder=DRIVE_FOLDER, fileFormat='CSV',
        selectors=['GEOID', 'GS_year', 'GS_exposure', 'pop_sum', 'pop_year'])
    safe_start(task, desc)
    return task, desc


# ── SUBMIT: GS_S2, county-level fallback ──────────────────────────────────────
def submit_gs_s2_county(fips, county, year, pop_g, pop_year, dry_run):
    all_tracts    = ee.FeatureCollection('TIGER/2020/TRACT')
    county_tracts = (all_tracts
                     .filter(ee.Filter.eq('STATEFP', fips))
                     .filter(ee.Filter.eq('COUNTYFP', county)))
    county_bbox   = county_tracts.geometry().bounds()
    pop           = pop_g.clip(county_bbox)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filterBounds(county_bbox)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .map(mask_s2).median())

    veg    = s2.unmix([VEG_S2, IMP_S2, SOIL_S2], True, True).select([0]).rename('veg_frac')
    gs     = veg.convolve(ee.Kernel.circle(radius=500, units='meters', normalize=True))
    gs_num = gs.unmask(0).multiply(pop).rename('gs_weighted')

    def reduce_tract(tract):
        geom  = tract.geometry()
        pop_n = ee.Number(rr(pop, geom, SCALE_GS_S2).get('population')).max(1)
        gs_n  = ee.Number(rr(gs_num, geom, SCALE_GS_S2).get('gs_weighted'))
        pop_s = rr(pop, geom, SCALE_GS_S2).get('population')
        return tract.set({
            'GS_exposure': gs_n.divide(pop_n),
            'GS_year':     year,
            'pop_sum':     pop_s,
            'pop_year':    pop_year,
        })

    desc = f'GreenSpace_S2_state{fips}_county{county}_{year}{SUFFIX}'
    if dry_run:
        log.info(f'  [DRY] {desc}')
        return None, desc
    task = ee.batch.Export.table.toDrive(
        collection=county_tracts.map(reduce_tract),
        description=desc, folder=DRIVE_FOLDER, fileFormat='CSV',
        selectors=['GEOID', 'GS_year', 'GS_exposure', 'pop_sum', 'pop_year'])
    safe_start(task, desc)
    return task, desc


# ── RENAME _rescale FILES ONTO CANONICAL NAMES ────────────────────────────────
def rename_rescale_files(export_dir: pathlib.Path):
    """
    After downloading *_rescale.csv files from Drive, this:
      1. Concatenates county-level _rescale fragments per (state, year)
         into a single state-level file.
      2. Backs up the existing canonical file to _preRescale_backup/.
      3. Moves _rescale file onto canonical name.
    Never deletes old files.
    """
    import pandas as pd

    backup_dir = export_dir / '_preRescale_backup'
    backup_dir.mkdir(parents=True, exist_ok=True)

    rescale_files = sorted(export_dir.glob('*_rescale.csv'))
    if not rescale_files:
        log.warning(f'No *_rescale.csv files found in {export_dir}')
        return

    log.info(f'Found {len(rescale_files)} _rescale files to process')

    # Step 1: concat county fragments
    county_pat = re.compile(r'(GreenSpace_S2_state\d{2})_county\d+_(\d{4})_rescale\.csv$')
    county_groups = defaultdict(list)
    for f in rescale_files:
        m = county_pat.match(f.name)
        if m:
            county_groups[(m.group(1), m.group(2))].append(f)

    for (state_prefix, year), files in county_groups.items():
        canonical_rescale = export_dir / f'{state_prefix}_{year}_rescale.csv'
        if canonical_rescale.exists():
            log.warning(f'  Skip concat: state-level rescale exists for {state_prefix}_{year}')
            continue
        df = pd.concat([pd.read_csv(f, dtype={'GEOID': str}) for f in files])
        df.to_csv(canonical_rescale, index=False)
        log.info(f'  CONCAT {len(files)} county fragments → {canonical_rescale.name} ({len(df)} rows)')
        for f in files:
            f.unlink()

    # Step 2: rename
    for f in sorted(export_dir.glob('*_rescale.csv')):
        canonical = export_dir / f.name.replace('_rescale', '')
        if canonical.exists():
            backup_target = backup_dir / canonical.name
            if backup_target.exists():
                backup_target = backup_dir / f'{canonical.stem}_{datetime.now():%Y%m%d%H%M%S}{canonical.suffix}'
            shutil.move(str(canonical), str(backup_target))
            log.info(f'  BACKED UP {canonical.name} → _preRescale_backup/{backup_target.name}')
        shutil.move(str(f), str(canonical))
        log.info(f'  REPLACED  {canonical.name}')

    log.info('Rename complete. Old (300m) files are in _preRescale_backup/.')
    log.info('Next: re-run merge step for GS_S2 2019-2020, then gee_gap_targeting.py and audit.')


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Re-export GS_S2 2019-2020 at scale=100m (scale unification)')
    parser.add_argument('--year', type=int, choices=TARGET_YEARS,
                        help='Single year only (default: both 2019 and 2020)')
    parser.add_argument('--state', help='Single state FIPS, e.g. 13')
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
    log.info('GS_S2 Scale Unification: 2019-2020 at 100m')
    log.info(f'  SCALE_GS_S2 = {SCALE_GS_S2}m (matches worldpop_repop_2021_2023.py)')
    log.info('  WorldPop: official collection (year-matched, confirmed available)')
    log.info('  tileScale=4 on all reduceRegion calls (prevents Error 3/8)')
    log.info('  County tiling for: AK(02) CA(06) CO(08) KS(20) NV(32) NM(35) TX(48)')
    log.info('=' * 70)

    init_ee()

    years  = [args.year] if args.year else TARGET_YEARS
    states = [args.state.zfill(2)] if args.state else [
        f for f in ALL_FIPS if f not in SKIP_STATES]

    existing      = get_existing_task_names() if not dry_run else set()
    submitted_ids = []
    skipped = errors = 0

    for year in years:
        pop_g, pop_year = get_population_official(year)
        log.info(f'\n── year {year} | WorldPop: official (year-matched) | pop_year={pop_year} ──')

        for fips in states:
            if fips in GS_S2_COUNTY_STATES:
                log.info(f'  {fips}: county-level tiling')
                counties = get_county_list(fips)
                for county in counties:
                    desc = f'GreenSpace_S2_state{fips}_county{county}_{year}{SUFFIX}'
                    if desc in existing:
                        log.info(f'  SKIP {desc}'); skipped += 1; continue
                    if not dry_run:
                        wait_for_slot(desc)
                    try:
                        task, desc = submit_gs_s2_county(fips, county, year, pop_g, pop_year, dry_run)
                        if task and not dry_run:
                            submitted_ids.append(task.id)
                            existing.add(desc)
                    except Exception as e:
                        log.error(f'  ERROR county state={fips} county={county} year={year}: {e}')
                        errors += 1
                    time.sleep(0.5)
            else:
                desc = f'GreenSpace_S2_state{fips}_{year}{SUFFIX}'
                if desc in existing:
                    log.info(f'  SKIP {desc}'); skipped += 1; continue
                if not dry_run:
                    wait_for_slot(desc)
                try:
                    task, desc = submit_gs_s2_state(fips, year, pop_g, pop_year, dry_run)
                    if task and not dry_run:
                        submitted_ids.append(task.id)
                        existing.add(desc)
                except Exception as e:
                    log.error(f'  ERROR state={fips} year={year}: {e}')
                    errors += 1
                time.sleep(0.5)

    log.info(f'\n── Summary ──')
    log.info(f'  Submitted: {len(submitted_ids)} | Skipped: {skipped} | Errors: {errors}')

    if not dry_run:
        log.info('\nAfter tasks complete and CSVs downloaded from Drive:')
        log.info(f'  python python/gs_s2_scale_unify_2019_2020.py --rename --dir {args.dir}')
        log.info('  Then re-run merge step, gee_gap_targeting.py, audit_gee_exports.py')

    if args.watch and submitted_ids:
        watch_until_done(submitted_ids)


if __name__ == '__main__':
    main()
