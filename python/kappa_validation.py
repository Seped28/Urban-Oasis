"""
kappa_validation.py — UrbanWell Analytics

Validates dual-sensor (Sentinel-2 MNDWI + Sentinel-1 SAR) bluespace
classification against NAIP aerial-imagery reference, reporting Cohen's
kappa and a full confusion matrix per county.

Water is classified where MNDWI > threshold AND/OR SAR VV backscatter is
below a decibel threshold, depending on --mode. The default dual-sensor
AND gate is more conservative than either sensor alone; the alternate
modes below exist because inland, tree-canopied water features can cause
SAR double-bounce backscatter (VV reads high, not low), making the
default gate systematically under-detect water in heavily forested
counties. This is a structural property of SAR over canopy, not a
tunable threshold error -- running all four modes for a given county
is the way to distinguish the two.

Modes:
  --mode mndwi-only     : MNDWI alone, no SAR gate
  --sar-threshold -12   : relaxed SAR threshold (default is -18 dB)
  --mode or             : water = MNDWI>0 OR VV<threshold (more inclusive)
  --scale 30            : coarser scale, useful for large counties prone
                           to GEE memory-limit timeouts at finer scale

Implementation notes:
  - Sentinel-2 surface reflectance (S2_SR_HARMONIZED, L2A), not top-of-atmosphere.
  - QA60 pixel-level cloud mask (bits 10 and 11).
  - MNDWI = (Green - SWIR1) / (Green + SWIR1), not the standard NDWI formula.
  - AOI is derived from TIGER/2020/TRACT dissolve rather than
    TIGER/2020/COUNTY, which isn't available in Earth Engine's catalog.
  - NAIP reference year is auto-detected to the most recent year actually
    available for each county, not assumed to be the panel's target year.
  - A retry wrapper handles transient GEE errors with backoff.

Setup:
  pip install earthengine-api --break-system-packages
  earthengine authenticate
  earthengine set_project urbangreenblue

Usage:
  # Compare all four detection modes for one county:
  python kappa_validation.py --city ATL --mode mndwi-only
  python kappa_validation.py --city ATL --sar-threshold -12
  python kappa_validation.py --city ATL --mode or
  python kappa_validation.py --city LAX --scale 30

  # Full three-county run with default settings:
  python kappa_validation.py

  # Save output:
  python kappa_validation.py > results/kappa_results.txt 2>&1
"""

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime

import ee

# ── CONFIG (defaults — all overridable via CLI) ───────────────────────────────
PROJECT            = 'urbangreenblue'
N_PER_CLASS        = 500
SAMPLE_SCALE       = 10        # metres; LAX use 30 via --scale 30
SEED               = 42
S2_YEAR            = 2023
SAR_VV_THRESHOLD   = -18.0    # dB — overridable via --sar-threshold
MNDWI_THRESHOLD    = 0.0
NAIP_NDWI_THRESHOLD = 0.0

# Study city definitions — tile_scale and sample_scale are per-city overrides
CITIES = {
    'ATL': {
        'name':        'Atlanta (Fulton County, GA)',
        'state':       '13',
        'county':      '121',
        'geoid_prefix':'13121',
        'tile_scale':   8,    # stratifiedSample tileScale (GEE server-side)
        'sample_scale': None, # None = use global SAMPLE_SCALE
    },
    'MIA': {
        'name':        'Miami (Miami-Dade County, FL)',
        'state':       '12',
        'county':      '086',
        'geoid_prefix':'12086',
        'tile_scale':   8,
        'sample_scale': None,
    },
    'LAX': {
        'name':        'Los Angeles (LA County, CA)',
        'state':       '06',
        'county':      '037',
        'geoid_prefix':'06037',
        'tile_scale':  16,    # large counties need a higher tileScale to avoid memory limits
        'sample_scale': 30,   # FIX: 10m causes timeout; 30m is reliable for LA
    },
}

OUT_PATH = pathlib.Path('data/processed/kappa_results.json')


# ── EE INIT ───────────────────────────────────────────────────────────────────

def init_ee():
    ee.Initialize(project=PROJECT)
    print(f'GEE initialised | project={PROJECT} | S2_YEAR={S2_YEAR}')


# ── AOI (derived from TIGER/2020/TRACT — TIGER/2020/COUNTY is not in GEE) ───

def get_aoi(state_fips: str, county_fips: str) -> ee.Geometry:
    """County boundary by dissolving TIGER/2020/TRACT features.
    TIGER/2020/COUNTY was removed from GEE. We dissolve all tract
    geometries for the county, which yields the identical boundary.
    """
    return (ee.FeatureCollection('TIGER/2020/TRACT')
              .filter(ee.Filter.And(
                  ee.Filter.eq('STATEFP', state_fips),
                  ee.Filter.eq('COUNTYFP', county_fips)))
              .geometry().dissolve(maxError=10))


# ── S2 CLOUD MASK (V1 + V2) ──────────────────────────────────────────────────

def mask_s2_sr(img):
    qa = img.select('QA60')
    return (img.updateMask(
        qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    ).divide(10000).copyProperties(img, ['system:time_start']))


# ── S2 MNDWI (V3, V8) ────────────────────────────────────────────────────────

def get_s2_mndwi(aoi: ee.Geometry, year: int) -> ee.Image:
    """Annual median S2 MNDWI = (B3-B11)/(B3+B11). V3: MNDWI not NDWI."""
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(f'{year}-01-01', f'{year}-12-31')
            .filterBounds(aoi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .map(mask_s2_sr))
    return s2.map(lambda i: i.normalizedDifference(['B3', 'B11']).rename('MNDWI')).median()


# ── SAR VV (V8) ───────────────────────────────────────────────────────────────

def get_sar_vv(aoi: ee.Geometry, year: int) -> ee.Image:
    """Annual median Sentinel-1 SAR VV backscatter."""
    return (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterDate(f'{year}-01-01', f'{year}-12-31')
              .filterBounds(aoi)
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .select('VV')
              .median())


# ── WATER CLASSIFIERS ────────────────────────────────────────────────────────

def get_water_mask(mndwi: ee.Image, sar_vv, mode: str,
                   sar_threshold: float) -> ee.Image:
    """
    mode options:
      'and'       — water = MNDWI>0 AND VV<sar_threshold  [default, V4]
      'mndwi-only'— water = MNDWI>0 only (diagnostic: isolates V3 fix)
      'or'        — water = MNDWI>0 OR VV<sar_threshold   (more inclusive)

    Use 'and' for production; use 'mndwi-only' and 'or' for diagnostics to
    understand the contribution of each component before finalising thresholds.
    """
    mndwi_gate = mndwi.gt(MNDWI_THRESHOLD)

    if mode == 'mndwi-only':
        # Tests the V3 fix alone (MNDWI vs NDWI) without SAR
        return mndwi_gate.rename('water_predicted').toUint8()

    if sar_vv is None:
        raise ValueError("SAR image required for 'and' and 'or' modes")

    sar_gate = sar_vv.lt(sar_threshold)

    if mode == 'or':
        # More inclusive: either MNDWI or SAR triggers water detection
        return mndwi_gate.Or(sar_gate).rename('water_predicted').toUint8()

    # Default: 'and' — dual-sensor gate
    return mndwi_gate.And(sar_gate).rename('water_predicted').toUint8()


# ── NAIP REFERENCE (V5, V7, V8) ──────────────────────────────────────────────

def get_naip_water_mask(aoi: ee.Geometry, target_year: int):
    """V7: auto-detect NAIP year. V8: median. V5: threshold=0.0."""
    naip_col = ee.ImageCollection('USDA/NAIP/DOQQ')
    naip_year_used = None
    col_used = None
    for yr in [target_year, target_year-1, target_year-2, target_year-3]:
        col = naip_col.filterBounds(aoi).filterDate(f'{yr}-01-01', f'{yr}-12-31')
        n = col.size().getInfo()
        print(f'    NAIP {yr}: {n} images in AOI')
        if n > 0:
            naip_year_used = yr
            col_used = col
            break
    if naip_year_used is None:
        raise RuntimeError('No NAIP found within 4 years — cannot compute reference mask')
    if naip_year_used != target_year:
        print(f'    WARNING: NAIP {target_year} unavailable — using {naip_year_used}. '
              f'Report this in Paper 2a Methods/Limitations.')
    naip = col_used.median()
    ndwi_naip = naip.normalizedDifference(['G', 'N']).rename('water_reference')
    water_ref = ndwi_naip.gt(NAIP_NDWI_THRESHOLD).rename('water_reference').toUint8()
    return water_ref, naip_year_used


# ── ACCURACY ──────────────────────────────────────────────────────────────────

def compute_accuracy(aoi, water_predicted, water_reference, city_name,
                     tile_scale=8, sample_scale=None):
    """Stratified sample → confusion matrix → κ."""
    scale = sample_scale if sample_scale is not None else SAMPLE_SCALE

    combined = water_predicted.rename('predicted').addBands(
        water_reference.rename('reference'))

    sample = combined.stratifiedSample(
        numPoints    = N_PER_CLASS,
        classBand    = 'reference',
        region       = aoi,
        scale        = scale,
        seed         = SEED,
        geometries   = False,
        tileScale    = tile_scale,
        dropNulls    = True,
    )
    n_sampled = sample.size().getInfo()
    print(f'    Sampled {n_sampled} points (target: {N_PER_CLASS*2}) at {scale}m scale')

    error_matrix = sample.errorMatrix('reference', 'predicted')
    matrix_array = error_matrix.array().getInfo()
    accuracy     = float(error_matrix.accuracy().getInfo())
    kappa        = float(error_matrix.kappa().getInfo())

    try:
        tn = matrix_array[0][0]; fp = matrix_array[0][1]
        fn = matrix_array[1][0]; tp = matrix_array[1][1]
        precision = tp / (tp+fp) if (tp+fp) > 0 else float('nan')
        recall    = tp / (tp+fn) if (tp+fn) > 0 else float('nan')
        f1        = (2*precision*recall/(precision+recall)
                     if (precision+recall) > 0 else float('nan'))
    except (IndexError, TypeError):
        tn=fp=fn=tp=precision=recall=f1 = float('nan')
        print(f'    WARNING: Cannot parse matrix: {matrix_array}')

    return {
        'city':              city_name,
        'n_per_class':       N_PER_CLASS,
        'n_sampled':         n_sampled,
        'sample_scale_m':    scale,
        'seed':              SEED,
        'overall_accuracy':  round(accuracy*100, 2),
        'kappa':             round(kappa, 4),
        'precision_water':   round(precision*100, 2) if precision==precision else None,
        'recall_water':      round(recall*100, 2) if recall==recall else None,
        'f1_water':          round(f1, 4) if f1==f1 else None,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'confusion_matrix':  matrix_array,
    }


# ── RUN ONE CITY ──────────────────────────────────────────────────────────────

def run_city(city_code: str, city_cfg: dict, mode: str,
             sar_threshold: float) -> dict:
    print(f'\n{"="*70}')
    print(f'CITY: {city_cfg["name"]} | mode={mode} | SAR threshold={sar_threshold} dB')
    print(f'{"="*70}')

    aoi = get_aoi(city_cfg['state'], city_cfg['county'])

    # Resolve NAIP reference year first, then build the predicted layer on the same year
    print('  Checking NAIP availability:')
    water_reference, naip_year = get_naip_water_mask(aoi, S2_YEAR)

    predicted_year = naip_year  # always match predicted year to reference year
    if predicted_year != S2_YEAR:
        print(f'    Year-matching: predicted layer will use {predicted_year} '
              f'(NAIP {S2_YEAR} unavailable). Both layers on same year = valid comparison.')

    print(f'  Getting S2 MNDWI ({predicted_year})...')
    mndwi = get_s2_mndwi(aoi, predicted_year)

    sar_vv = None
    if mode != 'mndwi-only':
        print(f'  Getting S1 SAR VV ({predicted_year})...')
        sar_vv = get_sar_vv(aoi, predicted_year)

    print(f'  Building water mask (mode={mode})...')
    water_predicted = get_water_mask(mndwi, sar_vv, mode, sar_threshold)

    tile_scale   = city_cfg.get('tile_scale', 8)
    sample_scale = city_cfg.get('sample_scale', None)  # None → global SAMPLE_SCALE

    print(f'  Computing accuracy (n={N_PER_CLASS}/class, '
          f'scale={sample_scale or SAMPLE_SCALE}m, tileScale={tile_scale})...')
    t0 = time.time()
    result = compute_accuracy(aoi, water_predicted, water_reference,
                              city_cfg['name'], tile_scale, sample_scale)
    elapsed = round(time.time()-t0, 1)

    result.update({
        'city_code':      city_code,
        'mode':           mode,
        'sar_threshold':  sar_threshold,
        's2_year_target': S2_YEAR,
        's2_year_used':   predicted_year,
        'naip_year':      naip_year,
        'year_matched':   (predicted_year == naip_year),
        'elapsed_sec':    elapsed,
        'classifier':     (f'S2_MNDWI(B3/B11)>{MNDWI_THRESHOLD}'
                           + (f' {mode.upper()} S1_VV<{sar_threshold}dB'
                              if mode != 'mndwi-only' else ' ONLY')),
        'reference':      f'NAIP_NDWI(G/N)>{NAIP_NDWI_THRESHOLD}',
        'v_corrections':  ['V1','V2','V3','V4','V5','V6','V7','V8'],
    })

    print(f'\n  ── Results: {city_cfg["name"]} ──')
    print(f'  Overall accuracy : {result["overall_accuracy"]}%')
    print(f'  Cohen\'s κ        : {result["kappa"]}  '
          f'(Paper 1 NDWI baseline: 0.65)')
    print(f'  Precision (water): {result["precision_water"]}%')
    print(f'  Recall (water)   : {result["recall_water"]}%')
    print(f'  F1 (water)       : {result["f1_water"]}')
    print(f'  Confusion matrix : TN={result["tn"]}, FP={result["fp"]}, '
          f'FN={result["fn"]}, TP={result["tp"]}')
    print(f'  [{elapsed}s]')

    if result['kappa'] < 0.1:
        print(f'  ⚠ κ < 0.10 — DIAGNOSTIC: classifier is barely detecting water.')
        if mode == 'and':
            print(f'    Try: --mode mndwi-only   (removes SAR gate, tests MNDWI alone)')
            print(f'    Try: --sar-threshold -12  (relaxes VV threshold)')
            print(f'    Try: --mode or            (water = MNDWI>0 OR VV<{sar_threshold}dB)')

    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global N_PER_CLASS  # must be declared before first use in this function

    parser = argparse.ArgumentParser(
        description='Paper 2a kappa validation — dual-sensor MNDWI+SAR vs NAIP')
    parser.add_argument('--city', choices=['ATL','MIA','LAX'],
                        help='Single city (default: all three)')
    parser.add_argument('--mode', choices=['and','mndwi-only','or'], default='and',
                        help=('and=MNDWI AND SAR [default/production]; '
                              'mndwi-only=MNDWI alone [diagnostic, isolates V3 fix]; '
                              'or=MNDWI OR SAR [diagnostic, more inclusive]'))
    parser.add_argument('--sar-threshold', type=float, default=SAR_VV_THRESHOLD,
                        help=f'SAR VV threshold in dB (default: {SAR_VV_THRESHOLD}). '
                              'Try -12 or -14 for inland cities.')
    parser.add_argument('--scale', type=int, default=None,
                        help='Override sample scale in metres for ALL cities '
                             '(e.g. --scale 30 for LAX to avoid timeout). '
                             'Per-city defaults: ATL/MIA=10m, LAX=30m.')
    parser.add_argument('--n-per-class', type=int, default=N_PER_CLASS,
                        help=f'Validation points per class per city (default: {N_PER_CLASS})')
    parser.add_argument('--retries', type=int, default=2,
                        help='Retries on transient GEE timeout errors (default: 2)')
    parser.add_argument('--out', default=str(OUT_PATH),
                        help='JSON output path')
    args = parser.parse_args()

    N_PER_CLASS = args.n_per_class

    # Apply global scale override if given
    cities = CITIES.copy()
    if args.scale is not None:
        for cfg in cities.values():
            cfg['sample_scale'] = args.scale

    init_ee()

    cities_to_run = ({args.city: cities[args.city]} if args.city else cities)

    print(f'\nClassifier mode: {args.mode}')
    if args.mode == 'and':
        print(f'SAR VV threshold: {args.sar_threshold} dB')
        print('Production mode — dual-sensor MNDWI AND SAR gate')
    elif args.mode == 'mndwi-only':
        print('DIAGNOSTIC: MNDWI-only (no SAR gate). Isolates the V3 improvement.')
        print('If Atlanta κ improves here, the SAR -18dB threshold is the blocker.')
    elif args.mode == 'or':
        print(f'DIAGNOSTIC: MNDWI>0 OR VV<{args.sar_threshold}dB (more inclusive gate)')
        print('If Atlanta κ improves here, OR logic works better for inland cities.')

    # Transient error markers for retry logic
    TRANSIENT = ('timed out','computation timed out','too many concurrent',
                 'internal error','deadline exceeded','user memory limit')

    all_results = []
    for code, cfg in cities_to_run.items():
        r = None
        for attempt in range(1, args.retries+2):
            try:
                r = run_city(code, cfg, args.mode, args.sar_threshold)
                break
            except Exception as e:
                msg = str(e)
                is_transient = any(m in msg.lower() for m in TRANSIENT)
                if is_transient and attempt <= args.retries:
                    wait = 30 * attempt
                    print(f'  TRANSIENT ERROR ({attempt}/{args.retries+1}): '
                          f'{msg} — retry in {wait}s')
                    time.sleep(wait)
                    continue
                print(f'  ERROR for {code}: {msg}')
                r = {'city_code': code, 'mode': args.mode, 'error': msg}
                break
        all_results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'SUMMARY — mode={args.mode} | SAR_threshold={args.sar_threshold}dB')
    print(f'{"="*70}')
    print(f'{"City":<35} {"OA%":>6} {"κ":>7} {"Prec%":>7} {"Recall%":>8} {"Year":>6}')
    print('-'*75)
    for r in all_results:
        if 'error' in r:
            print(f'{r["city_code"]:<35} ERROR: {r["error"][:40]}')
            continue
        yr = str(r.get('s2_year_used','?'))
        if not r.get('year_matched', True):
            yr += '*'
        print(f'{r["city"]:<35} {r["overall_accuracy"]:>6.1f} '
              f'{r["kappa"]:>7.4f} {str(r["precision_water"] or "N/A"):>7} '
              f'{str(r["recall_water"] or "N/A"):>8} {yr:>6}')
    print()
    print('* = year-matched fallback active (target year NAIP unavailable)')
    print()
    print(f'Paper 1 baseline: OA=82.5%, κ=0.65 (single-sensor NDWI)')
    print(f'Paper 2a target:  κ ≥ 0.80 per city ("almost perfect", Landis & Koch 1977)')

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = pathlib.Path(args.out)
    # Append mode to filename for diagnostic runs to avoid overwriting
    if args.mode != 'and':
        stem = out_path.stem + f'_{args.mode}'
        out_path = out_path.with_name(stem + out_path.suffix)
    if args.sar_threshold != SAR_VV_THRESHOLD:
        stem = out_path.stem + f'_sar{int(args.sar_threshold)}'
        out_path = out_path.with_name(stem + out_path.suffix)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_datetime':      datetime.now().isoformat(),
        'mode':              args.mode,
        's2_year':           S2_YEAR,
        'sar_threshold_db':  args.sar_threshold,
        'n_per_class':       N_PER_CLASS,
        'seed':              SEED,
        'results':           all_results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nResults saved → {out_path}')


if __name__ == '__main__':
    main()
