"""
verify_naip_dates.py — UrbanWell Analytics / Paper 2a
=======================================================
Diagnostic companion to kappa_validation.py.

PURPOSE:
  kappa_validation.py's get_naip_water_mask() only checks
  col.size().getInfo() > 0 for a given calendar year filter, then assumes
  the images are genuinely from that year. This script pulls the actual
  system:time_start (and system:index, which encodes state/flight-line
  info) for every NAIP image found in the AOI, so you can confirm whether
  "NAIP 2023" images are really 2023-acquired or leftover/mislabeled.

USAGE:
  python python/verify_naip_dates.py --city ATL
  python python/verify_naip_dates.py --city ATL --year 2023
  python python/verify_naip_dates.py            # all three cities
"""

import argparse
import datetime
import ee

PROJECT = 'urbangreenblue'

CITIES = {
    'ATL': {'name': 'Atlanta (Fulton County, GA)', 'state': '13', 'county': '121'},
    'MIA': {'name': 'Miami (Miami-Dade County, FL)', 'state': '12', 'county': '086'},
    'LAX': {'name': 'Los Angeles (LA County, CA)', 'state': '06', 'county': '037'},
}


def get_aoi(state_fips, county_fips):
    return (ee.FeatureCollection('TIGER/2020/TRACT')
              .filter(ee.Filter.And(
                  ee.Filter.eq('STATEFP', state_fips),
                  ee.Filter.eq('COUNTYFP', county_fips)))
              .geometry().dissolve(maxError=10))


def check_city(code, cfg, target_year):
    aoi = get_aoi(cfg['state'], cfg['county'])
    print(f'\n{"="*70}')
    print(f'{cfg["name"]} — NAIP acquisition-date check, target year {target_year}')
    print(f'{"="*70}')

    for yr in [target_year, target_year - 1, target_year - 2, target_year - 3]:
        col = (ee.ImageCollection('USDA/NAIP/DOQQ')
                 .filterBounds(aoi)
                 .filterDate(f'{yr}-01-01', f'{yr}-12-31'))
        n = col.size().getInfo()
        print(f'\n  Year filter {yr}: {n} images matched')
        if n == 0:
            continue

        # Pull actual acquisition timestamps + index (encodes state/quad/flightline)
        info = col.limit(10).toList(10).getInfo()
        seen_years = set()
        for img in info:
            props = img.get('properties', {})
            ts = props.get('system:time_start')
            idx = img.get('id', props.get('system:index', '?'))
            if ts is not None:
                dt = datetime.datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d')
                actual_year = dt[:4]
                seen_years.add(actual_year)
                flag = '  <-- MISMATCH' if actual_year != str(yr) else ''
                print(f'    {idx}: actual acquisition date = {dt}{flag}')
            else:
                print(f'    {idx}: no system:time_start property found')

        if seen_years - {str(yr)}:
            print(f'  ⚠ Filter says {yr} but real acquisition year(s) found: {sorted(seen_years)}')
        else:
            print(f'  ✓ All sampled images genuinely acquired in {yr}')


def main():
    parser = argparse.ArgumentParser(description='Verify real NAIP acquisition dates vs GEE year filter')
    parser.add_argument('--city', choices=['ATL', 'MIA', 'LAX'], help='Single city (default: all three)')
    parser.add_argument('--year', type=int, default=2023, help='Target S2/NAIP year to check (default: 2023)')
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    print(f'GEE initialised | project={PROJECT}')

    cities = {args.city: CITIES[args.city]} if args.city else CITIES
    for code, cfg in cities.items():
        check_city(code, cfg, args.year)


if __name__ == '__main__':
    main()
