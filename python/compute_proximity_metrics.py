"""
compute_proximity_metrics.py — UrbanWell Analytics
Computes two proximity-based UGBS exposure metrics as complements to the
satellite-derived Wu et al. continuous exposure metric.

WHY TWO METRICS?
  Metric 1 (from GEE pipeline): Wu et al. (2023) continuous population-weighted
  vegetation fraction within 500m neighbourhood. Captures ambient greenspace.
  Answers: "How much greenspace surrounds residents?"

  Metric 2 (this script): Binary/distance proximity to accessible green/blue space.
  Captures active recreation opportunity and psychological access to nature.
  Answers: "Do residents have a park or water body they can reach on foot?"

  Both are independently validated in the health literature (Richardson et al. 2013;
  Gascon et al. 2015; Völker & Kistemann 2011) and relate to mental health through
  different pathways (restorative environment vs active recreation opportunity).

DATA SOURCES:
  GREENSPACE PROXIMITY:
    Urban (city-level): Trust for Public Land (TPL) ParkServe
      https://www.tpl.org/parkserve
      Provides: % population within 10-min walk, park acres per 1,000 residents
      Coverage: 14,000+ US cities, census tract level
      Years: Annual (2022/2023 versions available)
      Citation: Wolch et al. (2014 Landscape Urban Plan); Rigolon (2016)

    Full national: NLCD (National Land Cover Database) — USGS
      https://www.mrlc.gov/data (download NLCD 2016, 2019, 2021)
      Classes used: 41, 42, 43 (forest), 71 (herbaceous), 52 (shrub)
      EXCLUDE: 81 (pasture), 82 (crops) — not accessible green space
      Citation: Homer et al. (2015 Remote Sensing Environ)

  BLUESPACE PROXIMITY:
    NWI (National Wetlands Inventory) — USFWS
      https://www.fws.gov/program/national-wetlands-inventory
      Provides: All mapped US wetlands, rivers, lakes, ponds, estuaries, coasts
      Format: State-level shapefiles
      Citation: Cowardin et al. (1979); Wheeler et al. (2015 Health Place)

OUTPUT COLUMNS (added to panel at GEOID+year level):
  GS_park_access    : From TPL — % population within 10-min walk of park (0-100)
                      NULL for tracts not in TPL coverage (rural areas)
  GS_park_acres_1k  : From TPL — park acres per 1,000 residents
                      NULL for tracts not in TPL coverage
  GS_proximity_m    : Distance in metres from tract centroid to nearest NLCD
                      green pixel (classes 41-43, 71, 52)
  GS_within_500m    : Binary — TRUE if GS_proximity_m <= 500
  BS_proximity_m    : Distance in metres from tract centroid to nearest NWI polygon
  BS_within_500m    : Binary — TRUE if BS_proximity_m <= 500

TIME TREATMENT:
  TPL variables    : Time-invariant (use latest available year for all panel years)
  NLCD variables   : Use nearest available NLCD year (2016→2015-2017, 2019→2018-2020,
                     2021→2021-2023). Slowly-changing.
  NWI variables    : Time-invariant (use as baseline characteristic)

SETUP:
  pip install geopandas shapely pandas pyarrow --break-system-packages
  pip install scipy --break-system-packages   # for KD-tree distance calc
  
  Download required:
  1. TPL ParkServe tract-level CSV from https://www.tpl.org/parkserve/downloads
  2. NLCD 2016, 2019, 2021 GeoTIFF from https://www.mrlc.gov/data
  3. NWI state shapefiles from https://www.fws.gov/program/national-wetlands-inventory
  4. TIGER 2020 tract centroids (or compute from tract shapefile)

USAGE:
  python python/compute_proximity_metrics.py \
    --tracts    data/raw/tiger/tl_2020_us_tract.shp \
    --tpl       data/raw/tpl/ParkServe_Tracts_2022.csv \
    --nlcd-dir  data/raw/nlcd/ \
    --nwi-dir   data/raw/nwi/ \
    --output    data/processed/proximity_metrics.csv
"""

import argparse
import pathlib
import logging
import sys
import warnings
from typing import Optional

import pandas as pd
import numpy as np

# Suppress geopandas/shapely version warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import nearest_points
    from scipy.spatial import cKDTree
except ImportError as e:
    print(f"ERROR: Missing dependency — {e}")
    print("Run: pip install geopandas shapely scipy --break-system-packages")
    sys.exit(1)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── NLCD year mapping ─────────────────────────────────────────────────────────
# Map each panel year to the nearest available NLCD product year
NLCD_YEAR_MAP = {
    2015: 2016, 2016: 2016, 2017: 2016,
    2018: 2019, 2019: 2019, 2020: 2019,
    2021: 2021, 2022: 2021, 2023: 2021,
}

# NLCD land cover classes considered accessible greenspace
GS_NLCD_CLASSES = {41, 42, 43, 52, 71}
# Excludes: 81 (pasture/hay), 82 (cultivated crops) — not publicly accessible
# Excludes: 90, 95 (wetlands) — handled by NWI for BS metric

# ── STEP 1: Load tract centroids ──────────────────────────────────────────────

def load_tract_centroids(tracts_path: pathlib.Path) -> gpd.GeoDataFrame:
    """Load TIGER tract shapefile and compute centroids in EPSG:5070 (Albers Equal Area)."""
    log.info(f'Loading tract shapefile: {tracts_path}')
    tracts = gpd.read_file(tracts_path)

    # Ensure GEOID is 11-digit zero-padded string
    tracts['GEOID'] = tracts['GEOID'].astype(str).str.zfill(11)

    # Project to EPSG:5070 for accurate distance calculation (metres, CONUS)
    tracts = tracts.to_crs('EPSG:5070')
    tracts['centroid'] = tracts.geometry.centroid

    centroids = gpd.GeoDataFrame(
        tracts[['GEOID']],
        geometry=tracts['centroid'],
        crs='EPSG:5070'
    )
    log.info(f'  {len(centroids):,} tract centroids loaded')
    return centroids


# ── STEP 2: TPL ParkServe ─────────────────────────────────────────────────────

def load_tpl(tpl_path: pathlib.Path, centroids: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Load TPL ParkServe tract-level data.
    
    TPL CSV columns include (varies by version — check actual file):
      tract_fips / GEOID, park_access (% within 10-min walk),
      park_acres_per_1000_residents, etc.
    
    Returns DataFrame with GEOID, GS_park_access, GS_park_acres_1k.
    """
    log.info(f'Loading TPL ParkServe: {tpl_path}')

    tpl = pd.read_csv(tpl_path, dtype=str, low_memory=False)

    # Normalise GEOID column — TPL uses different column names across versions
    geoid_candidates = [c for c in tpl.columns
                        if c.lower() in ('geoid', 'tract_fips', 'tractfips',
                                         'census_tract', 'tract_id', 'fips')]
    if not geoid_candidates:
        log.error(f'Cannot find GEOID column in TPL file. Columns: {list(tpl.columns)}')
        log.error('Check TPL file column names and update geoid_candidates list.')
        raise ValueError('No GEOID column found in TPL file')
    geoid_col = geoid_candidates[0]
    tpl['GEOID'] = tpl[geoid_col].astype(str).str.zfill(11)

    # Find park access and acreage columns — TPL changes column names between versions
    access_candidates = [c for c in tpl.columns
                         if 'access' in c.lower() or 'walk' in c.lower() or 'pct' in c.lower()]
    acres_candidates  = [c for c in tpl.columns
                         if 'acres' in c.lower() or 'acreage' in c.lower()]

    log.info(f'  TPL access column candidates: {access_candidates}')
    log.info(f'  TPL acres column candidates:  {acres_candidates}')
    log.info(f'  Using first match — verify these are correct before proceeding')

    result = tpl[['GEOID']].copy()
    if access_candidates:
        result['GS_park_access'] = pd.to_numeric(
            tpl[access_candidates[0]], errors='coerce')
    else:
        log.warning('No park access column found in TPL — GS_park_access will be NULL')
        result['GS_park_access'] = np.nan

    if acres_candidates:
        result['GS_park_acres_1k'] = pd.to_numeric(
            tpl[acres_candidates[0]], errors='coerce')
    else:
        log.warning('No park acres column found in TPL — GS_park_acres_1k will be NULL')
        result['GS_park_acres_1k'] = np.nan

    # TPL covers urban areas only — rural tracts will be NULL (correct)
    n_covered = result['GS_park_access'].notna().sum()
    n_total   = len(centroids)
    log.info(f'  TPL coverage: {n_covered:,} tracts ({100*n_covered/n_total:.1f}% of {n_total:,})')

    return result[['GEOID', 'GS_park_access', 'GS_park_acres_1k']]


# ── STEP 3: NLCD proximity ────────────────────────────────────────────────────

def compute_nlcd_proximity(centroids: gpd.GeoDataFrame,
                           nlcd_dir: pathlib.Path) -> pd.DataFrame:
    """
    For each tract centroid, compute distance to nearest NLCD greenspace pixel
    for each relevant NLCD year (2016, 2019, 2021).
    
    Uses rasterio to read NLCD, extracts greenspace pixel centroids, builds
    KD-tree for fast nearest-neighbour lookup.
    
    Returns DataFrame with GEOID, GS_proximity_m_{nlcd_year} columns.
    """
    try:
        import rasterio
        from rasterio.transform import xy as rio_xy
    except ImportError:
        log.error('rasterio not installed. Run: pip install rasterio --break-system-packages')
        raise

    nlcd_years = [2016, 2019, 2021]
    results = {c['GEOID']: {} for _, c in centroids.iterrows()}

    # Pre-extract centroid coordinates as numpy arrays (EPSG:5070)
    cent_x = np.array([g.x for g in centroids.geometry])
    cent_y = np.array([g.y for g in centroids.geometry])

    for nlcd_year in nlcd_years:
        # Look for NLCD file — common naming conventions
        candidates = list(nlcd_dir.glob(f'*{nlcd_year}*.img')) + \
                     list(nlcd_dir.glob(f'*{nlcd_year}*.tif')) + \
                     list(nlcd_dir.glob(f'NLCD_{nlcd_year}*.img')) + \
                     list(nlcd_dir.glob(f'nlcd_{nlcd_year}*.tif'))

        if not candidates:
            log.warning(f'NLCD {nlcd_year} file not found in {nlcd_dir} — skipping')
            for geoid in results:
                results[geoid][f'GS_proximity_m_{nlcd_year}'] = np.nan
            continue

        nlcd_file = candidates[0]
        log.info(f'Processing NLCD {nlcd_year}: {nlcd_file.name}')

        with rasterio.open(nlcd_file) as src:
            # Read full raster (NLCD is manageable at ~500MB)
            data   = src.read(1)
            transform = src.transform
            raster_crs = src.crs

            # Reproject centroids if NLCD uses different CRS
            if raster_crs.to_epsg() != 5070:
                log.info(f'  Reprojecting centroids from EPSG:5070 to {raster_crs}')
                import pyproj
                transformer = pyproj.Transformer.from_crs(
                    'EPSG:5070', raster_crs.to_epsg(), always_xy=True)
                proj_x, proj_y = transformer.transform(cent_x, cent_y)
            else:
                proj_x, proj_y = cent_x, cent_y

            # Find all greenspace pixels
            gs_mask = np.isin(data, list(GS_NLCD_CLASSES))
            rows, cols = np.where(gs_mask)
            if len(rows) == 0:
                log.error(f'No greenspace pixels found in NLCD {nlcd_year}')
                for geoid in results:
                    results[geoid][f'GS_proximity_m_{nlcd_year}'] = np.nan
                continue

            # Get pixel centre coordinates
            px, py = rasterio.transform.xy(transform, rows, cols)
            gs_coords = np.column_stack([px, py])

            log.info(f'  {len(rows):,} greenspace pixels → building KD-tree...')
            tree = cKDTree(gs_coords)

            # Query nearest greenspace pixel for each centroid
            centroid_coords = np.column_stack([proj_x, proj_y])
            distances, _ = tree.query(centroid_coords, k=1, workers=-1)

            col_name = f'GS_proximity_m_{nlcd_year}'
            for i, geoid in enumerate(centroids['GEOID']):
                results[geoid][col_name] = float(distances[i])

            log.info(f'  Done. Median distance: {np.median(distances):.0f}m, '
                     f'% within 500m: {100*(distances<=500).mean():.1f}%')

    # Build result DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'GEOID'
    df = df.reset_index()

    # Map panel years to NLCD proximity columns
    # Panel years 2015-2017 → NLCD 2016, 2018-2020 → NLCD 2019, 2021-2023 → NLCD 2021
    rows_list = []
    for panel_year in range(2015, 2024):
        nlcd_year = NLCD_YEAR_MAP[panel_year]
        col = f'GS_proximity_m_{nlcd_year}'
        if col in df.columns:
            year_df = df[['GEOID', col]].copy()
            year_df.columns = ['GEOID', 'GS_proximity_m']
            year_df['year'] = panel_year
            year_df['GS_within_500m'] = year_df['GS_proximity_m'] <= 500
            rows_list.append(year_df)

    if rows_list:
        return pd.concat(rows_list, ignore_index=True)
    else:
        log.error('No NLCD proximity data computed')
        return pd.DataFrame(columns=['GEOID', 'year', 'GS_proximity_m', 'GS_within_500m'])


# ── STEP 4: NWI proximity ─────────────────────────────────────────────────────

def compute_nwi_proximity(centroids: gpd.GeoDataFrame,
                          nwi_dir: pathlib.Path) -> pd.DataFrame:
    """
    For each tract centroid, compute distance to nearest NWI water body polygon.
    NWI is treated as time-invariant — one distance value per tract, applied
    to all panel years.
    
    NWI shapefiles are state-level. This function loads all state files,
    unions them, builds a spatial index, and computes nearest-polygon distance.
    
    Returns DataFrame with GEOID, BS_proximity_m, BS_within_500m (time-invariant,
    same value broadcast to all panel years).
    """
    nwi_files = sorted(nwi_dir.glob('**/*_wetlands.shp')) + \
                sorted(nwi_dir.glob('**/*.shp'))

    if not nwi_files:
        log.error(f'No NWI shapefiles found in {nwi_dir}')
        log.error('Download from: https://www.fws.gov/program/national-wetlands-inventory')
        log.error('Expected pattern: *_wetlands.shp or state shapefiles in subdirectories')
        raise FileNotFoundError(f'No NWI shapefiles in {nwi_dir}')

    log.info(f'Loading {len(nwi_files)} NWI state files...')

    # Load and combine all state NWI files
    chunks = []
    for f in nwi_files:
        try:
            gdf = gpd.read_file(f, columns=['geometry'])  # geometry only — NWI files are large
            chunks.append(gdf[['geometry']])
        except Exception as e:
            log.warning(f'  Could not read {f.name}: {e}')

    if not chunks:
        raise ValueError('No NWI files could be read')

    nwi = pd.concat(chunks, ignore_index=True)
    nwi = gpd.GeoDataFrame(nwi, geometry='geometry')

    # Set CRS — NWI uses geographic (EPSG:4326), reproject to EPSG:5070
    if nwi.crs is None:
        nwi = nwi.set_crs('EPSG:4326')
    nwi = nwi.to_crs('EPSG:5070')

    log.info(f'  {len(nwi):,} NWI polygons loaded. Computing distances...')

    # For large datasets: use polygon exterior boundary points for KD-tree
    # (faster than shapely nearest_points on millions of polygons)
    # Densify boundaries then extract point coordinates
    boundary_coords = []
    log.info('  Extracting boundary coordinates (may take a few minutes)...')
    for geom in nwi.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            # Get boundary points — use exterior for polygons
            if hasattr(geom, 'exterior'):
                coords = np.array(geom.exterior.coords)
            elif hasattr(geom, 'geoms'):
                for g in geom.geoms:
                    if hasattr(g, 'exterior'):
                        coords = np.array(g.exterior.coords)
                        boundary_coords.append(coords)
                continue
            else:
                coords = np.array(geom.coords)
            boundary_coords.append(coords)
        except Exception:
            continue

    if not boundary_coords:
        raise ValueError('Could not extract NWI boundary coordinates')

    all_coords = np.vstack(boundary_coords)
    log.info(f'  {len(all_coords):,} boundary points → building KD-tree...')
    tree = cKDTree(all_coords)

    # Query
    cent_x = np.array([g.x for g in centroids.geometry])
    cent_y = np.array([g.y for g in centroids.geometry])
    centroid_coords = np.column_stack([cent_x, cent_y])

    distances, _ = tree.query(centroid_coords, k=1, workers=-1)

    log.info(f'  Done. Median distance: {np.median(distances):.0f}m, '
             f'% within 500m: {100*(distances<=500).mean():.1f}%')

    # Build result — time-invariant, broadcast to all panel years
    rows_list = []
    for panel_year in range(2015, 2024):
        year_df = pd.DataFrame({
            'GEOID': centroids['GEOID'].values,
            'year': panel_year,
            'BS_proximity_m': distances,
            'BS_within_500m': distances <= 500
        })
        rows_list.append(year_df)

    return pd.concat(rows_list, ignore_index=True)


# ── STEP 5: Combine and export ────────────────────────────────────────────────

def build_proximity_panel(centroids, tpl_df, nlcd_df, nwi_df,
                          output_path: pathlib.Path):
    """Join all proximity metrics into a panel with GEOID + year as keys."""
    log.info('Building proximity panel...')

    # Start with full GEOID × year skeleton
    all_geoids = centroids['GEOID'].tolist()
    all_years  = list(range(2015, 2024))
    skeleton   = pd.DataFrame([
        {'GEOID': g, 'year': y}
        for g in all_geoids for y in all_years
    ])
    log.info(f'  Panel skeleton: {len(skeleton):,} rows ({len(all_geoids):,} tracts × {len(all_years)} years)')

    # Join TPL (time-invariant — same value for all years)
    if tpl_df is not None:
        panel = skeleton.merge(tpl_df, on='GEOID', how='left')
    else:
        panel = skeleton.copy()
        panel['GS_park_access']   = np.nan
        panel['GS_park_acres_1k'] = np.nan

    # Join NLCD (year-aware — different value per panel year)
    if nlcd_df is not None:
        panel = panel.merge(nlcd_df, on=['GEOID', 'year'], how='left')
    else:
        panel['GS_proximity_m'] = np.nan
        panel['GS_within_500m'] = np.nan

    # Join NWI (time-invariant — same value for all years)
    if nwi_df is not None:
        panel = panel.merge(nwi_df, on=['GEOID', 'year'], how='left')
    else:
        panel['BS_proximity_m'] = np.nan
        panel['BS_within_500m'] = np.nan

    log.info(f'  Final panel: {len(panel):,} rows')
    log.info(f'  Null rates:')
    for col in ['GS_park_access', 'GS_park_acres_1k',
                'GS_proximity_m', 'GS_within_500m',
                'BS_proximity_m', 'BS_within_500m']:
        if col in panel.columns:
            null_pct = 100 * panel[col].isna().mean()
            log.info(f'    {col:<22}: {null_pct:.1f}% null')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    log.info(f'Saved: {output_path}')

    return panel


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='UrbanWell proximity metric pipeline')
    parser.add_argument('--tracts',   required=True,
                        help='Path to TIGER 2020 tract shapefile')
    parser.add_argument('--tpl',      default=None,
                        help='Path to TPL ParkServe CSV (optional)')
    parser.add_argument('--nlcd-dir', default=None,
                        help='Directory containing NLCD GeoTIFF/IMG files')
    parser.add_argument('--nwi-dir',  default=None,
                        help='Directory containing NWI state shapefiles')
    parser.add_argument('--output',
                        default='data/processed/proximity_metrics.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    # Load tracts
    centroids = load_tract_centroids(pathlib.Path(args.tracts))

    # TPL
    tpl_df = None
    if args.tpl:
        try:
            tpl_df = load_tpl(pathlib.Path(args.tpl), centroids)
        except Exception as e:
            log.error(f'TPL load failed: {e}')
            log.warning('Continuing without TPL data — GS_park_* columns will be NULL')

    # NLCD
    nlcd_df = None
    if args.nlcd_dir:
        try:
            nlcd_df = compute_nlcd_proximity(
                centroids, pathlib.Path(args.nlcd_dir))
        except Exception as e:
            log.error(f'NLCD proximity failed: {e}')
            log.warning('Continuing without NLCD data — GS_proximity_* columns will be NULL')

    # NWI
    nwi_df = None
    if args.nwi_dir:
        try:
            nwi_df = compute_nwi_proximity(
                centroids, pathlib.Path(args.nwi_dir))
        except Exception as e:
            log.error(f'NWI proximity failed: {e}')
            log.warning('Continuing without NWI data — BS_proximity_* columns will be NULL')

    if tpl_df is None and nlcd_df is None and nwi_df is None:
        log.error('No data sources provided. Specify at least one of --tpl, --nlcd-dir, --nwi-dir')
        return

    # Build and export panel
    build_proximity_panel(
        centroids, tpl_df, nlcd_df, nwi_df,
        pathlib.Path(args.output)
    )

    log.info('')
    log.info('Next step: join proximity_metrics.csv to main panel in 00_data_prep_v2.R')
    log.info('  panel <- panel |> left_join(proximity, by = c("GEOID", "year"))')
    log.info('')
    log.info('Data download instructions:')
    log.info('  TPL:  https://www.tpl.org/parkserve/downloads')
    log.info('  NLCD: https://www.mrlc.gov/data (2016, 2019, 2021 products)')
    log.info('  NWI:  https://www.fws.gov/program/national-wetlands-inventory/data')


if __name__ == '__main__':
    main()
