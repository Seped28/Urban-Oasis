/**
 * tcc_reference_export.js — UrbanWell Analytics
 * =======================================================================
 * Exports tract-level zonal-mean Tree Canopy Cover from the USDA Forest
 * Service / USGS NLCD TCC product as an independent validation reference
 * for GreenEx_pct, alongside the NAIP-derived reference. TCC is a
 * federally produced, independently modeled product with its own
 * documented accuracy characterization — distinct from a custom-trained
 * classifier.
 *
 * Dataset: USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5 (USFS, part of the
 * USGS/MRLC NLCD consortium). 30m pixels, annual coverage 1985–2023.
 * Band used: NLCD_Percent_Tree_Canopy_Cover — the post-processed product
 * (non-treed pixels set to 0%), preferred over the raw
 * Science_Percent_Tree_Canopy_Cover band for general use.
 *
 * Scope note: TCC measures tree canopy only. GreenEx_pct (spectral
 * unmixing, vegetation endmember) captures all vegetation — grass, shrub,
 * and tree. This is not a like-for-like construct comparison the way the
 * unweighted-coverage validation is; expect TCC to run systematically
 * lower than GreenEx_pct wherever non-tree vegetation is common, by
 * construction, not as an error.
 */

var TARGET_YEAR = 2023;
var SCALE = 100;
var TILE_SCALE = 16;

var CITIES = {
  ATL: {name: 'Atlanta (Fulton County, GA)',   state: '13', county: '121'},
  MIA: {name: 'Miami (Miami-Dade County, FL)', state: '12', county: '086'},
  LAX: {name: 'Los Angeles (LA County, CA)',   state: '06', county: '037'},
};

function getTracts(stateFips, countyFips) {
  return ee.FeatureCollection('TIGER/2020/TRACT')
    .filter(ee.Filter.and(
      ee.Filter.eq('STATEFP', stateFips),
      ee.Filter.eq('COUNTYFP', countyFips)));
}

function getTccImage(year) {
  var col = ee.ImageCollection('USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5')
    .filter(ee.Filter.calendarRange(year, year, 'year'))
    .filter('study_area == "CONUS"');
  return col.first();
}

function runCity(code) {
  var cfg = CITIES[code];
  var tracts = getTracts(cfg.state, cfg.county);
  var tcc = getTccImage(TARGET_YEAR);

  var tccPct = tcc.select('NLCD_Percent_Tree_Canopy_Cover').rename('tcc_pct');
  var tccFrac = tccPct.divide(100).rename('tcc_frac');
  var combined = tccPct.addBands(tccFrac);

  var zonal = combined.reduceRegions({
    collection: tracts,
    reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.count(), sharedInputs: true}),
    scale: SCALE,
    tileScale: TILE_SCALE,
  });

  // reduceRegions on a two-band image produces per-band-suffixed count
  // properties (tcc_pct_count, tcc_frac_count, identical values); expose
  // one as a plain 'count' column for consistency with the other exports.
  zonal = zonal.map(function(f) {
    return f.set('count', f.get('tcc_pct_count'));
  });

  var outName = 'greenex_tcc_reference_' + code;
  Export.table.toDrive({
    collection: zonal.map(function(f) {
      return f.set({city: cfg.name, tcc_year: TARGET_YEAR, product: 'NLCD_Percent_Tree_Canopy_Cover'});
    }),
    description: outName,
    folder: 'UrbanWellExports',
    fileNamePrefix: outName,
    fileFormat: 'CSV',
    selectors: ['GEOID', 'city', 'tcc_pct_mean', 'tcc_frac_mean', 'count', 'tcc_year', 'product'],
  });
  print('Export task queued: ' + outName);
}

Object.keys(CITIES).forEach(runCity);
