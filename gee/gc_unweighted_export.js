
/**
 * gc_unweighted_export.js — UrbanWell Analytics
 * =======================================================================
 * Exports raw, unweighted, unbuffered greenspace coverage per census tract
 * — the construct Wu et al. (2023, Nat. Commun. 14:6460) validate in their
 * Supplementary Fig. 17. The project's primary exposure metric, GreenEx_pct,
 * is a different construct by design: population-weighted and passed
 * through a 500m convolution kernel (Wu et al. Eq. 2). Comparing GreenEx_pct
 * directly to an unweighted reference (e.g. NAIP) compares two different
 * constructs; this script produces the like-for-like coverage-to-coverage
 * comparison instead, matching Wu et al.'s own validation design.
 *
 * Classification: Sentinel-2 surface reflectance, three-endmember linear
 * spectral unmixing (vegetation / impervious / soil), identical band set,
 * cloud mask, and endmember values to the production GreenEx_S2 pipeline,
 * so this differs from the production metric only in the population
 * weighting and 500m smoothing steps it deliberately omits.
 */

var PROJECT = 'urbangreenblue';
var TARGET_YEAR = 2023;
var SCALE = 100;      // matches the project's standard GS_S2 export scale
var TILE_SCALE = 16;  // mitigates reduceRegions memory limits over large counties

var CITIES = {
  ATL: {name: 'Atlanta (Fulton County, GA)',   state: '13', county: '121'},
  MIA: {name: 'Miami (Miami-Dade County, FL)', state: '12', county: '086'},
  LAX: {name: 'Los Angeles (LA County, CA)',   state: '06', county: '037'},
};

// Vegetation / impervious / soil endmembers, band order [B2, B3, B4, B8, B11, B12].
var ENDMEMBERS = [
  [0.05, 0.04, 0.09, 0.40, 0.08, 0.03],  // vegetation
  [0.15, 0.18, 0.22, 0.24, 0.30, 0.21],  // impervious
  [0.10, 0.12, 0.18, 0.22, 0.36, 0.28],  // soil
];
var BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];

function getAOI(stateFips, countyFips) {
  return ee.FeatureCollection('TIGER/2020/TRACT')
    .filter(ee.Filter.and(
      ee.Filter.eq('STATEFP', stateFips),
      ee.Filter.eq('COUNTYFP', countyFips)))
    .geometry().dissolve({maxError: 10});
}

function getS2Composite(aoi, year) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(year + '-01-01', year + '-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(function(img) {
      var qa = img.select('QA60');
      var mask = qa.bitwiseAnd(1 << 10).eq(0).and(qa.bitwiseAnd(1 << 11).eq(0));
      return img.updateMask(mask);
    });
  // S2_SR_HARMONIZED bands are scaled reflectance x 10000.
  return col.median().divide(10000);
}

function classifyGreenspaceCoverage(s2img) {
  var unmixed = s2img.select(BANDS).unmix(ENDMEMBERS, true, true);
  // Output band order matches ENDMEMBERS: 0 = vegetation, 1 = impervious, 2 = soil.
  return unmixed.select([0]).rename('gc_unweighted_frac').clamp(0, 1);
}

function zonalMeanPerTract(gcImage, stateFips, countyFips) {
  var tracts = ee.FeatureCollection('TIGER/2020/TRACT')
    .filter(ee.Filter.and(
      ee.Filter.eq('STATEFP', stateFips),
      ee.Filter.eq('COUNTYFP', countyFips)));
  return gcImage.reduceRegions({
    collection: tracts,
    reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.count(), sharedInputs: true}),
    scale: SCALE,
    tileScale: TILE_SCALE,
  });
}

function runCity(code) {
  var cfg = CITIES[code];
  var aoi = getAOI(cfg.state, cfg.county);
  var s2 = getS2Composite(aoi, TARGET_YEAR);
  var gc = classifyGreenspaceCoverage(s2);
  var tractFc = zonalMeanPerTract(gc, cfg.state, cfg.county);

  var outName = 'greenex_gc_unweighted_' + code;
  Export.table.toDrive({
    collection: tractFc.map(function(f) {
      return f.set({city: cfg.name});
    }),
    description: outName,
    folder: 'UrbanWellExports',
    fileNamePrefix: outName,
    fileFormat: 'CSV',
    selectors: ['GEOID', 'city', 'mean', 'count'],
  });
  print('Export started: ' + outName);
}

Object.keys(CITIES).forEach(runCity);
