// lst_summer_gee_script.js
// ============================================================================
// PURPOSE: export mean summer 2023 land surface temperature (LST) per census
// tract, for an EXPLORATORY correlation check against GreenEx_pct/BlueEx_pct
// -- NOT a new covariate in the FMD model. This tests one small, bounded
// piece of the atmospheric mechanism cited in both papers' introductions
// (Bowler et al. 2010's cooling finding) with real data from your own study
// areas, without reopening the MGWR specification.
//
// Source: Landsat 8/9 Collection 2 Level 2 surface temperature (ST_B10),
// which comes pre-calibrated to Kelvin via the collection's own scale
// factors -- no separate at-sensor brightness temperature conversion needed.
// ============================================================================

var TARGET_YEAR = 2023;
var SUMMER_START = TARGET_YEAR + '-06-01';
var SUMMER_END = TARGET_YEAR + '-08-31';

var CITIES = {
  'ATL': {name: 'Atlanta (Fulton County, GA)', state: '13', county: '121'},
  'MIA': {name: 'Miami (Miami-Dade County, FL)', state: '12', county: '086'},
  'LAX': {name: 'Los Angeles (LA County, CA)', state: '06', county: '037'},
};

function getAOI(stateFips, countyFips) {
  return ee.FeatureCollection('TIGER/2020/TRACT')
    .filter(ee.Filter.and(
      ee.Filter.eq('STATEFP', stateFips),
      ee.Filter.eq('COUNTYFP', countyFips)))
    .geometry().dissolve({maxError: 10});
}

// Landsat Collection 2 Level 2 QA_PIXEL cloud/shadow mask.
// Bit 3 = cloud, Bit 4 = cloud shadow (standard USGS Collection 2 bit layout).
function maskL2Clouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloudMask = qa.bitwiseAnd(1 << 3).eq(0);
  var shadowMask = qa.bitwiseAnd(1 << 4).eq(0);
  return image.updateMask(cloudMask.and(shadowMask));
}

// Collection 2 Level 2 ST_B10 scale factors, per USGS documentation:
// LST_Kelvin = ST_B10_DN * 0.00341802 + 149.0
// Converted to Celsius here since that is the more interpretable unit for a
// correlation table alongside GreenEx_pct/BlueEx_pct (both in percent).
function scaleThermal(image) {
  var lstKelvin = image.select('ST_B10').multiply(0.00341802).add(149.0);
  var lstCelsius = lstKelvin.subtract(273.15).rename('lst_celsius');
  return image.addBands(lstCelsius, null, true);
}

function getSummerLST(aoi) {
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(SUMMER_START, SUMMER_END)
    .map(maskL2Clouds)
    .map(scaleThermal);
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate(SUMMER_START, SUMMER_END)
    .map(maskL2Clouds)
    .map(scaleThermal);
  // Merge Landsat 8 and 9 for denser summer coverage before compositing --
  // both carry the same TIRS sensor design and Collection 2 scale factors,
  // so a straight merge + median is appropriate here.
  var merged = l8.merge(l9);
  return merged.select('lst_celsius').median();
}

function zonalMeanPerTract(lstImage, stateFips, countyFips) {
  var tracts = ee.FeatureCollection('TIGER/2020/TRACT')
    .filter(ee.Filter.and(
      ee.Filter.eq('STATEFP', stateFips),
      ee.Filter.eq('COUNTYFP', countyFips)));
  return lstImage.reduceRegions({
    collection: tracts,
    reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.count(), sharedInputs: true}),
    scale: 30,       // native Landsat thermal resolution in Collection 2
    tileScale: 8,
  });
}

function runCity(code) {
  var cfg = CITIES[code];
  var aoi = getAOI(cfg.state, cfg.county);
  var lst = getSummerLST(aoi);
  var tractFc = zonalMeanPerTract(lst, cfg.state, cfg.county);

  var outName = 'lst_summer_' + code;
  Export.table.toDrive({
    collection: tractFc.map(function(f) {
      return f.set({city: cfg.name, year: TARGET_YEAR});
    }),
    description: outName,
    folder: 'UrbanWellExports',
    fileNamePrefix: outName,
    fileFormat: 'CSV',
    selectors: ['GEOID', 'city', 'year', 'mean', 'count'],
  });
  print('Export started: ' + outName);
}

Object.keys(CITIES).forEach(runCity);

// ============================================================================
// AFTER RUNNING: download the three lst_summer_{CODE}.csv files from
// Google Drive (UrbanWellExports folder) into data/processed/, rename the
// 'mean' column to 'lst_celsius_mean' for clarity, then merge against your
// existing GreenEx_pct/BlueEx_pct panel on GEOID and compute a simple
// Pearson correlation per county. This is a small standalone script, not a
// change to mgwr_analysis.py or the FMD model -- keep it that way.
//
// A LOW 'count' value (few valid thermal pixels after cloud masking) for any
// tract is worth checking before trusting that tract's LST mean -- summer in
// humid Atlanta/Miami can have persistent cloud cover reducing usable
// Landsat 8/9 scenes in the June-August window more than in Los Angeles.
// ============================================================================
