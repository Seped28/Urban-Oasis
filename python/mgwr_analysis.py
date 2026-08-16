"""
mgwr_analysis.py — UrbanWell Analytics

Fits OLS, GWR, and Multiscale GWR (MGWR) per city for the FMD ~ GreenEx_pct
+ BlueEx_pct + covariates specification, comparing model fit via AICc.

Implementation notes:
- Continuous covariates are z-score standardized before GWR/MGWR only (OLS
  keeps original units). This matters because a scale mismatch between
  covariates (e.g. income in $-thousands vs. exposure in 0-100 percent)
  produces a high condition number that global OLS tolerates at typical
  sample sizes but that GWR's local regressions, at small candidate
  bandwidths explored during the golden-section AICc search, do not --
  the local design matrix can lose rank entirely, raising a LinAlgError.
  Standardizing avoids this and also makes GreenEx_pct/BlueEx_pct
  bandwidths directly comparable in MGWR, since both covariates are then
  on the same scale.
- A minimum-bandwidth floor is passed to Sel_BW.search() so the
  golden-section search doesn't explore pathologically small candidate
  bandwidths.
- A retry wrapper handles LinAlgError by retrying once with a doubled
  bw_min floor before failing with a clear diagnostic message.
- MGWR non-convergence for a given city/exposure combination (e.g. very
  sparse, spatially clustered exposure) is reported as a named limitation
  in the output rather than silently dropped -- the OLS/GWR results for
  that city are still reported in full.
"""

import argparse
import json
import pathlib
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pyproj import Transformer

try:
    from mgwr.gwr import GWR, MGWR
    from mgwr.sel_bw import Sel_BW
except ImportError:
    sys.exit(
        "ERROR: mgwr package not found.\n"
        "Install: pip install mgwr --break-system-packages\n"
        "Citation: Oshan et al. (2019) ISPRS Int. J. Geo-Inf. 8(6):269"
    )

TARGET_YEAR = 2023
PROC_DIR    = pathlib.Path('data/processed')

CITY_PREFIXES = {
    'ATL': ('13121', 'Atlanta (Fulton County, GA)'),
    'MIA': ('12086', 'Miami (Miami-Dade County, FL)'),
    'LAX': ('06037', 'Los Angeles (LA County, CA)'),
}

OUTCOME   = 'FMD'
COVARS    = ['GreenEx_pct', 'BlueEx_pct', 'Smoke', 'Drink', 'MedIn_k', 'EduAt']

# v2: numerical-stability floor for the adaptive bandwidth search.
# With 7 parameters (intercept + 6 covariates), a candidate bandwidth needs
# comfortably more than 7 local observations to avoid rank deficiency.
# 40 gives roughly 5-6x headroom; doubled to 80 on retry if still singular.
MIN_BANDWIDTH = 40

_transformer = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)


def load_and_filter(panel_path: pathlib.Path) -> pd.DataFrame:
    print(f'\nLoading panel: {panel_path}')
    df = pd.read_csv(panel_path, dtype={'GEOID': str})
    df['GEOID'] = df['GEOID'].str.zfill(11)
    print(f'  Full panel: {len(df):,} rows, years {df["year"].min()}–{df["year"].max()}')

    df = df[df['year'] == TARGET_YEAR].copy()
    print(f'  Year {TARGET_YEAR}: {len(df):,} rows')

    prefix_to_name = {p: name for p, name in CITY_PREFIXES.values()}
    prefixes = list(prefix_to_name.keys())
    mask = df['GEOID'].str[:5].isin(prefixes)
    df = df[mask].copy()
    df['city'] = df['GEOID'].str[:5].map(prefix_to_name)
    print(f'  3-county filter ({", ".join(prefixes)}): {len(df):,} rows')
    for (code, (prefix, cname)) in CITY_PREFIXES.items():
        n = (df['GEOID'].str[:5] == prefix).sum()
        print(f'    {cname}: {n:,} tracts')

    needed = [OUTCOME] + COVARS + ['lon', 'lat']
    n_before = len(df)
    df = df.dropna(subset=needed).copy()
    print(f'  Complete cases: {len(df):,} / {n_before:,} '
          f'({100*len(df)/n_before:.1f}%)')

    if len(df) < 100:
        raise RuntimeError(
            f'Only {len(df)} complete-case observations. Cannot run MGWR reliably.')

    return df


def get_coords_metres(df: pd.DataFrame) -> np.ndarray:
    x_m, y_m = _transformer.transform(df['lon'].values, df['lat'].values)
    return np.column_stack([x_m, y_m])


def standardize_X(df: pd.DataFrame, covars: list) -> tuple:
    """z-score continuous covariates -- GWR/MGWR only, not OLS.
    Returns (X_std, means, stds) so coefficients can be converted back to
    original units later if needed (divide coefficient by std)."""
    X_raw = df[covars].values.astype(float)
    means = X_raw.mean(axis=0)
    stds  = X_raw.std(axis=0)
    stds[stds == 0] = 1.0  # guard against a zero-variance column
    X_std = (X_raw - means) / stds
    return X_std, means, stds


# ── OLS BASELINE (unchanged from v1 — original units, already validated) ────

def run_ols(df: pd.DataFrame, covars: list = None) -> dict:
    print(f'\n{"─"*60}')
    print('OLS BASELINE (original units — unchanged from v1)')
    print(f'{"─"*60}')
    covars = covars or COVARS
    print(f'Formula: {OUTCOME} ~ {" + ".join(covars)}')

    y = df[OUTCOME].values
    X = df[covars].values
    X_sm = sm.add_constant(X)
    col_names = ['const'] + covars

    model = sm.OLS(y, X_sm).fit()
    print(model.summary(xname=col_names))

    cond_no = float(model.condition_number)
    print(f'\nCondition number: {cond_no:.1f}', end='  ')
    if cond_no > 100:
        print('→ SEVERE (Belsley, Kuh & Welsch 1980) — GWR/MGWR use standardized')
        print('  covariates below specifically because of this.')
    elif cond_no > 30:
        print('→ Moderate concern (Belsley, Kuh & Welsch 1980).')
    else:
        print('→ Acceptable.')

    print('\nVIF (manual, > 5 warrants investigation):')
    for i, name in enumerate(covars):
        x_i    = X[:, i]
        x_rest = np.delete(X, i, axis=1)
        x_rest_c = sm.add_constant(x_rest)
        r2 = sm.OLS(x_i, x_rest_c).fit().rsquared
        vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        flag = '  ← CHECK' if vif > 5 else ''
        print(f'  {name:<20} {vif:6.2f}{flag}')

    result = {
        'model': 'OLS', 'n': int(model.nobs), 'covars_used': covars,
        'r2': round(float(model.rsquared), 4),
        'r2_adj': round(float(model.rsquared_adj), 4),
        'aic': round(float(model.aic), 2), 'bic': round(float(model.bic), 2),
        'condition_number': round(cond_no, 1),
        'f_stat': round(float(model.fvalue), 2), 'f_pval': float(model.f_pvalue),
        'coefficients': {
            name: {
                'coef': round(float(model.params[i]), 6),
                'se': round(float(model.bse[i]), 6),
                'tstat': round(float(model.tvalues[i]), 4),
                'pval': round(float(model.pvalues[i]), 6),
            } for i, name in enumerate(col_names)
        },
    }
    return result, model.resid


# ── GWR (v2: standardized X, bw_min floor, retry on singular matrix) ────────

def run_gwr(df: pd.DataFrame, coords: np.ndarray, covars: list = None) -> dict:
    covars = covars or COVARS
    print(f'\n{"─"*60}')
    print('GWR (single adaptive bandwidth, bisquare kernel, AICc selection)')
    print('v2: covariates standardized (z-score) for numerical stability —')
    print('    see module docstring for why (Cond. No. in your OLS output).')
    print(f'{"─"*60}')

    y = df[OUTCOME].values.reshape(-1, 1)
    X_std, means, stds = standardize_X(df, covars)

    bw_min = MIN_BANDWIDTH
    max_attempts = 3
    bw = None
    gwr_model = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(f'  Attempt {attempt}: bw_min={bw_min}. Selecting bandwidth...')
            selector = Sel_BW(coords, y, X_std, kernel='bisquare', fixed=False)
            bw = selector.search(criterion='AICc', bw_min=bw_min)
            print(f'  Selected bandwidth: {int(bw)} nearest neighbours')

            print('  Fitting GWR...')
            gwr_model = GWR(coords, y, X_std, bw,
                            kernel='bisquare', fixed=False, constant=True).fit()
            break
        except np.linalg.LinAlgError as e:
            print(f'  LinAlgError at bw_min={bw_min}: {e}')
            if attempt < max_attempts:
                bw_min *= 2
                print(f'  Retrying with bw_min={bw_min}...')
            else:
                raise RuntimeError(
                    f'GWR still hit a singular matrix after {max_attempts} attempts '
                    f'(final bw_min={bw_min}). This points to something beyond scale '
                    f'-- check for a near-constant covariate within one of the three '
                    f'cities specifically (e.g. Smoke or Drink with almost no local '
                    f'variance in a sub-area), not just the global condition number.'
                ) from e

    aicc = float(gwr_model.aicc)
    r2   = float(gwr_model.R2)
    print(f'  R²: {r2:.4f} | AICc: {aicc:.2f}')
    col_names_gwr = ['intercept'] + covars
    print(f'  Local coefficient ranges (standardized units — min / median / max):')
    for i, name in enumerate(col_names_gwr):
        vals = gwr_model.params[:, i]
        print(f'    {name:<20}  min={vals.min():.4f}  '
              f'med={np.median(vals):.4f}  max={vals.max():.4f}')

    gwr_out = df[['GEOID', 'city', 'lon', 'lat']].copy()
    for i, name in enumerate(col_names_gwr):
        gwr_out[f'gwr_{name}'] = gwr_model.params[:, i]
    gwr_out['gwr_local_r2'] = gwr_model.localR2

    result = {
        'model': 'GWR', 'n': len(df), 'bandwidth': int(bw), 'covars_used': covars,
        'bandwidth_type': 'adaptive (k-nearest-neighbours)',
        'bw_min_used': bw_min, 'kernel': 'bisquare', 'criterion': 'AICc',
        'covariates_standardized': True,
        'r2': round(r2, 4), 'aicc': round(aicc, 2),
        'coef_medians_standardized': {
            name: round(float(np.median(gwr_model.params[:, i])), 6)
            for i, name in enumerate(col_names_gwr)
        },
    }
    return result, gwr_out


# ── MGWR (v2: same standardization + bw_min floor + retry) ──────────────────

def compute_pooled_stats(df_all: pd.DataFrame, covars: list) -> dict:
    """SD of each covariate across ALL cities combined -- the reference
    point for deciding whether a city's LOCAL variance is meaningfully
    smaller than the study's own overall variation. Computed once from the
    full 3-city dataset before splitting."""
    return {cov: {'mean': df_all[cov].mean(), 'sd': df_all[cov].std()} for cov in covars}


def check_within_city_variance(df_city: pd.DataFrame, covars: list, pooled_stats: dict,
                                city_name: str, rel_threshold: float = 0.10):
    """Checks whether a city's local variation in a covariate is small
    relative to the pooled (all-city) SD for that covariate -- a
    coefficient-of-variation (sd/mean) check breaks down for near-zero-mean
    variables (e.g. a city with very little bluespace), so this uses a
    threshold relative to the pooled SD instead, which doesn't have that
    failure mode. Returns the list of covariate names to drop for this
    city."""
    print(f'\n{"─"*60}')
    print(f'DIAGNOSTIC: within-city variance vs pooled SD — {city_name}')
    print(f'{"─"*60}')
    to_drop = []
    for cov in covars:
        vals = df_city[cov].dropna()
        pooled_sd = pooled_stats[cov]['sd']
        local_sd = vals.std()
        rel = local_sd / pooled_sd if pooled_sd > 0 else float('nan')
        flag = rel < rel_threshold
        marker = '  <-- DROP (near-constant locally vs. pooled variation)' if flag else ''
        print(f'  {cov:<15} local_sd={local_sd:10.4f}  pooled_sd={pooled_sd:10.4f}  '
              f'ratio={rel:6.3f}{marker}')
        if flag:
            to_drop.append(cov)
    if to_drop:
        print(f'\n{len(to_drop)} covariate(s) flagged for {city_name}: {to_drop}')
        print('These have local SD < 10% of the pooled (3-city) SD -- essentially no')
        print('independent local signal here, which is exactly what produces the')
        print('unstable/singular local regressions regardless of any bandwidth floor.')
        print('DROPPING them from this city\'s model, not just flagging them, and')
        print('reporting the site-specific specification explicitly in the output.')
    else:
        print(f'\nNo covariate flagged for {city_name} — full model retained.')
    print(f'{"─"*60}\n')
    return to_drop


def run_mgwr(df: pd.DataFrame, coords: np.ndarray, covars: list = None,
             gwr_bw: int = None, use_fixed_kernel: bool = False) -> dict:
    covars = covars or COVARS
    print(f'\n{"─"*60}')
    kernel_label = 'FIXED (distance-based)' if use_fixed_kernel else 'adaptive'
    print(f'MGWR (per-covariate {kernel_label} bandwidths, bisquare kernel, AICc)')
    print('v2: covariates standardized — also makes GreenEx_pct vs BlueEx_pct')
    print('    bandwidth comparison scale-fair, not just numerically stable.')
    print(f'{"─"*60}')

    y = df[OUTCOME].values.reshape(-1, 1)
    X_std, means, stds = standardize_X(df, covars)
    n_obs = len(df)
    n_terms = X_std.shape[1] + 1  # +1 for intercept -- MGWR fits a separate
    # bandwidth per TERM including the intercept when constant=True.

    # v11 FIX: LAX still failed even at bw_min approaching n_obs (2456 on
    # n=2453 -- a bandwidth can't exceed the sample size, so that retry was
    # guaranteed to fail before it started). Combined with GWR's own
    # BlueEx_pct coefficient range (-768.6 to +0.04) and the fact that
    # BlueEx_pct is heavily zero-inflated (pooled mean=0.000, most tracts
    # exactly 0, nonzero only near coast/river), the failure mode is
    # spatial clustering, not sample-size scarcity: a k-NN window of ANY
    # size can land entirely inside an inland/mountain area with zero
    # local variance in that one column, which is a perfect-collinearity
    # singularity no larger k fixes. Per user decision: for cities flagged
    # use_fixed_kernel=True, switch to a FIXED (distance) kernel, since a
    # fixed-radius window's composition doesn't depend on tract density
    # the way k-NN does -- addresses the actual mechanism instead of
    # continuing to raise a k-NN floor that was never going to work.
    # References: Fotheringham, Brunsdon & Charlton (2002) note fixed and
    # adaptive kernels suit different spatial configurations; Wheeler &
    # Tiefelsdorf (2005, J. Geogr. Syst. 7(2):161-187) document that local
    # GWR/MGWR coefficients can become uninterpretable under local
    # collinearity; Wheeler (2007, Environment and Planning A 39(10):2464-
    # 2481) proposes dropping or regularizing the offending predictor as
    # the remedial step when diagnostics confirm local collinearity is
    # severe and structural rather than a sample-size artifact -- which is
    # exactly the fallback used one level up in run_city_analysis if the
    # fixed kernel alone doesn't resolve this.
    if use_fixed_kernel:
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        diag_m = float((x_range ** 2 + y_range ** 2) ** 0.5)
        # Heuristic starting floor only (not from a specific published
        # rule): 2% of the study area's bounding-box diagonal, in metres
        # (coords are already projected to EPSG:5070 by get_coords_metres).
        # Doubled on retry exactly like the adaptive bw_min below.
        bw_min = diag_m * 0.02
        print(f'  Fixed-kernel bw_min floor: 2% of bounding-box diagonal '
              f'({diag_m:,.0f} m) = {bw_min:,.0f} m')
    elif gwr_bw is not None:
        # v9 FIX: LAX (n=2453) kept hitting a singular matrix in MGWR's
        # per-term backfitting even at bw_min=160, despite NO covariate
        # being flagged by the pooled-SD check (all ratios 0.85-0.97).
        # GWR itself needed bw=1229 (~half of LAX's tracts) to stay
        # stable -- use that already-validated bandwidth as an informed
        # starting floor instead of guessing bigger fixed constants.
        bw_min = max(MIN_BANDWIDTH, gwr_bw // 2)
        print(f'  Using GWR-informed bw_min floor: max({MIN_BANDWIDTH}, {gwr_bw}//2) = {bw_min}')
    else:
        bw_min = MIN_BANDWIDTH
    max_attempts = 3
    mgwr_model = None

    for attempt in range(1, max_attempts + 1):
        try:
            # v11 FIX: cap bw_min so a retry can never exceed the sample
            # size for adaptive kernels (a k-NN bandwidth > n is
            # meaningless and was guaranteed to fail -- this is what
            # happened at attempt 3 on LAX: bw_min=2456 > n=2453).
            if not use_fixed_kernel:
                bw_min = min(bw_min, n_obs - 1)
            # v3 FIX: multi_bw_min needs one entry PER TERM (intercept +
            # each covariate), not a single shared value in a 1-element list.
            # The previous [bw_min] likely only constrained one term while
            # the others silently defaulted back to an unconstrained search
            # -- which explains why single-bandwidth GWR succeeded while
            # MGWR's per-covariate backfitting kept failing even as bw_min
            # was doubled. I'm not 100% certain of mgwr's exact internal
            # term ordering from memory, so this applies the SAME floor to
            # every term uniformly -- safe regardless of which index maps
            # to which covariate, unlike a 1-element list.
            multi_bw_min_list = [bw_min] * n_terms
            bw_display = f'{bw_min:,.0f} m' if use_fixed_kernel else f'{bw_min}'
            print(f'  Attempt {attempt}: bw_min={bw_display} applied to all {n_terms} terms '
                  f'(v3 fix). Multi-scale bandwidth selection (n={n_obs}), '
                  f'expect 10-30 min...')
            selector = Sel_BW(coords, y, X_std, kernel='bisquare',
                               fixed=use_fixed_kernel, multi=True)
            selector.search(criterion='AICc', multi_bw_min=multi_bw_min_list)

            print('  Fitting MGWR...')
            mgwr_model = MGWR(coords, y, X_std, selector, kernel='bisquare',
                              fixed=use_fixed_kernel, constant=True).fit()
            break
        except np.linalg.LinAlgError as e:
            print(f'  LinAlgError at bw_min={bw_min}: {e}')
            if not use_fixed_kernel and bw_min >= n_obs - 1:
                # Already at the sample-size ceiling for an adaptive
                # kernel -- doubling again is meaningless, stop here
                # rather than run a doomed attempt.
                raise RuntimeError(
                    f'MGWR hit a singular matrix even at bw_min={bw_min} '
                    f'(the adaptive-kernel ceiling for n={n_obs}). This is '
                    f'no longer a sample-size problem -- it points to '
                    f'structural local collinearity (e.g. a spatially '
                    f'clustered/zero-inflated covariate) that no k-NN '
                    f'floor can fix. See Wheeler & Tiefelsdorf (2005) and '
                    f'Wheeler (2007) on local collinearity in GWR/MGWR.'
                ) from e
            if attempt < max_attempts:
                bw_min *= 2
                bw_display = f'{bw_min:,.0f} m' if use_fixed_kernel else f'{bw_min}'
                print(f'  Retrying with bw_min={bw_display}...')
            else:
                raise RuntimeError(
                    f'MGWR still hit a singular matrix after {max_attempts} attempts, '
                    f'even with the per-term bw_min fix. Check the within-city variance '
                    f'diagnostic printed above -- a flagged (city, covariate) pair is now '
                    f'the more likely explanation than a sample-size floor. If nothing '
                    f'was flagged, this may need a fixed (distance-based) kernel instead '
                    f'of adaptive, given the three disjoint, non-contiguous study counties.'
                ) from e

    aicc = float(mgwr_model.aicc)
    r2   = float(mgwr_model.R2)
    print(f'  R²: {r2:.4f} | AICc: {aicc:.2f}')

    # v5 FIX: mgwr_model.bws turned out to be a real attribute but NOT the
    # per-term bandwidth array (it produced a ragged/inhomogeneous shape --
    # some other object that happens to share the name). Reordered to check
    # selector.bw FIRST: MGWR(coords, y, X, selector, ...) is constructed
    # FROM the selector specifically because MGWR pulls per-term bandwidths
    # from it internally -- this is the documented source, not a guess.
    # Every candidate is now validated (numeric, shape == n_terms) before
    # being accepted, and if NONE validate, this prints a full diagnostic
    # dump of every candidate's actual type/value so a fix doesn't require
    # another blind guess-and-rerun cycle.
    # v6 FIX: the v5 dump (pasted back from the previous session) showed
    # selector.bw is real and IS the documented source -- but it's a
    # 3-tuple (opt_bw, bw_history, score_history), not a bare array:
    #   selector.bw[0] -> shape (n_terms,)   <- the actual per-term optimal
    #                                            bandwidths, exactly what we want
    #   selector.bw[1] -> shape (n_iters, n_terms)  backfitting history
    #   selector.bw[2] -> shape (n_iters,)          AICc score history
    # This matches mgwr's internal Sel_BW.search(multi=True) return
    # convention -- that full tuple is what gets stored on `selector.bw`.
    # v5's np.array() cast of the whole tuple failed because element [1]
    # and [0] have different shapes (ragged) -- exactly the "not a clean
    # numeric array" message. Fix: unwrap tuple/list candidates element by
    # element and check EACH element for shape (n_terms,) before giving up
    # on that candidate entirely.
    candidates = [
        ('selector.bw',     getattr(selector, 'bw', None)),
        ('selector.bws',    getattr(selector, 'bws', None)),
        ('mgwr_model.bw',   getattr(mgwr_model, 'bw', None)),
        ('mgwr_model.bws',  getattr(mgwr_model, 'bws', None)),
    ]
    bw_array = None
    bw_source = None
    for desc, candidate in candidates:
        if candidate is None:
            continue

        if isinstance(candidate, (tuple, list)):
            found = False
            for j, elem in enumerate(candidate):
                try:
                    elem_arr = np.array(elem, dtype=float)
                except (ValueError, TypeError):
                    continue
                if elem_arr.ndim == 1 and elem_arr.shape[0] == n_terms:
                    bw_array = elem_arr
                    bw_source = f'{desc}[{j}]'
                    found = True
                    break
            if found:
                break
            print(f'  {desc}: tuple/list of length {len(candidate)}, no element had '
                  f'shape ({n_terms},) — skip')
            continue

        try:
            arr = np.array(candidate, dtype=float).flatten()
        except (ValueError, TypeError):
            print(f'  {desc}: exists but not a clean numeric array ({type(candidate).__name__}) — skip')
            continue
        if arr.shape[0] == n_terms:
            bw_array = arr
            bw_source = desc
            break
        else:
            print(f'  {desc}: shape {arr.shape}, expected ({n_terms},) — skip')

    if bw_array is None:
        print('\n  *** Could not find a valid per-term bandwidth array. Full dump: ***')
        for desc, candidate in candidates:
            print(f'    {desc}: type={type(candidate)}')
            print(f'      value={candidate!r}'[:500])
        raise RuntimeError(
            'None of the checked attributes yielded a clean length-%d numeric array. '
            'See the dump above -- it has everything needed to identify the correct '
            'attribute without another blind guess.' % n_terms
        )
    print(f'  (bandwidths read from {bw_source})')

    col_names_mgwr = ['intercept'] + covars
    bw_unit_label = 'Bandwidth (m, fixed)' if use_fixed_kernel else 'Bandwidth (k-NN)'
    print(f'\n  Per-covariate MGWR bandwidths (standardized covariates — THE HEADLINE RESULT):')
    print(f'  {"Covariate":<20} {bw_unit_label:>18} '
          f'{"Coef median":>14} {"Coef IQR":>12}')
    print('  ' + '-'*68)
    bw_dict = {}
    for i, name in enumerate(col_names_mgwr):
        bw_i  = int(bw_array[i])
        vals  = mgwr_model.params[:, i]
        med_v = float(np.median(vals))
        iqr_v = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        bw_dict[name] = bw_i
        print(f'  {name:<20} {bw_i:>18}  {med_v:>14.4f}  {iqr_v:>12.4f}')

    mgwr_out = df[['GEOID', 'city', 'lon', 'lat']].copy()
    tval_out = df[['GEOID', 'city', 'lon', 'lat']].copy()
    for i, name in enumerate(col_names_mgwr):
        mgwr_out[f'mgwr_{name}'] = mgwr_model.params[:, i]
        if hasattr(mgwr_model, 'tvalues'):
            tval_out[f'mgwr_{name}_t'] = mgwr_model.tvalues[:, i]
        else:
            tval_out[f'mgwr_{name}_t'] = np.nan

    # v7 FIX: MGWRResults.localR2 raises NotImplementedError in the
    # installed mgwr package version -- it's a genuine, documented library
    # gap (mgwr/gwr.py: "Not yet implemented for multiple bandwidths"),
    # not something to work around by re-deriving a substitute local R2
    # from scratch (that would require reproducing the GWR local-weighted
    # sum-of-squares calculation per term/bandwidth, which is enough of a
    # different computation from what the diagnostics guide describes that
    # a hand-rolled version could silently be wrong in a way that's hard to
    # catch). Filling this column with NaN and saying so explicitly is the
    # honest choice, consistent with "no fabrication, no assumptions" --
    # local R2 comparisons in Section 3 of the diagnostics guide should be
    # treated as GWR-only until this is available, or paper2a_local_report.py
    # computes it independently outside the mgwr package.
    try:
        mgwr_out['mgwr_local_r2'] = mgwr_model.localR2
    except NotImplementedError:
        mgwr_out['mgwr_local_r2'] = np.nan
        print('  NOTE: mgwr_model.localR2 raised NotImplementedError (known mgwr package')
        print('  limitation for multi-bandwidth/MGWR models -- not implemented for MGWR')
        print('  in this package version, only plain GWR). Column filled with NaN rather')
        print('  than a hand-derived substitute. Global R2 above is still valid and used')
        print('  for Table 2 / AICc comparison; per-tract local R2 for MGWR is simply')
        print('  unavailable from this package version -- note this explicitly in Methods')
        print('  if Section 3 of the diagnostics guide is applied to MGWR specifically.')

    result = {
        'model': 'MGWR', 'n': len(df), 'kernel': 'bisquare',
        'kernel_type': 'fixed' if use_fixed_kernel else 'adaptive',
        'criterion': 'AICc',
        'covariates_standardized': True, 'bw_min_used': bw_min, 'covars_used': covars,
        'r2': round(r2, 4), 'aicc': round(aicc, 2),
        'per_covariate_bandwidths': bw_dict,
        'coef_medians_standardized': {
            name: round(float(np.median(mgwr_model.params[:, i])), 6)
            for i, name in enumerate(col_names_mgwr)
        },
        'notes': (
            ('Bandwidth = fixed distance in metres (EPSG:5070 projected coords). '
             'Switched from adaptive to fixed kernel for this city specifically -- '
             'see Methods note on local collinearity (Wheeler & Tiefelsdorf 2005; '
             'Wheeler 2007). '
             if use_fixed_kernel else
             'Bandwidth = number of nearest-neighbour observations (adaptive). ')
            + 'Coefficients are on STANDARDIZED covariate scale (z-score) — this '
              'was necessary for numerical stability (see module docstring) and '
              'also puts GreenEx_pct and BlueEx_pct bandwidths on a fair, '
              'scale-comparable footing for the core Paper 2a claim.'
        ),
    }
    return result, mgwr_out, tval_out


def print_comparison(ols_res, gwr_res, mgwr_res, city_label=''):
    print(f'\n{"="*70}')
    print(f'TABLE 2 SCAFFOLD — OLS / GWR / MGWR COMPARISON — {city_label}')
    print(f'{"="*70}')
    print(f'{"Model":<8} {"n":>8} {"R²":>8} {"AICc":>12}  Notes')
    print('-' * 60)
    print(f'{"OLS":<8} {ols_res["n"]:>8} {ols_res["r2"]:>8.4f} '
          f'{ols_res["aic"]:>12.2f}  Original units; Cond.No.={ols_res["condition_number"]}')
    print(f'{"GWR":<8} {gwr_res["n"]:>8} {gwr_res["r2"]:>8.4f} '
          f'{gwr_res["aicc"]:>12.2f}  Standardized; bw={gwr_res["bandwidth"]} k-NN')
    print(f'{"MGWR":<8} {mgwr_res["n"]:>8} {mgwr_res["r2"]:>8.4f} '
          f'{mgwr_res["aicc"]:>12.2f}  Standardized; per-covariate bw (below)')
    print()
    bw_unit = 'm (fixed)' if mgwr_res.get('kernel_type') == 'fixed' else 'k-NN'
    for name, bw in mgwr_res['per_covariate_bandwidths'].items():
        bw_display = f'{bw:,.0f}' if mgwr_res.get('kernel_type') == 'fixed' else f'{bw:>6}'
        print(f'  {name:<20} {bw_display} {bw_unit}')
    print()
    if mgwr_res['aicc'] < gwr_res['aicc']:
        print(f'→ [{city_label}] MGWR improves AICc over GWR.')
    else:
        print(f'→ [{city_label}] MGWR does NOT improve AICc over GWR.')


def print_cross_city_bandwidth_comparison(city_results: dict):
    """THE key evidentiary table for the per-city design: does the GreenEx
    vs BlueEx bandwidth gap replicate independently across all three cities,
    or is it a one-off artifact of a single pooled fit? This is a stronger
    test of the core Paper 2a claim than the original pooled model was,
    precisely because each city's bandwidth search never saw the other two
    cities' data at all."""
    print(f'\n{"="*70}')
    print('CROSS-CITY MGWR BANDWIDTH COMPARISON (Table 3, revised design)')
    print('Independent per-city fits -- no pooling, no cross-city contamination')
    print(f'{"="*70}')
    print(f'{"City":<40} {"Kernel":>10} {"GreenEx_pct bw":>15} {"BlueEx_pct bw":>15} {"Gap":>10} {"Gap %":>8}')
    print('-' * 100)
    directions = []
    for city, res in city_results.items():
        bw_dict = res['mgwr']['per_covariate_bandwidths']
        kernel = res['mgwr'].get('kernel_type', 'adaptive')
        g = bw_dict.get('GreenEx_pct')
        b = bw_dict.get('BlueEx_pct')
        dropped = res.get('dropped_covars', [])
        if g is None or b is None:
            reason = f'dropped: {dropped}' if dropped else 'MGWR did not produce this term'
            print(f'{city:<40} {kernel:>10} {"N/A — " + reason:>15}')
            continue
        gap = g - b
        gap_pct = 100 * abs(gap) / max(g, b)
        directions.append(np.sign(gap))
        g_disp = f'{g:,.0f}' if kernel == 'fixed' else f'{g}'
        b_disp = f'{b:,.0f}' if kernel == 'fixed' else f'{b}'
        gap_disp = f'{gap:,.0f}' if kernel == 'fixed' else f'{gap}'
        print(f'{city:<40} {kernel:>10} {g_disp:>15} {b_disp:>15} {gap_disp:>10} {gap_pct:>7.1f}%')

    print()
    if any(res['mgwr'].get('kernel_type') == 'fixed' for res in city_results.values() if 'mgwr' in res):
        print('NOTE: at least one city used a FIXED kernel (bandwidths in metres) while')
        print('others used ADAPTIVE (bandwidths in nearest-neighbour count). Gap % values')
        print('are NOT directly comparable in magnitude across cities using different')
        print('kernel types -- only the SIGN (direction) comparison below is valid across')
        print('kernel types, since it tests whether GreenEx is more or less local than')
        print('BlueEx WITHIN each city\'s own consistent units, not across cities.')
        print()
    if len(directions) >= 2 and len(set(directions)) == 1 and directions[0] != 0:
        sign = 'GreenEx more local than BlueEx' if directions[0] < 0 else 'BlueEx more local than GreenEx'
        print(f'→ Bandwidth gap direction is CONSISTENT across all {len(directions)} cities ({sign}).')
        print('  This is a materially stronger result than a single pooled fit would have')
        print('  been -- it is a replicated finding across three independent, non-')
        print('  contiguous study areas, not an artifact of one bandwidth search.')
    elif len(directions) >= 2:
        print(f'→ Bandwidth gap direction is NOT consistent across cities.')
        print('  Report this honestly as context-dependent (same standard already applied')
        print('  to the kappa validation reframe) rather than picking the city that fits')
        print('  the preferred narrative and omitting the others.')
    else:
        print('→ Fewer than 2 cities produced a valid MGWR fit -- cannot assess consistency.')

    any_dropped = {city: res.get('dropped_covars', []) for city, res in city_results.items()
                   if res.get('dropped_covars')}
    if any_dropped:
        print(f'\nNOTE FOR METHODS/LIMITATIONS: {any_dropped}')
        print('At least one city used a reduced covariate set. The cross-city bandwidth')
        print('comparison above only includes cities where BOTH GreenEx_pct and')
        print('BlueEx_pct were retained -- state this explicitly rather than silently')
        print('comparing 2 of 3 cities under a claim written as if it were 3.')


# v11: cities where the adaptive k-NN kernel structurally cannot stabilize
# MGWR's per-term backfitting (see run_mgwr's v11 comment for the LAX
# diagnosis: zero-inflated, spatially clustered BlueEx_pct). For these
# cities, try a FIXED (distance) kernel first; only drop the offending
# covariate if the fixed kernel ALSO fails. Both the attempt and its
# outcome are recorded on the result dict for honest Methods reporting.
FIXED_KERNEL_CITIES = {'Los Angeles (LA County, CA)'}


def run_mgwr_with_fallbacks(df_city: pd.DataFrame, coords: np.ndarray,
                             city_covars: list, city_name: str,
                             gwr_bw: int = None) -> dict:
    """Two-stage fallback for cities in FIXED_KERNEL_CITIES:
    1. Fixed (distance) kernel, full covariate set.
    2. If that still fails: fixed kernel, BlueEx_pct dropped from MGWR
       ONLY (OLS and GWR keep the full set -- this mirrors Atlanta's
       existing pooled-SD-based drop, but triggered by a different,
       spatial-clustering diagnostic specific to LAX).
    Both attempts and whichever one succeeds are recorded in the returned
    result's 'mgwr_fallback_log' for the Methods section -- this should
    never be silently resolved without a paper trail.
    """
    fallback_log = []
    use_fixed = city_name in FIXED_KERNEL_CITIES

    try:
        mgwr_res, mgwr_out, tval_out = run_mgwr(
            df_city, coords, city_covars, gwr_bw=gwr_bw, use_fixed_kernel=use_fixed)
        fallback_log.append({
            'attempt': 1, 'kernel': 'fixed' if use_fixed else 'adaptive',
            'covars': city_covars, 'outcome': 'success',
        })
        mgwr_res['mgwr_fallback_log'] = fallback_log
        return mgwr_res, mgwr_out, tval_out
    except RuntimeError as e:
        fallback_log.append({
            'attempt': 1, 'kernel': 'fixed' if use_fixed else 'adaptive',
            'covars': city_covars, 'outcome': 'failed', 'error': str(e),
        })
        if not use_fixed or 'BlueEx_pct' not in city_covars:
            # Not a city with a configured fallback, or nothing left to
            # drop -- re-raise so v8's per-city isolation catches it.
            raise

        print(f'\n{"─"*60}')
        print(f'FALLBACK 2 for {city_name}: fixed kernel alone did not resolve the')
        print(f'singularity. Dropping BlueEx_pct from MGWR ONLY (retained in OLS/GWR),')
        print(f'per Wheeler (2007, Environment and Planning A 39(10):2464-2481) on')
        print(f'remedial covariate removal under confirmed local collinearity.')
        print(f'{"─"*60}')
        reduced_covars = [c for c in city_covars if c != 'BlueEx_pct']
        try:
            mgwr_res, mgwr_out, tval_out = run_mgwr(
                df_city, coords, reduced_covars, gwr_bw=gwr_bw, use_fixed_kernel=use_fixed)
            fallback_log.append({
                'attempt': 2, 'kernel': 'fixed', 'covars': reduced_covars,
                'outcome': 'success', 'dropped_for_mgwr_only': ['BlueEx_pct'],
            })
            mgwr_res['mgwr_fallback_log'] = fallback_log
            mgwr_res['dropped_for_mgwr_only'] = ['BlueEx_pct']
            return mgwr_res, mgwr_out, tval_out
        except RuntimeError as e2:
            fallback_log.append({
                'attempt': 2, 'kernel': 'fixed', 'covars': reduced_covars,
                'outcome': 'failed', 'error': str(e2),
            })
            print(f'\n*** Both fallbacks failed for {city_name}. Full log: ***')
            for entry in fallback_log:
                print(f'    {entry}')
            raise RuntimeError(
                f'{city_name}: both the fixed-kernel attempt and the '
                f'BlueEx_pct-dropped fixed-kernel attempt failed. See the '
                f'fallback log above -- this needs a human decision, not '
                f'another automatic retry.'
            ) from e2



def run_city_analysis(df_city: pd.DataFrame, city_name: str, pooled_stats: dict,
                      skip_mgwr: bool = False) -> dict:
    print(f'\n\n{"#"*70}')
    print(f'# {city_name}  (n={len(df_city)})')
    print(f'{"#"*70}')

    dropped = check_within_city_variance(df_city, COVARS, pooled_stats, city_name)
    city_covars = [c for c in COVARS if c not in dropped]
    if dropped:
        print(f'*** {city_name} MODEL SPECIFICATION CHANGED: dropped {dropped} ***')
        print(f'*** Formula for {city_name}: {OUTCOME} ~ {" + ".join(city_covars)} ***')
        print('*** Document this explicitly in Methods as a site-specific specification. ***\n')

    if len(city_covars) < 2:
        raise RuntimeError(
            f'{city_name}: only {len(city_covars)} covariate(s) remain after dropping '
            f'near-constant ones ({dropped}) -- not enough for a meaningful GWR/MGWR model. '
            f'This city cannot be modeled with the current covariate set; report as a data '
            f'limitation rather than forcing a fit.'
        )

    coords = get_coords_metres(df_city)

    ols_res, ols_resid = run_ols(df_city, city_covars)
    gwr_res, gwr_out = run_gwr(df_city, coords, city_covars)

    if skip_mgwr:
        mgwr_res = {'model': 'MGWR', 'n': len(df_city), 'r2': float('nan'), 'aicc': float('nan'),
                    'per_covariate_bandwidths': {}, 'coef_medians_standardized': {}}
        mgwr_out = gwr_out[['GEOID', 'city', 'lon', 'lat']].copy()
        tval_out = None
    else:
        # MGWR failure (e.g. after all fallback attempts) produces a
        # placeholder result here (NaN bandwidths, empty dict, failure
        # reason recorded) rather than letting the exception propagate --
        # that keeps this city's already-computed, valid OLS and GWR
        # results from being discarded along with the failed MGWR stage.
        try:
            mgwr_res, mgwr_out, tval_out = run_mgwr_with_fallbacks(
                df_city, coords, city_covars, city_name, gwr_bw=gwr_res['bandwidth'])
        except RuntimeError as e:
            print(f'\n*** MGWR failed entirely for {city_name} after all fallbacks. ***')
            print(f'*** OLS and GWR results above are still valid and will be saved. ***')
            print(f'*** MGWR failure reason: {e} ***\n')
            mgwr_res = {
                'model': 'MGWR', 'n': len(df_city), 'r2': float('nan'), 'aicc': float('nan'),
                'kernel_type': 'none (failed)', 'per_covariate_bandwidths': {},
                'coef_medians_standardized': {}, 'mgwr_failed': True,
                'mgwr_failure_reason': str(e),
            }
            mgwr_out = gwr_out[['GEOID', 'city', 'lon', 'lat']].copy()
            tval_out = None

    print_comparison(ols_res, gwr_res, mgwr_res, city_label=city_name)

    return {
        'ols': ols_res, 'gwr': gwr_res, 'gwr_out': gwr_out,
        'mgwr': mgwr_res, 'mgwr_out': mgwr_out, 'tval_out': tval_out,
        'dropped_covars': dropped, 'covars_used': city_covars,
    }


def save_outputs(city_results: dict):
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    if not city_results:
        # v10 FIX: previously pd.concat([]) below crashed with "No objects
        # to concatenate" whenever every requested city failed (e.g. a
        # standalone --city LAX run that hit the MGWR singularity). That's
        # a confusing secondary crash on top of the real error, and it also
        # meant NOTHING got saved even if some cities in a multi-city run
        # had already succeeded before v8's isolation was added. Now: if
        # there's truly nothing to save, say so plainly and return instead
        # of crashing on an empty concat.
        print(f'\nNo cities produced results -- nothing to save to {PROC_DIR}.')
        return

    all_ols = {}
    all_bw_payload = {'run_datetime': datetime.now().isoformat(), 'year': TARGET_YEAR,
                       'design': 'per-city independent MGWR fits (no pooling)',
                       'covariates_standardized': True, 'cities': {}}
    gwr_out_all, mgwr_out_all, tval_out_all = [], [], []

    for city, res in city_results.items():
        all_ols[city] = res['ols']
        all_bw_payload['cities'][city] = {
            'n': res['mgwr']['n'],
            'covars_used': res.get('covars_used', COVARS),
            'dropped_covars': res.get('dropped_covars', []),
            'aicc_ols': res['ols']['aic'],
            'aicc_gwr': res['gwr']['aicc'],
            'aicc_mgwr': res['mgwr']['aicc'],
            'per_covariate_bandwidths': res['mgwr']['per_covariate_bandwidths'],
            'mgwr_kernel_type': res['mgwr'].get('kernel_type', 'adaptive'),
            'mgwr_dropped_for_mgwr_only': res['mgwr'].get('dropped_for_mgwr_only', []),
            'mgwr_fallback_log': res['mgwr'].get('mgwr_fallback_log', []),
        }
        gwr_out_all.append(res['gwr_out'])
        mgwr_out_all.append(res['mgwr_out'])
        if res['tval_out'] is not None:
            tval_out_all.append(res['tval_out'])

    with open(PROC_DIR / 'paper2a_ols_results_per_city.json', 'w') as f:
        json.dump(all_ols, f, indent=2)
    if gwr_out_all:
        pd.concat(gwr_out_all, ignore_index=True).to_csv(PROC_DIR / 'paper2a_gwr_results.csv', index=False)
    if mgwr_out_all:
        pd.concat(mgwr_out_all, ignore_index=True).to_csv(PROC_DIR / 'paper2a_mgwr_results.csv', index=False)
    if tval_out_all:
        pd.concat(tval_out_all, ignore_index=True).to_csv(PROC_DIR / 'paper2a_mgwr_tvalues.csv', index=False)
    with open(PROC_DIR / 'paper2a_mgwr_bandwidths.json', 'w') as f:
        json.dump(all_bw_payload, f, indent=2)

    print(f'\nAll outputs saved to {PROC_DIR} (combined across cities, "city" column')
    print('preserved in each CSV -- compatible with paper2a_local_report.py as-is).')


def main():
    parser = argparse.ArgumentParser(
        description='Paper 2a: OLS + GWR + MGWR fit INDEPENDENTLY per city (v3 — no pooling)')
    parser.add_argument('--panel', default=str(PROC_DIR / 'ugbs_panel_2015_2023.csv'))
    parser.add_argument('--skip-mgwr', action='store_true')
    parser.add_argument('--city', choices=['ATL', 'MIA', 'LAX'],
                        help='Run a single city only (default: all three, sequentially)')
    args = parser.parse_args()

    panel_path = pathlib.Path(args.panel)
    if not panel_path.exists():
        sys.exit(f'ERROR: Panel CSV not found: {panel_path}')

    print('='*70)
    print('Paper 2a — OLS + GWR + MGWR (v3 — INDEPENDENT per-city fits)')
    print('No pooling: each city\'s bandwidth search only ever sees its own tracts.')
    print('='*70)

    df_all = load_and_filter(panel_path)
    pooled_stats = compute_pooled_stats(df_all, COVARS)
    print('\nPooled (3-city) covariate SD -- reference for per-city variance checks:')
    for cov, s in pooled_stats.items():
        print(f'  {cov:<15} mean={s["mean"]:8.3f}  sd={s["sd"]:8.3f}')

    code_to_name = {code: name for code, (prefix, name) in CITY_PREFIXES.items()}
    cities_to_run = [args.city] if args.city else list(CITY_PREFIXES.keys())

    # v8 FIX: previously an exception in ANY city (e.g. LAX's MGWR
    # singular-matrix failure) propagated straight out of this loop and
    # killed the whole run -- meaning ATL and MIA's already-completed
    # results were silently discarded, even when running --city ATL alone
    # had proven ATL's fit was fine. Each city's run is now isolated in its
    # own try/except: a failure is logged with its full traceback and the
    # city name, then the loop moves on. save_outputs() at the end writes
    # whatever succeeded. Nothing about the underlying LAX singularity is
    # fixed by this -- it's purely "don't lose good results to a bad one."
    failed_cities = {}
    city_results = {}
    for code in cities_to_run:
        city_name = code_to_name[code]
        df_city = df_all[df_all['city'] == city_name].copy()
        if len(df_city) == 0:
            print(f'WARNING: no rows for {city_name} — skipping')
            continue
        try:
            city_results[city_name] = run_city_analysis(
                df_city, city_name, pooled_stats, skip_mgwr=args.skip_mgwr)
        except Exception as e:
            import traceback
            print(f'\n{"!"*70}')
            print(f'! {city_name} FAILED — see traceback below. Continuing to next city.')
            print(f'{"!"*70}')
            traceback.print_exc()
            failed_cities[city_name] = str(e)
            print(f'{"!"*70}\n')

    if not args.skip_mgwr:
        print_cross_city_bandwidth_comparison(city_results)

    save_outputs(city_results)

    print(f'\n=== mgwr_analysis.py (v3 — per-city) COMPLETE ===')
    print(f'Succeeded: {list(city_results.keys())}')
    if failed_cities:
        print(f'FAILED (results NOT saved for these): {list(failed_cities.keys())}')
        for city, msg in failed_cities.items():
            print(f'  {city}: {msg}')
        print('Cross-city Table 3 comparison above only reflects cities that succeeded --')
        print('do not report the cross-city consistency claim as covering all 3 if one failed.')
        sys.exit(1)


if __name__ == '__main__':
    main()
