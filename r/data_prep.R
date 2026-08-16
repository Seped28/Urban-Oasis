# data_prep.R — UrbanWell Analytics
# Builds the master panel and spatial objects for all downstream models.
# Run once. Outputs:
#   data/processed/ugbs_panel_2015_2023.csv     <- flat panel
#   data/processed/ugbs_panel_sf_5070.rds       <- sf object projected to NAD83 Albers
#   data/processed/spatial_weights_k8.rds       <- kNN-8 listw object

suppressPackageStartupMessages({
  library(sf)
  library(spdep)
  library(GWmodel)
  library(car)
  library(tidyverse)
  library(splm)
  library(tidycensus)
})

# -- 1. Load GEE exports -------------------------------------------------------
# Expects one CSV per year per variable. Stacks into a long panel.
# Column standards: GEOID (11-char string), BS_year/GS_year, BS_exposure/GS_exposure

load_gee_stack <- function(pattern, value_col, year_col) {
  files <- list.files("data/raw/gee_exports", pattern = pattern,
                      full.names = TRUE, recursive = TRUE)
  if (length(files) == 0) stop(paste("No GEE export files found matching:", pattern))
  map_dfr(files, function(f) {
    df <- read_csv(f, col_types = cols(GEOID = col_character()), show_col_types = FALSE)
    df <- df |> mutate(GEOID = str_pad(GEOID, 11, "left", "0"))
    df
  })
}

cat("Loading GEE exports...\n")
bs_raw <- load_gee_stack("BlueSpace_USA_|BlueSpace_GA_", "BS_exposure", "BS_year")
gs_raw <- load_gee_stack("GreenSpace_|UGBS_HLS", "GS_exposure", "GS_year")

bs_panel <- bs_raw |>
  select(GEOID, year = BS_year, BlueEx = BS_exposure, pop_sum_bs = pop_sum)

# GEE greenspace exports use different column names depending on which
# script produced them (HLS-era vs S2-era) -- handle both.
if ("GS_exposure" %in% names(gs_raw)) {
  gs_panel <- gs_raw |>
    select(GEOID, year = GS_year, GreenEx = GS_exposure)
} else if ("GS_exposure_HLS30" %in% names(gs_raw)) {
  gs_panel <- gs_raw |>
    select(GEOID, year, GreenEx = GS_exposure_HLS30)
} else {
  stop("Greenspace export column not recognised. Check GEE output headers.")
}

cat("Blue space rows:", nrow(bs_panel), "| Unique tracts:", n_distinct(bs_panel$GEOID), "\n")
cat("Green space rows:", nrow(gs_panel), "| Unique tracts:", n_distinct(gs_panel$GEOID), "\n")

# -- 2. Load FMD from PLACES (after nhgis_crosswalk.py) ------------------------
fmd_files <- list.files("data/processed", pattern = "FMD_harmonized_",
                        full.names = TRUE)
if (length(fmd_files) == 0) {
  stop("FMD harmonized files not found. Run python/nhgis_crosswalk.py first.")
}
fmd_panel <- map_dfr(fmd_files, function(f) {
  read_csv(f, col_types = cols(GEOID = col_character()), show_col_types = FALSE) |>
    mutate(GEOID = str_pad(GEOID, 11, "left", "0"))
}) |>
  rename(FMD = FMD_interp) |>
  select(GEOID, year, FMD)

cat("FMD rows:", nrow(fmd_panel), "\n")

# -- 3. Load ACS covariates -----------------------------------------------------
acs_file <- "data/processed/ACS_panel_2015_2023.csv"
if (!file.exists(acs_file)) {
  stop("ACS panel not found. Run python/00_download_acs.py first.")
}
acs_panel <- read_csv(acs_file,
                      col_types = cols(GEOID = col_character()),
                      show_col_types = FALSE) |>
  mutate(GEOID = str_pad(GEOID, 11, "left", "0"))

cat("ACS rows:", nrow(acs_panel), "\n")

# -- 4. Additional covariates: smoking, drinking --------------------------------
# From the same PLACES download as FMD, different measureid. If not yet
# extracted, flag and continue -- the model drops rows with NA.
smoke_file <- "data/processed/SMOKE_panel.csv"
drink_file <- "data/processed/DRINK_panel.csv"
if (file.exists(smoke_file) & file.exists(drink_file)) {
  smoke <- read_csv(smoke_file, col_types = cols(GEOID = col_character()), show_col_types = FALSE) |>
    mutate(GEOID = str_pad(GEOID, 11, "left", "0")) |>
    rename(Smoke = data_value) |> select(GEOID, year, Smoke)
  drink <- read_csv(drink_file, col_types = cols(GEOID = col_character()), show_col_types = FALSE) |>
    mutate(GEOID = str_pad(GEOID, 11, "left", "0")) |>
    rename(Drink = data_value) |> select(GEOID, year, Drink)
} else {
  cat("WARNING: Smoke/Drink covariates not found -- will be NA in model.\n")
  cat("Add CSMOKING and BINGE to python/00_download_places.py measureid filter.\n")
  smoke <- tibble(GEOID = character(), year = integer(), Smoke = double())
  drink <- tibble(GEOID = character(), year = integer(), Drink = double())
}

# -- 5. Merge all sources ---------------------------------------------------------
panel_raw <- bs_panel |>
  inner_join(gs_panel,  by = c("GEOID", "year")) |>
  inner_join(fmd_panel, by = c("GEOID", "year")) |>
  left_join(acs_panel,  by = c("GEOID", "year")) |>
  left_join(smoke,      by = c("GEOID", "year")) |>
  left_join(drink,      by = c("GEOID", "year"))

cat("\nMerged panel:", nrow(panel_raw), "rows |",
    n_distinct(panel_raw$GEOID), "tracts |",
    n_distinct(panel_raw$year), "years\n")

# -- 6. Derive variables and year dummies ------------------------------------------
panel <- panel_raw |>
  mutate(
    MedIn_k     = MedIncome / 1000,
    GreenEx_pct = GreenEx * 100,
    BlueEx_pct  = BlueEx  * 100,
    year_fct    = factor(year)
  )

for (y in 2016:2023) {
  panel[[paste0("dum_", y)]] <- as.integer(panel$year == y)
}

# -- 7. Outlier removal, accumulated on data_clean --------------------------------
remove_outliers_var <- function(df, var) {
  if (!var %in% names(df)) return(df)
  out <- boxplot.stats(df[[var]])$out
  df[!df[[var]] %in% out, ]
}

data_clean <- panel
for (var in c("FMD", "BlueEx_pct", "GreenEx_pct", "Smoke", "Drink",
              "MedIn_k", "EduAt")) {
  n_before   <- nrow(data_clean)
  data_clean <- remove_outliers_var(data_clean, var)
  cat(sprintf("  Outlier %s: %d -> %d rows\n", var, n_before, nrow(data_clean)))
}
cat("Rows after outlier removal:", nrow(data_clean), "\n")

# -- 8. Join SVI --------------------------------------------------------------------
# CDC/ATSDR Social Vulnerability Index -- RPL_THEMES (0-1 composite, higher = more vulnerable)
# Forward-filled from biennial releases (2014/2016/2018/2020/2022) by python/process_svi.py
svi_file <- "data/processed/SVI_panel_2015_2023.csv"
if (file.exists(svi_file)) {
  svi <- read_csv(svi_file,
                  col_types = cols(GEOID = col_character()),
                  show_col_types = FALSE) |>
    mutate(GEOID = str_pad(GEOID, 11, "left", "0")) |>
    select(GEOID, year, RPL_THEMES, SVI_quartile)

  data_clean <- data_clean |>
    left_join(svi, by = c("GEOID", "year"))

  n_null <- sum(is.na(data_clean$RPL_THEMES))
  cat(sprintf("SVI joined: RPL_THEMES nulls = %d (%.1f%%) -- suppressed/unmatched tracts\n",
              n_null, 100 * n_null / nrow(data_clean)))
} else {
  cat("WARNING: SVI panel not found at", svi_file, "\n")
  cat("Run python/process_svi.py first. RPL_THEMES will be NA.\n")
  data_clean$RPL_THEMES  <- NA_real_
  data_clean$SVI_quartile <- NA_integer_
}

# -- 9. OLS baseline -------------------------------------------------------------------
model_ols <- lm(
  FMD ~ GreenEx_pct + BlueEx_pct + Smoke + Drink + MedIn_k + EduAt +
        RPL_THEMES +
        dum_2016 + dum_2017 + dum_2018 + dum_2019 +
        dum_2020 + dum_2021 + dum_2022 + dum_2023,
  data = data_clean
)
cat("\nOLS summary:\n")
print(summary(model_ols))

# VIF check -- warn if > 5 for primary exposures
vif_vals <- vif(model_ols)
cat("\nVIF:\n"); print(round(vif_vals, 2))
if (any(vif_vals[c("GreenEx_pct","BlueEx_pct")] > 5, na.rm = TRUE)) {
  warning("VIF > 5 for GS or BS. Run separate models + compare AICc before joint model.")
}
if (!is.na(vif_vals["RPL_THEMES"]) && vif_vals["RPL_THEMES"] > 5) {
  warning("VIF > 5 for RPL_THEMES -- collinearity with MedIn_k/EduAt likely. Consider SVI_quartile as factor instead.")
}

# -- 10. Spatial objects, projected to metres (NAD83 CONUS Albers) -----------------
tiger_path <- "data/raw/tiger/tl_2020_us_tract.shp"
if (!file.exists(tiger_path)) {
  cat("\nTIGER shapefile not found at", tiger_path, "\n")
  cat("Download: curl -O https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_us_tract.zip\n")
  cat("Or per-state: https://www2.census.gov/geo/tiger/TIGER2020/TRACT/\n")
  cat("Spatial models (GWR/MGWR/GTWR) require this. Saving flat panel only.\n")
  write_csv(data_clean, "data/processed/ugbs_panel_2015_2023.csv")
  cat("Flat panel saved -> data/processed/ugbs_panel_2015_2023.csv\n")
  stop("Download TIGER shapefiles then re-run data_prep.R")
}

tracts_sf <- st_read(tiger_path, quiet = TRUE) |>
  mutate(GEOID = as.character(GEOID))

data_sf <- data_clean |>
  inner_join(tracts_sf |> select(GEOID, geometry), by = "GEOID") |>
  st_as_sf()

# EPSG:5070 (NAD83 CONUS Albers) -- units in metres
data_proj  <- st_transform(data_sf, crs = 5070)
coords_m   <- st_coordinates(st_centroid(data_proj))

# Centroid coordinates in WGS84 for MGWR, which expects lon/lat.
centroids_wgs84 <- st_transform(st_centroid(data_sf), 4326) |>
  st_coordinates()
data_proj$lon <- centroids_wgs84[, 1]
data_proj$lat <- centroids_wgs84[, 2]

# Spatial weights: k=8 nearest neighbours, distances in metres
nb    <- knn2nb(knearneigh(coords_m, k = 8))
listw <- nb2listw(nb)

# Moran's I on OLS residuals -- confirms spatial autocorrelation before
# proceeding to GWR/MGWR/GTWR
moran_result <- moran.test(residuals(model_ols), listw)
cat("\nMoran's I on OLS residuals:\n")
print(moran_result)
if (moran_result$p.value < 0.05) {
  cat("-> Spatial autocorrelation confirmed -- proceed to GWR/MGWR/GTWR\n")
} else {
  cat("-> WARNING: no significant spatial autocorrelation -- check data merge\n")
}

# -- 11. Save all outputs -----------------------------------------------------------
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
write_csv(st_drop_geometry(data_proj), "data/processed/ugbs_panel_2015_2023.csv")
saveRDS(data_proj, "data/processed/ugbs_panel_sf_5070.rds")
saveRDS(listw,     "data/processed/spatial_weights_k8.rds")

cat("\n=== data_prep.R COMPLETE ===\n")
cat("Outputs:\n")
cat("  data/processed/ugbs_panel_2015_2023.csv\n")
cat("  data/processed/ugbs_panel_sf_5070.rds\n")
cat("  data/processed/spatial_weights_k8.rds\n")

