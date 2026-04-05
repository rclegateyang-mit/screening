"""Data loading and feature engineering for education imputation.

All functions are stateless. Data flows:
  ICF_US  -->  load_icf  -->  encode_demographics  -->  Model 1 features
  RCF     -->  load_rcf  --+
  ICF     -----------------+-->  compute_tract_aggregates  -->  Model 2 features
  EHF     -->  load_ehf_multi_state  -->  identify_dominant_job  --+
  ECF     -->  load_ecf_naics  --------------------------------+-->  compute_naics_features  -->  Model 3 features
  RCF+NAICS  -->  compute_tract_naics_distribution  ------------>  Model 3 features (cont.)

  build_feature_matrix  -->  assemble for model level 1/2/3
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from screening.rdc.helpers import log, log_section, time_block, read_sas_chunked

from . import config as C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """Decode byte-string columns from SAS reads to str."""
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().iloc[:5] if len(df[col].dropna()) > 0 else pd.Series()
        if len(sample) > 0 and isinstance(sample.iloc[0], bytes):
            df[col] = df[col].apply(
                lambda v: v.decode("utf-8", errors="replace").strip() if isinstance(v, bytes) else v
            )
    return df


def _read_file(path: str, columns: list[str] | None = None,
               filters: dict | None = None,
               logfile: str | None = None) -> pd.DataFrame:
    """Read SAS or Parquet depending on config. Applies column select + filters."""
    p = Path(path)
    if C.USE_PARQUET or p.suffix == ".parquet" or p.is_dir():
        log(f"  Reading parquet: {path}", logfile)
        df = pd.read_parquet(path, columns=columns)
    else:
        chunks = []
        for chunk in read_sas_chunked(path):
            chunk.columns = chunk.columns.str.lower()
            chunk = _decode_bytes(chunk)
            if columns:
                keep = [c for c in columns if c in chunk.columns]
                chunk = chunk[keep]
            if filters:
                for col, (lo, hi) in filters.items():
                    if col in chunk.columns:
                        chunk = chunk[(chunk[col] >= lo) & (chunk[col] <= hi)]
            if len(chunk) > 0:
                chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return df


# ---------------------------------------------------------------------------
# ICF loading & demographic encoding
# ---------------------------------------------------------------------------

def load_icf(debug_n: int | None = None,
             logfile: str | None = None) -> pd.DataFrame:
    """Load ICF_US, filter to age 25+, compute target variable.

    Returns DataFrame indexed by PIK with columns:
      age, sex, pob, race, ethnicity, educ_c, educ_years, educ_c_imputed
    """
    log_section("Loading ICF_US", logfile)

    cols = [C.COL_PIK, C.COL_DOB, C.COL_SEX, C.COL_POB, C.COL_RACE,
            C.COL_ETHNICITY, C.COL_EDUC, C.COL_EDUC_IMPUTED]

    with time_block("Read ICF", logfile):
        df = _read_file(C.ICF_US_PATH, columns=cols, logfile=logfile)

    log(f"  Raw ICF rows: {len(df):,}", logfile)

    # Age filter: 25+ as of REF_YEAR — dob may be datetime (SAS) or numeric
    dob = df[C.COL_DOB]
    if pd.api.types.is_datetime64_any_dtype(dob):
        df["age"] = C.REF_YEAR - dob.dt.year
    else:
        df["age"] = C.REF_YEAR - dob.astype(int)
    df = df[df["age"] >= 25].copy()
    log(f"  After age >= 25 filter: {len(df):,}", logfile)

    # Map education category to years
    df["educ_years"] = df[C.COL_EDUC].map(C.EDUC_TO_YEARS).astype("float32")

    # Debug subsample
    if debug_n is not None and len(df) > debug_n:
        df = df.iloc[:debug_n].copy()
        log(f"  Debug subsample: {len(df):,}", logfile)

    df = df.set_index(C.COL_PIK)
    log(f"  Final ICF: {len(df):,} PIKs", logfile)
    return df


def encode_demographics(icf: pd.DataFrame,
                        logfile: str | None = None,
                        ) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Create Model 1 feature columns from raw ICF demographics.

    Returns (df_with_features, feature_names, categorical_feature_indices).
    """
    df = icf.copy()

    # Binary encodings
    df["sex_binary"] = (df[C.COL_SEX] == "M").astype("float32")
    df["ethnicity_binary"] = (df[C.COL_ETHNICITY] == "H").astype("float32")

    # Race one-hot (keep all levels for tree-based model)
    for code in ["1", "2", "3", "4", "5", "7"]:
        df[f"race_{code}"] = (df[C.COL_RACE] == code).astype("float32")

    # POB ordinal (treated as categorical by HistGBR)
    df["pob_ordinal"] = df[C.COL_POB].map(C.POB_MAP).astype("float32")

    feature_names = (
        ["age", "sex_binary", "ethnicity_binary"]
        + [f"race_{c}" for c in ["1", "2", "3", "4", "5", "7"]]
        + ["pob_ordinal"]
    )

    # pob_ordinal is categorical for HistGBR
    cat_indices = [feature_names.index("pob_ordinal")]

    log(f"  Model 1 features: {len(feature_names)} columns", logfile)
    return df, feature_names, cat_indices


# ---------------------------------------------------------------------------
# RCF loading
# ---------------------------------------------------------------------------

def load_rcf(logfile: str | None = None) -> pd.DataFrame:
    """Load RCF residence file, filter to WINDOW years.

    Returns DataFrame with columns: [pik, address_year, state_county_tract].
    """
    log_section("Loading RCF", logfile)

    cols = [C.COL_PIK, C.COL_ADDRESS_YEAR, C.COL_COUNTY, C.COL_TRACT]

    with time_block("Read RCF", logfile):
        df = _read_file(C.RCF_PATH, columns=cols,
                        filters={C.COL_ADDRESS_YEAR: C.WINDOW},
                        logfile=logfile)

    log(f"  RCF rows in window: {len(df):,}", logfile)

    # Construct 11-char tract FIPS: county_live(5) + tract(6)
    df["state_county_tract"] = df[C.COL_COUNTY].astype(str) + df[C.COL_TRACT].astype(str)
    df["county"] = df[C.COL_COUNTY].astype(str)  # keep for thin-cell fallback
    df = df[[C.COL_PIK, C.COL_ADDRESS_YEAR, "state_county_tract", "county"]].copy()
    return df


# ---------------------------------------------------------------------------
# Tract-level aggregates (Model 2)
# ---------------------------------------------------------------------------

def compute_tract_aggregates(icf: pd.DataFrame,
                             rcf: pd.DataFrame,
                             train_mask: np.ndarray,
                             logfile: str | None = None) -> pd.DataFrame:
    """Compute leave-one-out tract-level features from training-set co-residents.

    Returns DataFrame indexed by PIK with ~12 neighborhood feature columns.
    """
    log_section("Computing tract aggregates", logfile)

    # ----- Step 1: training-set demographics joined to RCF -----
    train_piks = icf.index[train_mask]
    train_demo = icf.loc[train_piks, ["age", C.COL_SEX, C.COL_RACE,
                                       C.COL_ETHNICITY, "educ_years"]].copy()
    train_demo["is_ba_plus"] = (icf.loc[train_piks, C.COL_EDUC] == "4").astype("float32")
    train_demo["is_lths"] = (icf.loc[train_piks, C.COL_EDUC] == "1").astype("float32")
    train_demo["is_male"] = (train_demo[C.COL_SEX] == "M").astype("float32")
    train_demo["is_white"] = (train_demo[C.COL_RACE] == "1").astype("float32")
    train_demo["is_hispanic"] = (train_demo[C.COL_ETHNICITY] == "H").astype("float32")

    # Join training-set PIKs to their RCF records
    with time_block("Join train ICF to RCF", logfile):
        tr_rcf = rcf.merge(train_demo, left_on=C.COL_PIK, right_index=True, how="inner")
    log(f"  Training PIK-year-tract records: {len(tr_rcf):,}", logfile)

    # ----- Step 2: tract-year summary stats -----
    with time_block("Tract-year summaries", logfile):
        agg = tr_rcf.groupby(["state_county_tract", C.COL_ADDRESS_YEAR]).agg(
            sum_educ=("educ_years", "sum"),
            count=("educ_years", "count"),
            sum_ba=("is_ba_plus", "sum"),
            sum_lths=("is_lths", "sum"),
            sum_age=("age", "sum"),
            sum_male=("is_male", "sum"),
            sum_white=("is_white", "sum"),
            sum_hisp=("is_hispanic", "sum"),
        ).reset_index()
        agg["county"] = agg["state_county_tract"].str[:5]

    # ----- County-year fallback for thin cells -----
    with time_block("County-year fallback", logfile):
        county_agg = agg.groupby(["county", C.COL_ADDRESS_YEAR]).agg(
            cty_sum_educ=("sum_educ", "sum"),
            cty_count=("count", "sum"),
            cty_sum_ba=("sum_ba", "sum"),
            cty_sum_lths=("sum_lths", "sum"),
            cty_sum_age=("sum_age", "sum"),
            cty_sum_male=("sum_male", "sum"),
            cty_sum_white=("sum_white", "sum"),
            cty_sum_hisp=("sum_hisp", "sum"),
        ).reset_index()

        # For thin tracts, replace with county stats
        thin = agg["count"] < C.THIN_CELL_THRESHOLD
        n_thin = thin.sum()
        log(f"  Thin tract-years (<{C.THIN_CELL_THRESHOLD} obs): {n_thin:,} / {len(agg):,}", logfile)

        if n_thin > 0:
            agg = agg.merge(county_agg, on=["county", C.COL_ADDRESS_YEAR], how="left")
            for base, cty in [("sum_educ", "cty_sum_educ"), ("count", "cty_count"),
                              ("sum_ba", "cty_sum_ba"), ("sum_lths", "cty_sum_lths"),
                              ("sum_age", "cty_sum_age"), ("sum_male", "cty_sum_male"),
                              ("sum_white", "cty_sum_white"), ("sum_hisp", "cty_sum_hisp")]:
                agg.loc[thin, base] = agg.loc[thin, cty]
            agg = agg.drop(columns=[c for c in agg.columns if c.startswith("cty_")])

    # ----- Step 3: leave-one-out for training-set PIKs -----
    with time_block("Leave-one-out join", logfile):
        pik_ty = tr_rcf[[C.COL_PIK, C.COL_ADDRESS_YEAR, "state_county_tract",
                          "educ_years", "is_ba_plus", "is_lths",
                          "age", "is_male", "is_white", "is_hispanic"]].merge(
            agg, on=["state_county_tract", C.COL_ADDRESS_YEAR], how="left"
        )

        n = pik_ty["count"]
        # LOO means: subtract own value, divide by (count-1)
        pik_ty["loo_mean_educ"] = np.where(n > 1, (pik_ty["sum_educ"] - pik_ty["educ_years"]) / (n - 1), np.nan)
        pik_ty["loo_share_ba"] = np.where(n > 1, (pik_ty["sum_ba"] - pik_ty["is_ba_plus"]) / (n - 1), np.nan)
        pik_ty["loo_share_lths"] = np.where(n > 1, (pik_ty["sum_lths"] - pik_ty["is_lths"]) / (n - 1), np.nan)
        pik_ty["loo_n_obs"] = np.where(n > 1, n - 1, np.nan)
        pik_ty["loo_mean_age"] = np.where(n > 1, (pik_ty["sum_age"] - pik_ty["age"]) / (n - 1), np.nan)
        pik_ty["loo_frac_male"] = np.where(n > 1, (pik_ty["sum_male"] - pik_ty["is_male"]) / (n - 1), np.nan)
        pik_ty["loo_frac_white"] = np.where(n > 1, (pik_ty["sum_white"] - pik_ty["is_white"]) / (n - 1), np.nan)
        pik_ty["loo_frac_hisp"] = np.where(n > 1, (pik_ty["sum_hisp"] - pik_ty["is_hispanic"]) / (n - 1), np.nan)

    # ----- Step 4: non-LOO tract stats for predict-set PIKs -----
    agg["mean_educ"] = agg["sum_educ"] / agg["count"]
    agg["share_ba"] = agg["sum_ba"] / agg["count"]
    agg["share_lths"] = agg["sum_lths"] / agg["count"]
    agg["n_obs"] = agg["count"]
    agg["mean_age"] = agg["sum_age"] / agg["count"]
    agg["frac_male"] = agg["sum_male"] / agg["count"]
    agg["frac_white"] = agg["sum_white"] / agg["count"]
    agg["frac_hisp"] = agg["sum_hisp"] / agg["count"]

    predict_piks = icf.index[~train_mask]
    pred_rcf = rcf[rcf[C.COL_PIK].isin(predict_piks)]
    pred_ty = pred_rcf.merge(
        agg[["state_county_tract", C.COL_ADDRESS_YEAR,
             "mean_educ", "share_ba", "share_lths", "n_obs",
             "mean_age", "frac_male", "frac_white", "frac_hisp"]],
        on=["state_county_tract", C.COL_ADDRESS_YEAR], how="left"
    )
    # Rename to match LOO columns
    pred_ty = pred_ty.rename(columns={
        "mean_educ": "loo_mean_educ", "share_ba": "loo_share_ba",
        "share_lths": "loo_share_lths", "n_obs": "loo_n_obs",
        "mean_age": "loo_mean_age", "frac_male": "loo_frac_male",
        "frac_white": "loo_frac_white", "frac_hisp": "loo_frac_hisp",
    })

    # ----- Step 5: aggregate over years per PIK -----
    loo_cols = ["loo_mean_educ", "loo_share_ba", "loo_share_lths", "loo_n_obs",
                "loo_mean_age", "loo_frac_male", "loo_frac_white", "loo_frac_hisp"]

    combined = pd.concat([
        pik_ty[[C.COL_PIK, C.COL_ADDRESS_YEAR, "state_county_tract"] + loo_cols],
        pred_ty[[C.COL_PIK, C.COL_ADDRESS_YEAR, "state_county_tract"] + loo_cols],
    ], ignore_index=True)

    with time_block("Aggregate over years", logfile):
        result = _aggregate_over_years(combined, loo_cols, logfile)

    log(f"  Tract features computed for {len(result):,} PIKs", logfile)
    return result


def _aggregate_over_years(df: pd.DataFrame, loo_cols: list[str],
                          logfile: str | None = None) -> pd.DataFrame:
    """Per-PIK aggregation of annual tract features: mean, ref-year, slope, counts."""
    # Window-mean of each tract feature
    means = df.groupby(C.COL_PIK)[loo_cols].mean()
    means.columns = [f"tract_{c.replace('loo_', '')}_mean" for c in loo_cols]

    # Reference-year value
    ref = df[df[C.COL_ADDRESS_YEAR] == C.REF_YEAR].set_index(C.COL_PIK)[loo_cols]
    ref = ref[~ref.index.duplicated(keep="first")]  # one record per PIK
    ref.columns = [f"tract_{c.replace('loo_', '')}_refyr" for c in loo_cols[:1]]
    # Only keep ref-year mean_educ (most important ref-year feature)
    ref = ref.iloc[:, :1]

    # Slope of tract education over time (linear trend)
    def _slope(group):
        if len(group) < 2:
            return np.nan
        years = group[C.COL_ADDRESS_YEAR].values.astype(float)
        vals = group["loo_mean_educ"].values
        mask = ~np.isnan(vals)
        if mask.sum() < 2:
            return np.nan
        return np.polyfit(years[mask], vals[mask], 1)[0]

    slopes = df.groupby(C.COL_PIK).apply(_slope).rename("tract_educ_slope")

    # Count distinct tracts and years observed
    counts = df.groupby(C.COL_PIK).agg(
        tract_distinct_tracts=("state_county_tract", "nunique"),
        tract_years_observed=(C.COL_ADDRESS_YEAR, "nunique"),
    )

    result = means.join(ref, how="left").join(slopes, how="left").join(counts, how="left")
    return result.astype("float32")


# ---------------------------------------------------------------------------
# EHF / ECF loading (Model 3)
# ---------------------------------------------------------------------------

def load_ehf_multi_state(states: list[str],
                         logfile: str | None = None) -> pd.DataFrame:
    """Load EHF across states, filter to WINDOW years. Returns [pik, year, earn_ann, sein, state]."""
    log_section("Loading EHF", logfile)
    cols = [C.COL_PIK, C.COL_YEAR, C.COL_EARN_ANN, C.COL_SEIN, C.COL_STATE]
    chunks = []
    for st in states:
        path = C.EHF_PATH_TEMPLATE.format(state=st)
        with time_block(f"EHF {st}", logfile):
            df = _read_file(path, columns=cols,
                            filters={C.COL_YEAR: C.WINDOW}, logfile=logfile)
            if len(df) > 0:
                chunks.append(df)
                log(f"  State {st}: {len(df):,} rows", logfile)
    if not chunks:
        return pd.DataFrame(columns=cols)
    result = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    log(f"  Total EHF rows: {len(result):,}", logfile)
    return result


def identify_dominant_job(ehf: pd.DataFrame,
                          logfile: str | None = None) -> pd.DataFrame:
    """For each PIK x year, find the SEIN with highest annual earnings."""
    with time_block("Identify dominant jobs", logfile):
        idx = ehf.groupby([C.COL_PIK, C.COL_YEAR])[C.COL_EARN_ANN].idxmax()
        result = ehf.loc[idx, [C.COL_PIK, C.COL_YEAR, C.COL_SEIN, C.COL_STATE]].copy()
    log(f"  Dominant jobs: {len(result):,}", logfile)
    return result


def load_ecf_naics(states: list[str],
                   logfile: str | None = None) -> pd.DataFrame:
    """Load ECF_SEIN NAICS codes across states. Returns [sein, year, state, naics_2d]."""
    log_section("Loading ECF NAICS", logfile)
    cols = [C.COL_SEIN, C.COL_YEAR, C.COL_QUARTER, C.COL_NAICS, C.COL_STATE]
    chunks = []
    for st in states:
        path = C.ECF_SEIN_PATH_TEMPLATE.format(state=st)
        with time_block(f"ECF {st}", logfile):
            df = _read_file(path, columns=cols,
                            filters={C.COL_YEAR: C.WINDOW}, logfile=logfile)
            # Keep Q1 only (or closest available quarter per year)
            if C.COL_QUARTER in df.columns:
                df = df[df[C.COL_QUARTER] == C.REF_QUARTER]
            if len(df) > 0:
                chunks.append(df)
    if not chunks:
        return pd.DataFrame(columns=[C.COL_SEIN, C.COL_YEAR, C.COL_STATE, "naics_2d"])
    result = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # Extract 2-digit NAICS sector
    result["naics_2d"] = result[C.COL_NAICS].astype(str).str[:2]
    result = result[[C.COL_SEIN, C.COL_YEAR, C.COL_STATE, "naics_2d"]]
    log(f"  Total ECF NAICS rows: {len(result):,}", logfile)
    return result


def compute_naics_features(dominant_jobs: pd.DataFrame,
                           ecf_naics: pd.DataFrame,
                           logfile: str | None = None) -> pd.DataFrame:
    """Per-PIK NAICS features from dominant job history.

    Returns DataFrame indexed by PIK with columns:
      naics_modal, naics_refyr, naics_distinct, naics_years_obs
    """
    log_section("Computing NAICS features", logfile)

    with time_block("Merge jobs to ECF", logfile):
        merged = dominant_jobs.merge(
            ecf_naics, on=[C.COL_SEIN, C.COL_YEAR, C.COL_STATE], how="left"
        )

    # Modal NAICS over window
    def _mode(s):
        counts = s.dropna().value_counts()
        return counts.index[0] if len(counts) > 0 else np.nan

    with time_block("Per-PIK NAICS aggregation", logfile):
        modal = merged.groupby(C.COL_PIK)["naics_2d"].agg(_mode).rename("naics_modal")

        # NAICS at reference year
        ref = merged[merged[C.COL_YEAR] == C.REF_YEAR].drop_duplicates(C.COL_PIK)
        ref_naics = ref.set_index(C.COL_PIK)["naics_2d"].rename("naics_refyr")

        # Count distinct sectors and years with data
        counts = merged.groupby(C.COL_PIK).agg(
            naics_distinct=("naics_2d", "nunique"),
            naics_years_obs=(C.COL_YEAR, "nunique"),
        )

    result = modal.to_frame().join(ref_naics, how="left").join(counts, how="left")

    # Encode NAICS as ordinal integers for HistGBR categorical treatment
    all_codes = {c: i for i, c in enumerate(C.NAICS_2D_CODES)}
    result["naics_modal"] = result["naics_modal"].map(all_codes).astype("float32")
    result["naics_refyr"] = result["naics_refyr"].map(all_codes).astype("float32")
    result["naics_distinct"] = result["naics_distinct"].astype("float32")
    result["naics_years_obs"] = result["naics_years_obs"].astype("float32")

    log(f"  NAICS features for {len(result):,} PIKs", logfile)
    return result


def compute_tract_naics_distribution(rcf: pd.DataFrame,
                                     naics_features: pd.DataFrame,
                                     train_piks: pd.Index,
                                     logfile: str | None = None) -> pd.DataFrame:
    """Per-PIK tract-level NAICS sector shares from training-set co-residents.

    Returns DataFrame indexed by PIK with columns tract_naics_share_{sector}.
    """
    log_section("Computing tract NAICS distribution", logfile)

    # Get training-set PIKs with known NAICS
    train_naics = naics_features.loc[naics_features.index.isin(train_piks), "naics_modal"].dropna()

    with time_block("Join NAICS to RCF", logfile):
        # Join: PIK -> naics_modal, then PIK -> tract via RCF
        rcf_naics = rcf[rcf[C.COL_PIK].isin(train_naics.index)].merge(
            train_naics.reset_index(), on=C.COL_PIK, how="inner"
        )

    # Tract-year NAICS distribution
    with time_block("Tract NAICS distribution", logfile):
        n_sectors = len(C.NAICS_2D_CODES)
        # Count per tract-year-sector
        tract_yr = rcf_naics.groupby(["state_county_tract", C.COL_ADDRESS_YEAR, "naics_modal"]).size()
        tract_yr = tract_yr.unstack(fill_value=0)

        # Normalize to shares
        row_sums = tract_yr.sum(axis=1)
        tract_yr_shares = tract_yr.div(row_sums, axis=0)
        tract_yr_shares.columns = [f"tract_naics_share_{int(c)}" if not np.isnan(c)
                                   else "tract_naics_share_nan"
                                   for c in tract_yr_shares.columns]
        tract_yr_shares = tract_yr_shares.reset_index()

    # Join to all PIKs via RCF and average over years
    with time_block("Join tract NAICS to PIKs", logfile):
        pik_naics = rcf.merge(
            tract_yr_shares,
            on=["state_county_tract", C.COL_ADDRESS_YEAR], how="left"
        )
        share_cols = [c for c in pik_naics.columns if c.startswith("tract_naics_share_")]
        result = pik_naics.groupby(C.COL_PIK)[share_cols].mean()

    result = result.astype("float32")
    log(f"  Tract NAICS features for {len(result):,} PIKs, {len(share_cols)} sectors", logfile)
    return result


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(icf: pd.DataFrame,
                         feature_names_m1: list[str],
                         cat_indices_m1: list[int],
                         tract_agg: pd.DataFrame | None = None,
                         naics_feats: pd.DataFrame | None = None,
                         tract_naics: pd.DataFrame | None = None,
                         model_level: int = 1,
                         logfile: str | None = None,
                         ) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Assemble final feature matrix for the given model level.

    Returns (X, y, feature_names, categorical_indices).
    X is float32; y is float32 (educ_years).
    """
    log_section(f"Building feature matrix (Model {model_level})", logfile)

    feature_names = list(feature_names_m1)
    cat_indices = list(cat_indices_m1)
    parts = [icf[feature_names_m1]]

    if model_level >= 2 and tract_agg is not None:
        tract_cols = [c for c in tract_agg.columns]
        joined = tract_agg.reindex(icf.index)
        parts.append(joined)
        base = len(feature_names)
        feature_names.extend(tract_cols)

    if model_level >= 3:
        if naics_feats is not None:
            naics_cols = list(naics_feats.columns)
            joined_n = naics_feats.reindex(icf.index)
            parts.append(joined_n)
            base = len(feature_names)
            # naics_modal and naics_refyr are categorical
            for col in ["naics_modal", "naics_refyr"]:
                if col in naics_cols:
                    cat_indices.append(base + naics_cols.index(col))
            feature_names.extend(naics_cols)

        if tract_naics is not None:
            tn_cols = list(tract_naics.columns)
            joined_tn = tract_naics.reindex(icf.index)
            parts.append(joined_tn)
            feature_names.extend(tn_cols)

    X_df = pd.concat(parts, axis=1)
    X = X_df.values.astype(np.float32)
    y = icf["educ_years"].values.astype(np.float32)

    log(f"  X shape: {X.shape}, features: {len(feature_names)}", logfile)
    log(f"  Categorical indices: {cat_indices}", logfile)
    log(f"  NaN rate: {np.isnan(X).mean():.4f}", logfile)
    return X, y, feature_names, cat_indices


# ---------------------------------------------------------------------------
# Disk persistence for staged execution
# ---------------------------------------------------------------------------

def save_prepared_data(output_dir: str,
                       X: np.ndarray, y: np.ndarray,
                       train_mask: np.ndarray, predict_mask: np.ndarray,
                       feature_names: list[str], cat_indices: list[int],
                       piks: np.ndarray,
                       model_level: int,
                       logfile: str | None = None) -> None:
    """Save prepared arrays to disk so train step can load without re-running prep."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"prepared_model{model_level}.npz"
    np.savez(
        path,
        X=X, y=y,
        train_mask=train_mask, predict_mask=predict_mask,
        piks=piks,
        feature_names=np.array(feature_names),
        cat_indices=np.array(cat_indices),
    )
    log(f"  Saved prepared data: {path} ({path.stat().st_size / 1e6:.0f} MB)", logfile)


def load_prepared_data(output_dir: str,
                       model_level: int,
                       logfile: str | None = None,
                       ) -> dict:
    """Load prepared arrays from disk."""
    path = Path(output_dir) / f"prepared_model{model_level}.npz"
    log(f"  Loading prepared data: {path}", logfile)
    data = np.load(path, allow_pickle=True)
    return {
        "X": data["X"],
        "y": data["y"],
        "train_mask": data["train_mask"],
        "predict_mask": data["predict_mask"],
        "piks": data["piks"],
        "feature_names": list(data["feature_names"]),
        "cat_indices": list(data["cat_indices"]),
    }
