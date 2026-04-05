"""Education imputation configuration: paths, column names, hyperparameters.

Every path and file-naming constant must be verified on first RDC login.
Search for '>>> VERIFY' to find all items that need checking.
"""

# ---------------------------------------------------------------------------
# Python environment activation  (>>> VERIFY ON FIRST LOGIN <<<)
# ---------------------------------------------------------------------------
# On compute nodes, activate Anaconda BEFORE running any Python:
#   source /apps/anaconda/bin/activate py3cf
# Use py3cf (conda-forge, most complete). Check alternatives:
#   conda env list
# Verify: python -c "from sklearn.ensemble import HistGradientBoostingRegressor"

# ---------------------------------------------------------------------------
# Paths  (>>> VERIFY ON FIRST LOGIN <<<)
# ---------------------------------------------------------------------------
# LEHD infrastructure files — actual location depends on project approval.
# Typical pattern: /data/lehd/{vintage}/ or /projects/{id}/transfer/
# Parquet files may be under a parquet/ subdirectory.

DATA_ROOT = "/data/lehd/current"                          # >>> VERIFY <<<

ICF_US_PATH = f"{DATA_ROOT}/icf_us.sas7bdat"              # >>> VERIFY <<<
RCF_PATH = f"{DATA_ROOT}/icf_us_residence_rcf.sas7bdat"   # >>> VERIFY <<<

# Per-state files: {state} replaced with 2-digit FIPS code (e.g. "24" for MD)
EHF_PATH_TEMPLATE = DATA_ROOT + "/ehf_{state}.sas7bdat"   # >>> VERIFY <<<
ECF_SEIN_PATH_TEMPLATE = DATA_ROOT + "/ecf_{state}_sein.sas7bdat"  # >>> VERIFY <<<

# Project output directory
PROJECT_DIR = "/projects/{project_id}"                     # >>> VERIFY <<<
OUTPUT_DIR = PROJECT_DIR + "/data/educ_impute"

# Set True if parquet/ directory available (much faster reads)
USE_PARQUET = False                                        # >>> VERIFY <<<
PARQUET_ICF_PATH = f"{DATA_ROOT}/parquet/icf_us"           # >>> VERIFY <<<
PARQUET_RCF_PATH = f"{DATA_ROOT}/parquet/icf_us_residence_rcf"  # >>> VERIFY <<<

# ---------------------------------------------------------------------------
# State lists
# ---------------------------------------------------------------------------
# Full list of state FIPS codes with LEHD coverage.
# Not all states may be available on your project — check approval docs.
ALL_STATES = [                                             # >>> VERIFY <<<
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56",
]

# Small subset for interactive debugging
DEBUG_STATES = ["24"]  # Maryland

# ---------------------------------------------------------------------------
# Reference period
# ---------------------------------------------------------------------------
REF_YEAR = 2014
REF_QUARTER = 1
WINDOW = (2012, 2016)  # residence/earnings window around REF_YEAR

# ---------------------------------------------------------------------------
# Column names  (exact LEHD variable names — >>> VERIFY case on first login <<<)
# SAS files may return UPPERCASE; if so, lowercase them after read.
# ---------------------------------------------------------------------------

# ICF_US
COL_PIK = "pik"
COL_DOB = "dob"
COL_SEX = "sex"
COL_POB = "pob"
COL_RACE = "race"
COL_ETHNICITY = "ethnicity"
COL_EDUC = "educ_c"
COL_EDUC_IMPUTED = "educ_c_imputed"

# RCF (2012+)
COL_ADDRESS_YEAR = "address_year"
COL_COUNTY = "county_live"
COL_TRACT = "tract"

# EHF
COL_YEAR = "year"
COL_EARN_ANN = "earn_ann"
COL_SEIN = "sein"
COL_STATE = "state"

# ECF_SEIN
COL_QUARTER = "quarter"
COL_NAICS = "mode_naics2017fnl_emp"  # >>> VERIFY vintage (2012/2017/2022) <<<

# ---------------------------------------------------------------------------
# Education mapping
# ---------------------------------------------------------------------------
EDUC_TO_YEARS = {"1": 10, "2": 12, "3": 14, "4": 16}
EDUC_LABELS = {1: "<HS", 2: "HS", 3: "SomeCol", 4: "BA+"}

# Bin edges for discretizing continuous predictions back to 1-4
# <11 -> 1,  11-13 -> 2,  13-15 -> 3,  >=15 -> 4
BIN_EDGES = [0, 11, 13, 15, 20]

# ---------------------------------------------------------------------------
# Place-of-birth ordinal encoding (27 categories -> 0..26)
# ---------------------------------------------------------------------------
POB_MAP = {
    "1": 0,  "2": 1,  "3": 2,  "4": 3,  "5": 4,
    "6": 5,  "7": 6,  "8": 7,  "9": 8,
    "A": 9,  "B": 10, "C": 11, "D": 12, "E": 13,
    "F": 14, "G": 15, "H": 16, "I": 17, "J": 18,
    "K": 19, "L": 20, "M": 21, "N": 22, "O": 23,
    "P": 24, "Q": 25, "R": 26, "S": 27, "T": 28,
    "U": 29, "V": 30, "W": 31, "X": 32, "Y": 33, "Z": 34,
}

# ---------------------------------------------------------------------------
# 2-digit NAICS sectors (for tract NAICS distribution features)
# ---------------------------------------------------------------------------
NAICS_2D_CODES = [
    "11", "21", "22", "23", "31", "32", "33",
    "42", "44", "45", "48", "49", "51", "52", "53",
    "54", "55", "56", "61", "62", "71", "72", "81", "92",
]

# ---------------------------------------------------------------------------
# Model hyperparameters (XGBoost)
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=100,
    max_bin=255,
    tree_method="hist",
    random_state=42,
    n_jobs=1,
)
EARLY_STOPPING_ROUNDS = 20
VALIDATION_FRACTION = 0.1

N_FOLDS = 5

# Thin-cell threshold: tracts with fewer observed-education residents
# fall back to county-level aggregates
THIN_CELL_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Debug settings
# ---------------------------------------------------------------------------
DEBUG_N_PIKS = 100_000
