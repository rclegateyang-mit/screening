# LEHD SAS Pipeline Documentation

These three SAS scripts were written for a mergers project and construct establishment- and worker-level panels from Census LEHD microdata. This document explains each step's logic, input/output schemas, and key cleaning decisions.

> **Note:** These scripts are from a different project ("Aggregate Implications of Mergers for the US Labor Market") and will need to be adapted for the screening project. They are useful as references for how LEHD/LBD/SSL data are structured and linked.

---

## Step 1: Creating LEHD-LBD Base File

**Purpose:** Build an annual establishment-level panel from quarterly LEHD Employer Characteristics Files (ECF), link to LBD firm identifiers, and merge in commuting zone geography.

### Input Datasets

```
┌─────────────────────────────────────────────────────┐
│  ECF_<state>_SEIN_T26  (by state, quarterly)        │
│  ─────────────────────────────────────────────       │
│  SEIN          │ State Employer ID                   │
│  YEAR          │ Calendar year                       │
│  QUARTER       │ 1-4                                 │
│  FIRMID        │ Census firm identifier (may be "")  │
│  FAS_EIN       │ Federal EIN (14-char: 5-digit       │
│                │   prefix + 9-digit EIN)             │
│  FAS_EIN_FLAG  │ Quality flag for EIN linkage        │
│  LBD_MATCH     │ Indicator: matched to LBD           │
│  MULTI_UNIT_LBD│ Multi-unit indicator from LBD       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  ECF_<state>_SEINUNIT  (by state, quarterly)        │
│  ─────────────────────────────────────────────       │
│  SEIN          │ State Employer ID                   │
│  SEINUNIT      │ Establishment unit within SEIN      │
│  YEAR          │ Calendar year                       │
│  QUARTER       │ 1-4                                 │
│  IN_202        │ In ES-202 (UI) reports              │
│  IN_UI         │ In UI wage records                  │
│  ES_STATE      │ FIPS state code                     │
│  LEG_COUNTY    │ FIPS county code                    │
│  LEG_CBSA      │ CBSA (metro area) code              │
│  LEG_CBSA_MEMI │ Metro/micro indicator               │
│  NAICS2017FNL  │ 6-digit NAICS (2017 vintage)        │
│  BEST_EMP1-3   │ Employment (month 1, 2, 3 of qtr)  │
│  BEST_WAGES    │ Total quarterly wages               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  LBD<year>_V201900  (revised LBD, by year)          │
│  ─────────────────────────────────────────────       │
│  EIN           │ Federal EIN (9-digit)               │
│  FIRMID        │ Census firm identifier              │
│  LBDFID        │ LBD firm identifier                 │
│  YEAR          │ Calendar year                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  county_cz_all_crosswalk                            │
│  ─────────────────────────────────────────────       │
│  FIPSCOU       │ FIPS county code                    │
│  CZ_ID2000     │ Commuting zone (2000 definition)    │
│  CZ_ID1990     │ Commuting zone (1990 definition)    │
│  CZ_ID1980     │ Commuting zone (1980 definition)    │
│  COUNTY_NAME   │ County name                         │
│  MSA_2003_NAME │ MSA name (2003)                     │
│  COUNTY_POP_2003│ County population                  │
│  CZ_POP_2000   │ CZ population                      │
└─────────────────────────────────────────────────────┘
```

### Processing Steps

```
 ECF_SEIN_T26 ──┐                                   LBD (by year)
 (quarterly,    │                                        │
  28 states)    │                                        │
                ▼                                        │
         ┌──────────────┐                                │
         │ 1. SQL JOIN   │ on SEIN × YEAR × QUARTER      │
         │    T26 ⟕     │                                │
         │    SEINUNIT   │                                │
         └──────┬───────┘                                │
                │                                        │
                ▼                                        │
         ┌──────────────┐                                │
         │ 2. Filter to  │ Keep years 2000-2018           │
         │    study      │ Fill missing FIRMID within     │
         │    period     │ SEIN-SEINUNIT-year using       │
         │              │ non-missing quarters            │
         └──────┬───────┘                                │
                │                                        │
                ▼                                        │
         ┌──────────────┐                                │
         │ 3. Annualize  │ Employment = BEST_EMP3 (March) │
         │              │ Payroll = Σ BEST_WAGES (4 qtrs) │
         │              │ Keep Q1 record; if Q1 missing,  │
         │              │ keep first available quarter     │
         └──────┬───────┘                                │
                │                                        │
                ▼                                        │
         ┌──────────────┐                                │
         │ 4. Append     │ Stack 28 state files            │
         │    states     │                                │
         └──────┬───────┘                                │
                │                                        │
                ▼                                        ▼
         ┌──────────────────────────────────────────────────┐
         │ 5. Improve FIRMID linkage                        │
         │    EIN (9-digit) from LEHD → match to LBD by    │
         │    EIN × YEAR to get FIRMID_LBD and LBDFID      │
         │    Use LBD FIRMID when ECF FIRMID is missing     │
         │    AND EIN prefix = "EINUS"                      │
         └──────────────────┬───────────────────────────────┘
                            │
                            ▼
         ┌──────────────────────────────────────────────────┐
         │ 6. Merge CZ geography                            │
         │    LEFT JOIN on FIPSCOU → county_cz_crosswalk    │
         └──────────────────┬───────────────────────────────┘
                            │
                            ▼
                    OUTPUT DATASET
```

### Output Dataset

```
┌──────────────────────────────────────────────────────────────┐
│  lehd_lbd_allstates_CZ                                       │
│  Unit of observation: SEIN × SEINUNIT × YEAR                │
│  Coverage: 28 states, 2001-2018                              │
│  ────────────────────────────────────────────────────         │
│  SEIN           │ State employer ID                          │
│  SEINUNIT       │ Establishment unit within SEIN             │
│  LEHDNUM        │ Concatenation of SEIN + SEINUNIT           │
│  YEAR           │ Calendar year                              │
│  FIRMID         │ Census firm ID (ECF or LBD-filled)         │
│  LBDFID         │ LBD firm identifier                        │
│  FAS_EIN        │ Federal EIN (14 characters)                │
│  FAS_EIN_FLAG   │ EIN quality flag                           │
│  LBD_MATCH      │ Matched to LBD indicator                   │
│  MULTI_UNIT_LBD │ Multi-unit indicator                       │
│  IN_202         │ In ES-202 reports                          │
│  IN_UI          │ In UI wage records                         │
│  EMP_LEHD       │ March employment (BEST_EMP3)               │
│  PAY_LEHD       │ Annual payroll (Σ quarterly wages)         │
│  FIPSST         │ FIPS state code                            │
│  FIPSCOU        │ FIPS county code                           │
│  CBSA           │ CBSA metro area code                       │
│  CBSA_F         │ Metro/micro indicator                      │
│  FKNAICS2017    │ 6-digit NAICS (2017)                       │
│  CZ_ID2000      │ Commuting zone (2000)                      │
│  CZ_ID1990      │ Commuting zone (1990)                      │
│  CZ_ID1980      │ Commuting zone (1980)                      │
│  COUNTY_NAME    │ County name                                │
│  MSA_2003_NAME  │ MSA name                                   │
│  COUNTY_POP_2003│ County population                          │
│  CZ_POP_2000    │ CZ population                              │
└──────────────────────────────────────────────────────────────┘
```

### Key Cleaning Decisions

| Decision | Rationale |
|----------|-----------|
| **March employment** (BEST_EMP3 from Q1) as the annual employment measure | Standard in LBD/LEHD literature; March is the BLS reference month |
| **Annual payroll** summed across 4 quarters | Gives comparable annual measure |
| **Q1 preferred** for annual snapshot; fallback to later quarter if Q1 missing | Ensures maximum coverage while maintaining consistency |
| **FIRMID gap-filling** within SEIN-SEINUNIT-year across quarters | FIRMID may be missing in some quarters but present in others |
| **LBD-based FIRMID backfill** when ECF FIRMID missing AND EIN prefix = "EINUS" | Improves firm-level linkage; "EINUS" prefix indicates a domestic US EIN |
| **NODUPKEY** on SEIN × SEINUNIT × YEAR after annualization | Ensures one row per establishment-year |
| **Filter years 2000-2018** (2000 kept for ownership change detection) | 2000 is a lookback year, not in the analysis sample |

---

## Step 4: Full Job History Construction

**Purpose:** For workers ever employed at establishments involved in "large" mergers (from a prior step 3), extract their complete job history from the JHF (Job History Files), then merge in geography and industry.

> **Note for screening project:** This step is merger-specific (filters to workers at merger-affected firms). For our purposes, the relevant parts are (a) how JHF data is structured and (b) how worker job histories are constructed and linked to establishment characteristics.

### Input Datasets

```
┌──────────────────────────────────────────────────────┐
│  JHF_<state>  (Job History Files, by state)          │
│  ─────────────────────────────────────────────       │
│  PIK           │ Person identifier (worker ID)       │
│  SEIN          │ State employer ID                   │
│  SEINUNIT1     │ Establishment unit (first in spell) │
│  FID           │ Firm identifier                     │
│  SPELL_U2W     │ Spell identifier (U→W transition)   │
│  FIRST_ACC     │ Accession date (start of job)       │
│  LAST_SEP      │ Separation date (end of job)        │
│  E1-E21        │ Quarterly earnings indicators       │
│                │ (number varies by state; MD has E21) │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  <state>_seinlist  (from Step 3)                     │
│  ─────────────────────────────────────────────       │
│  SEIN          │ SEINs at merger-affected estabs     │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  ECF_<state>_SEINUNIT  (same as Step 1)              │
│  ─────────────────────────────────────────────       │
│  SEIN, SEINUNIT, LEG_CBSA, LEG_COUNTY,              │
│  NAICS2017FNL                                        │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  all_mergers_list  (from LBD path)                   │
│  ─────────────────────────────────────────────       │
│  SEIN, SEINUNIT, CZ_ID2000                           │
└──────────────────────────────────────────────────────┘
```

### Processing Steps

```
  JHF_<state>           <state>_seinlist
  (all workers)         (merger SEINs)
       │                      │
       ▼                      ▼
 ┌────────────────────────────────────────┐
 │ 1. Inner join JHF × seinlist on SEIN  │
 │    → PIKs who ever worked at a merger │
 │      firm (deduplicate to PIK list)   │
 └──────────────┬─────────────────────────┘
                │
                ▼
 ┌────────────────────────────────────────┐
 │ 2. Inner join JHF × PIK list on PIK   │
 │    → FULL job history of these workers│
 │      (all employers, not just merger  │
 │       firms)                          │
 │    Also compute:                      │
 │      LAST_SEP_MAX = max(LAST_SEP)     │
 │      FIRST_ACC_MIN = min(FIRST_ACC)   │
 │    Grouped by PIK × FID × SPELL_U2W  │
 └──────────────┬─────────────────────────┘
                │
                ▼
 ┌────────────────────────────────────────┐
 │ 3. Collapse to one row per spell      │
 │    PLANT = SEIN || "_" || SEINUNIT1   │
 │    Keep first row per SPELL_U2W       │
 │    Track PLANT_FIRST, PLANT_LAST      │
 │    Add STATE_SOURCE indicator         │
 └──────────────┬─────────────────────────┘
                │
                ▼
 ┌────────────────────────────────────────┐
 │ 4. Merge establishment geography      │
 │    ECF (deduplicated to SEIN×SEINUNIT)│
 │    → LEG_CBSA, LEG_COUNTY,           │
 │      NAICS2017FNL, CZ_ID2000         │
 └──────────────┬─────────────────────────┘
                │
                ▼
 ┌────────────────────────────────────────┐
 │ 5. Append across 28 states            │
 └──────────────┬─────────────────────────┘
                │
                ▼
 ┌────────────────────────────────────────┐
 │ 6. Sort by PIK, create JOB_SEQ        │
 │    (sequential job number per worker) │
 └──────────────┬─────────────────────────┘
                │
                ▼
            OUTPUT
```

### Output Dataset

```
┌──────────────────────────────────────────────────────────────┐
│  step4_full_job_history_all2                                 │
│  Unit of observation: PIK × job spell                        │
│  ────────────────────────────────────────────────────         │
│  PIK            │ Worker person identifier                   │
│  STATE_SOURCE   │ State where this spell was recorded        │
│  PLANT_FIRST    │ Establishment at start of spell            │
│                 │ (SEIN_SEINUNIT)                            │
│  PLANT_LAST     │ Establishment at end of spell              │
│  FIRST_ACC_MIN  │ Earliest accession date across all         │
│                 │ records in spell group                     │
│  LAST_SEP_MAX   │ Latest separation date across all          │
│                 │ records in spell group                     │
│  JOB_SEQ        │ Sequential job number for this worker      │
│  SEIN           │ State employer ID (from ECF merge)         │
│  SEINUNIT       │ Establishment unit (from ECF merge)        │
│  LEG_CBSA       │ CBSA code (from ECF)                       │
│  LEG_COUNTY     │ FIPS county (from ECF)                     │
│  NAICS2017FNL   │ NAICS industry (from ECF)                  │
│  CZ_ID2000      │ Commuting zone (from mergers list merge)   │
└──────────────────────────────────────────────────────────────┘
```

### Key Cleaning Decisions

| Decision | Rationale |
|----------|-----------|
| **Worker selection via merger SEINs** | Merger-project-specific; our project would skip this filter |
| **Full job history** retrieved for selected workers (all employers) | Allows tracking transitions across all firms, not just merger firms |
| **Collapse to one row per spell** (first row of each SPELL_U2W) | JHF may have multiple rows per spell due to establishment transfers within the same firm; PLANT_FIRST/PLANT_LAST track these |
| **FIRST_ACC_MIN / LAST_SEP_MAX** computed at PIK × FID × SPELL_U2W level | Gives the broadest window for each employment spell |
| **ECF geography deduplicated to SEIN × SEINUNIT** (dropping time variation) | Location/industry taken as time-invariant at establishment level |
| **Inner join on geography** (not left join) | Drops spells at establishments not in the ECF or missing geography — potential data loss |
| **JOB_SEQ** created by sorting on PIK, FIRST_ACC_MIN, LAST_SEP_MAX | Produces a chronological sequence of jobs per worker |

---

## Step 7: Establishment-Level Revenues

**Purpose:** Construct establishment-level revenues by pulling EIN- or firm-level revenue from Census administrative data (SSL/BR files) and computing state-level employment shares for later revenue apportionment to establishments.

### Input Datasets

```
┌──────────────────────────────────────────────────────┐
│  SSL<year>SU  (Statistics of US Business, by year)   │
│  Years 2001-2016                                     │
│  ─────────────────────────────────────────────       │
│  EIN            │ Federal EIN                        │
│  ADMNAICS       │ Administrative NAICS               │
│  ACSR1-ACSR4    │ Revenue components (2001 only)     │
│  ACSR1F-ACSR4F  │ IRS form flags for revenue         │
│  BESTADMIN_RCPT_<year> │ Best admin receipts          │
│                 │       (2002-2016)                   │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  BRFIRM_REV<year>_V201900  (BR firm revenue files)   │
│  Years 2017-2018                                     │
│  ─────────────────────────────────────────────       │
│  FIRMID         │ Census firm identifier             │
│  NREV2          │ Net revenue (measure 2)            │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  LBD<year>_V201900  (revised LBD, by year)           │
│  ─────────────────────────────────────────────       │
│  EIN            │ Federal EIN                        │
│  FIRMID         │ Census firm ID                     │
│  BDS_ST         │ State FIPS code                    │
│  EMP            │ Establishment employment           │
└──────────────────────────────────────────────────────┘
```

### Processing Steps

```
           SSL/BR revenue files                     LBD employment files
           (by year)                                (by year)
                │                                        │
                ▼                                        │
  ┌──────────────────────────────┐                       │
  │  Part A: Revenue extraction  │                       │
  │                              │                       │
  │  2001: ACSR1-4 + form flags  │                       │
  │    → sector-specific rules   │                       │
  │    (Haltiwanger et al. 2020) │                       │
  │                              │                       │
  │  2002-2016: BESTADMIN_RCPT   │                       │
  │    → direct use              │                       │
  │                              │                       │
  │  2017-2018: NREV2 from       │                       │
  │    BRFIRM files (firmid lvl) │                       │
  └────────────┬─────────────────┘                       │
               │                                         │
               ▼                                         ▼
  ┌─────────────────────┐             ┌──────────────────────────────┐
  │ Revenue by EIN×year │             │ Part B: Employment weights   │
  │ (2001-2016)         │             │                              │
  │                     │             │ 2001-2016: by EIN × state    │
  │ Revenue by          │             │   emp_upper_st / emp_upper   │
  │ firmid×year         │             │                              │
  │ (2017-2018)         │             │ 2017-2018: by firmid × state │
  └─────────┬───────────┘             │   emp_upper_st / emp_upper   │
            │                         └──────────────┬───────────────┘
            │                                        │
            ▼                                        ▼
    ┌───────────────────────────────────────────────────────┐
    │  (Merged in a later step 10, not shown here)          │
    │  Revenue_estab = Revenue_EIN × (emp_st / emp_total)   │
    └───────────────────────────────────────────────────────┘
```

### Output Datasets

```
┌──────────────────────────────────────────────────────────────┐
│  br_2001_2016_revenues                                       │
│  Unit of observation: EIN × YEAR                             │
│  ────────────────────────────────────────────────────         │
│  EIN            │ Federal EIN                                │
│  REVENUES       │ Calculated annual revenue/receipts         │
│  YEAR           │ Calendar year                              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  br_2017_2018_revenues                                       │
│  Unit of observation: FIRMID × YEAR                          │
│  ────────────────────────────────────────────────────         │
│  FIRMID         │ Census firm identifier                     │
│  REVENUES       │ Net revenue (NREV2)                        │
│  YEAR           │ Calendar year                              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  state_rev_weights_allyears1                                 │
│  Unit of observation: EIN × STATE × YEAR  (2001-2016)        │
│  ────────────────────────────────────────────────────         │
│  EIN            │ Federal EIN                                │
│  BDS_ST         │ State FIPS code                            │
│  YEAR           │ Calendar year                              │
│  EMP_UPPER_ST   │ Total employment at EIN in this state      │
│  EMP_UPPER      │ Total employment at EIN nationally         │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  state_rev_weights_allyears2                                 │
│  Unit of observation: FIRMID × STATE × YEAR  (2017-2018)     │
│  ────────────────────────────────────────────────────         │
│  FIRMID         │ Census firm identifier                     │
│  BDS_ST         │ State FIPS code                            │
│  YEAR           │ Calendar year                              │
│  EMP_UPPER_ST   │ Total employment at firmid in this state   │
│  EMP_UPPER      │ Total employment at firmid nationally      │
└──────────────────────────────────────────────────────────────┘
```

### Key Cleaning Decisions

| Decision | Rationale |
|----------|-----------|
| **Year-specific revenue calculation for 2001** | 2001 SSL files use raw IRS form fields (ACSR1-4) with sector-specific aggregation rules per Haltiwanger et al. (2020) |
| **Sector-specific revenue rules** | Different IRS forms report revenue components differently by 2-digit NAICS; the flowchart ensures correct totals |
| **Switch to BESTADMIN_RCPT for 2002-2016** | Census pre-computed best-available revenue measure eliminates need for manual calculation |
| **FIRMID-level revenue for 2017-2018** | EIN-level data unavailable; only firm-level BRFIRM files are accessible |
| **Employment-weighted revenue distribution** | Multi-state firms have one EIN revenue figure; allocating to states by employment share is standard practice |
| **Two separate output revenue files** (EIN-based vs FIRMID-based) | Reflects the structural break in identifiers between 2016 and 2017 |
| **NODUPKEY on EIN (or FIRMID) within year** for weights | Ensures one weight per EIN-state-year |

---

## Cross-Step Data Flow

```
                    LEHD ECF (quarterly, by state)
                    ┌───────────────┐
                    │ SEIN, SEINUNIT│
                    │ YEAR, QUARTER │
                    │ FIRMID, EIN   │
                    │ location,     │
                    │ industry, emp,│
                    │ wages         │
                    └───┬───────┬───┘
                        │       │
              ┌─────────┘       └──────────┐
              ▼                            ▼
    ┌─────────────────┐          ┌──────────────────┐
    │   STEP 1        │          │   STEP 4         │
    │   Estab-year    │          │   Worker job      │
    │   panel         │          │   history panel   │
    │   (SEIN×SEIN-   │          │   (PIK × spell)   │
    │    UNIT×YEAR)   │          │                   │
    └────────┬────────┘          └────────┬──────────┘
             │                            │
             │    ┌───────────────┐        │
             │    │   STEP 7      │        │
             │    │   Revenue     │        │
             │    │   (EIN×YEAR)  │        │
             │    │   + state     │        │
             │    │   weights     │        │
             │    └───────┬───────┘        │
             │            │               │
             ▼            ▼               ▼
    ┌─────────────────────────────────────────────┐
    │           STEP 10 (not provided)            │
    │  Merge establishment panel + revenues +     │
    │  worker histories → analysis dataset        │
    └─────────────────────────────────────────────┘
```
