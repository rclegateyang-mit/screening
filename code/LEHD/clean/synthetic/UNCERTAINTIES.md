# Synthetic Data: Uncertainties and Assumptions

This document flags aspects of the LEHD/LBD data structure where the SAS code was ambiguous, incomplete, or where our synthetic data makes assumptions that may not match reality.

---

## Step 1: Establishment Panel (`lehd_lbd_allstates_CZ.csv`)

### Things we are confident about
- **Schema:** The column names and types are directly readable from the SAS code (variable names in `keep`, `rename` statements).
- **Unit of observation:** SEIN × SEINUNIT × YEAR, deduplicated via `proc sort nodupkey`.
- **EMP_LEHD** is March employment (`best_emp3` from Q1).
- **PAY_LEHD** is summed across 4 quarters within SEIN × SEINUNIT × YEAR.

### Uncertainties

| Issue | What we assumed | What might differ |
|-------|----------------|-------------------|
| **SEIN format** | 6-character alphanumeric (e.g., "A00001") | Real SEINs may be longer or have different character patterns; format varies by state |
| **SEINUNIT format** | 3-digit zero-padded string (e.g., "001") | Could be different length; might be numeric in some states |
| **LEHDNUM construction** | Simple concatenation `SEIN || SEINUNIT` | SAS `cat()` function concatenates without delimiter; exact behavior with varying-length strings may produce trailing spaces |
| **FAS_EIN structure** | 14 characters: 5-char prefix ("EINUS") + 9-digit EIN | The code splits with `substr(fas_ein, 1, 5)` and `substr(fas_ein, 6, 9)`, confirming this. But non-"EINUS" prefixes are possible for foreign entities |
| **FAS_EIN_FLAG values** | Used 0/1 in synthetic data | May be a character flag with multiple values; the SAS code does not reveal the codebook |
| **LBD_MATCH values** | 0/1 indicator | Appears to be binary but might have additional values |
| **CBSA codes** | Numeric 5-digit codes | Correct per Census definitions, but some establishments may have missing CBSA (rural areas) — we show this with A00003 |
| **CBSA_F values** | 0/1 (non-metro/metro) | The original variable `leg_cbsa_memi` likely has values like 1 (metro) and 2 (micro) per Census convention, not 0/1 |
| **PAY_LEHD magnitude** | Dollar amounts in the millions for mid-size establishments | Plausible but we don't know if values are reported in dollars, thousands, or another unit |
| **Missing FIRMID** | Shown for A00003 (small Ohio contractor) | The code backfills from LBD only when `ein_5d="EINUS"` and `firmid=""`, so some establishments genuinely lack FIRMID even after backfill |
| **CZ variables** | We included CZ_ID2000, CZ_ID1990, CZ_ID1980 | These come from the crosswalk; some counties may map to different CZs across vintages, but we kept them identical in synthetic data for simplicity |
| **IN_202 and IN_UI** | Set to 1 for all synthetic rows | In practice, some establishments may be in ES-202 but not UI or vice versa; these flags matter for data quality |

---

## Step 4: Job History (`step4_full_job_history_all2.csv`)

### Things we are confident about
- **Schema:** The kept variables are explicit in the SAS `keep` statement.
- **Unit of observation:** PIK × job spell, sorted by PIK then chronological order.
- **JOB_SEQ** is a within-PIK sequential counter starting at 1.
- **PLANT_FIRST/PLANT_LAST** are `SEIN || "_" || SEINUNIT1` (via SAS `catx('_', sein, seinunit1)`).

### Uncertainties

| Issue | What we assumed | What might differ |
|-------|----------------|-------------------|
| **PIK format** | 6-character alphanumeric (e.g., "P00001") | Real PIKs are 9-digit numeric identifiers (Protected Identification Keys) |
| **Date format for FIRST_ACC_MIN, LAST_SEP_MAX** | ISO dates (YYYY-MM-DD) | SAS dates are stored as integers (days since Jan 1, 1960); the CSV representation depends on format applied. In practice these would be SAS date values |
| **PLANT_FIRST vs PLANT_LAST** | Different only for P00002 (internal transfer A00001_001 → A00001_002) | The code tracks these using `retain`, keeping the plant at the start and end of each spell. Transfers within a firm (same FID, different SEINUNIT) are the typical case |
| **What constitutes a "spell"** | One continuous employment episode at a firm | SPELL_U2W in the JHF marks unemployment-to-work transitions. A worker can have multiple spells at the same firm (quit and return). Our synthetic data shows this is tracked at the PIK × FID × SPELL_U2W level |
| **Cross-state job histories** | Worker P00001 has jobs in MD then NY | The code loops over states and appends, so a worker's full history can span states. But the LEHD only covers 28 states — jobs in non-covered states are invisible |
| **ECF geography merge** | Inner join drops spells at establishments not in ECF | This is a potential source of data loss; our synthetic data assumes all spells survive the merge |
| **CZ_ID2000 in step 4** | Comes from `all_mergers_list`, not the county-CZ crosswalk | In the merger project, CZ is sourced from the merger establishment list. For our project, we would source it from the county-CZ crosswalk instead |
| **Quarterly earnings (E1-E21)** | Not in the output | The JHF `keep` statement in step 4 only retains PIK, SEIN, SEINUNIT1, FID, SPELL_U2W, FIRST_ACC, LAST_SEP. Earnings variables are dropped. For our project, we would want to retain these |

---

## Step 7: Revenues and Weights

### Things we are confident about
- **Schema:** Variable names are explicit in the SAS code.
- **Two-period structure:** EIN-based (2001-2016) vs FIRMID-based (2017-2018) is clear.
- **Revenue calculation for 2001** follows Haltiwanger et al. (2020) Appendix A flowchart.
- **Employment weights** are EIN × state totals divided by EIN national total.

### Uncertainties

| Issue | What we assumed | What might differ |
|-------|----------------|-------------------|
| **EIN format** | 9-digit numeric string | The SAS code extracts EIN via `substr(fas_ein, 6, 9)` in step 1, but the SSL/BR files may use a different EIN format or include check digits |
| **Revenue units** | Dollar amounts | Could be in thousands; SSL documentation would clarify |
| **Revenue = missing for some sectors in 2001** | When `acsr1f in ("6","7")` for non-FIRE/services sectors, `revenues = .` (missing) | These are legitimate missing values where IRS form type doesn't contain revenue information |
| **NREV2 for 2017-2018** | Used directly as revenue | NREV2 is "net revenue measure 2" — the exact definition (gross vs net, what's included) requires BR documentation |
| **Employment weights: inner join** | The state weights use an inner join (`where A.ein=B.ein`) | EINs/FIRMIDs not in both the weights and revenue files are dropped. We assume this is intentional |
| **Multi-state firms** | A00004 (hospital) shown with TX and CA establishments | Revenue is at the EIN level; the weight `emp_upper_st / emp_upper` distributes proportionally. This assumes revenue productivity is uniform across states within a firm, which may be poor for heterogeneous firms |
| **Revenue apportionment to SEINUNIT level** | Not shown — step 7 only goes to EIN × state | Step 10 (not provided) presumably further distributes to SEIN × SEINUNIT using within-state employment shares. Our synthetic establishment panel would need this additional step |

---

## General Uncertainties Affecting All Files

1. **Variable encoding:** SAS character vs numeric types are not always clear from the code. Variables like `in_202`, `lbd_match` could be character "0"/"1" or numeric 0/1.

2. **Missing value representation:** SAS uses `.` for numeric missing and `""` for character missing. Our CSVs use empty strings, which is the closest CSV convention.

3. **Scale of identifiers:** Real SEIN, PIK, FIRMID, and LBDFID values have specific formats defined by Census. Our alphanumeric placeholders are structurally simplified.

4. **States covered:** The code lists 28 states: MD, AL, AZ, CO, CT, DE, IA, IN, KS, MA, ME, ND, NJ, NM, NV, NY, OH, OK, PA, SC, SD, TN, TX, UT, VA, WA, WI, WY. The paper says LEHD covers these same 28 states over 2000-2022. Jobs in the remaining 22 states + DC are not observed.

5. **Time coverage:** The SAS code processes 2001-2018 (with 2000 as a lookback year). The NSF proposal mentions 2000-2022. The code predates the proposal or was written for an earlier vintage.

6. **The "simplified" suffix:** These files are labeled `_simplified.sas`, suggesting they are cleaned/reduced versions of larger original scripts. Steps, variables, or processing logic may have been removed.
