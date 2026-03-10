# LEHD Snapshot Documentation: Successor-Predecessor File (SPF) — Section 3.2

Source URL: https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/employer_level/spf.html

Release: S2024_R2025Q4. Last updated: January 23, 2026.

---

## 3.2.1. Overview

The Successor-Predecessor File tracks employer identifier changes in the LEHD infrastructure. State-specific UI account numbers, called SEINs, can change for various reasons including legal restructuring, mergers, or divestitures. When a SEIN changes without other modifications, workers may appear to separate from their employer despite maintaining employment, creating spurious job flow statistics.

**File Details:**
- **Successor-Predecessor File (SPF_ZZ)**
- Scope: State
- Key: SEIN SEIN_SUCC QTIME

The SPF uses quarterly UI wage records to identify large worker movements between SEINs. It provides link characteristics showing worker counts and relative employment shares, supplemented by directly reported predecessor-successor relationships from QCEW (ES202) firm microdata. Cross-state relocations are not identified and appear as separations and accessions in respective states.

### Research Use of the SPF

The linkages are incorporated into the Job History File and QWI tables. The JHF is recommended for job-level analyses, while QWI supports establishment-level job flow calculations. Most researchers do not require the SPF directly, though researchers exploring alternative SEIN change accounting methods or firm structure evolution may find it valuable.

## 3.2.2. User Guidance

### Identifying Transitions in UI Data

SPF processing identifies job-level separations and accessions from quarterly UI data. A separation occurs when an individual receives earnings from firm *j* in quarter *t* but not *t+1*. An accession occurs when earnings are received in quarter *t* but not *t-1*.

Individual separations are linked with accessions in periods *t* or *t+1* to identify possible SEIN transitions. These individual-level transitions aggregate into predecessor-successor counts by quarter (qtime), recording the NUM_LEFT variable representing the job count. The MATCH_PERIOD variable reports accession timing. Links with fewer than 6 transitioning jobs are excluded, forming the UI-sourced SPF frame.

### Assessing Strength of SEIN Transitions in UI Data

Four factors assess whether a linkage constitutes a true predecessor-successor relationship:

- Share of employment at predecessor transitioning to successor
- Share of employment at successor from predecessor
- Predecessor firm death (no UI records after transition)
- Successor firm birth (no UI records before transition)

Transition shares appear in RATIO (predecessor) and SUCC_RATIO (successor) variables. An 80% threshold indicates a large employment share. Ratios cap at 100%.

**LINK_UI (SUCC_LINK_UI) Flag Values:**

| Code | Label |
|------|-------|
| 1 | Firm death (birth), flow constitutes large percentage of predecessor (successor) employment |
| 2 | Firm death (birth), flow does not constitute large percentage of predecessor (successor) employment |
| 4 | Not firm death (birth), flow constitutes large percentage of predecessor (successor) employment |
| 5 | Not firm death (birth), flow does not constitute large percentage of predecessor (successor) employment |

### Directly Reported Links from Firm Data

UI-sourced flows supplement with firm-reported links from ES202 data. Firms report predecessors, successors, or both quarterly. The SOURCE flag identifies the transition source:

**SOURCE Flag Values:**

| Code | Label |
|------|-------|
| UI | Wage record flow in UI data only |
| ES | Transition reported in ES202 data only |
| UIES | Wage record flow in UI and ES202 data |

Timing discrepancies between ES202 reports and UI observations result in ES202 transitions being flagged when UI transitions occur.

### Identifying Successor-Predecessor Relationships

For LEHD data products, only the strongest UI wage record-based links filter spurious identifier changes. Directly reported ES202 links qualify even without meeting UI thresholds. Business rules:

- 80% of predecessor jobs link to successor accessions with predecessor death: LINK_UI=1
- 80% of successor jobs from predecessor separations with successor birth: SUCC_LINK_UI=1
- Separations represent 80% of successor employment AND accessions represent 80% of successor employment: LINK_UI=4 AND SUCC_LINK_UI=4
- Wage record flow identified with predecessor/successor reported on ES202: SOURCE="UIES"

Relationships meeting at least one requirement link job spells in the Job History File.

## 3.2.3. Codebook

### Table Metadata for SPF_ZZ

**Access Requirements:** State Approval Required, IRS Approval Required, SSA Approval Required

**Description:** "Flows of separations and hires between employers (SEINs) used to identify successor-predecessor relationships."

**Scope:** State

**Key:** sein sein_succ qtime

**Sort Order:** sein sein_succ qtime

**File Formats:** SAS Data Table, Parquet

### Variable Information for SPF_ZZ

| Variable Name | SAS Type | Length | Parquet Type | Description |
|---|---|---|---|---|
| sein | char | 12 | string | SEIN - predecessor |
| qtime | num | 8 | uint32 | Quarter of separation, 1985Q1=1 |
| sein_succ | char | 12 | string | SEIN - successor |
| match_period | num | 8 | float64 | Percent of transitions where separation precedes accession quarter |
| num_left | num | 8 | float64 | Number of jobs transitioning between firms |
| active_beg_qtr_a | num | 3 | uint32 | First quarter predecessor active on UI |
| active_end_qtr_a | num | 3 | uint32 | Last quarter predecessor active on UI |
| active_beg_qtr_b | num | 3 | uint32 | First quarter successor active on UI |
| active_end_qtr_b | num | 3 | uint32 | Last quarter successor active on UI |
| link_ui | num | 3 | uint8 | Type of link for predecessor firm |
| succ_link_ui | num | 3 | uint8 | Type of link for successor firm |
| bpemp_master | num | 8 | uint32 | Predecessor UI B Employment, max of last 3 quarters |
| ratio | num | 8 | float64 | Percent of jobs at predecessor transitioning to successor (estimated) |
| succ_ratio | num | 8 | float64 | Percent of jobs at successor transitioning from predecessor (estimated) |
| pred_size_class_ui | num | 8 | uint8 | Size class of predecessor based on UI data (B employment in qtime) |
| succ_size_class_ui | num | 8 | uint8 | Size class of successor based on UI data (M employment in qtime+1) |
| emp_es | num | 8 | uint32 | Predecessor ES202 Month 1 Employment, max of last three quarters |
| es_qtrs_off | num | 3 | float32 | Number of quarters removed ES202 event date is from UI flow |
| source | char | 4 | string | Data source of link between firms |

### Codebook Value Details

**link_ui Variable:**

| Code | Label |
|------|-------|
| 1 | Firm death, flow constitutes large percentage of predecessor employment |
| 2 | Firm death, flow does not constitute large percentage of predecessor employment |
| 4 | Not firm death, flow constitutes large percentage of predecessor employment |
| 5 | Not firm death, flow does not constitute large percentage of predecessor employment |

**succ_link_ui Variable:**

| Code | Label |
|------|-------|
| 1 | Firm birth, flow constitutes large percentage of successor employment |
| 2 | Firm birth, flow does not constitute large percentage of successor employment |
| 4 | Not firm birth, flow constitutes large percentage of successor employment |
| 5 | Not firm birth, flow does not constitute large percentage of successor employment |

**pred_size_class_ui Variable:**

| Code | Label |
|------|-------|
| 1 | Predecessor <5 employment on UI |
| 2 | Predecessor 5-19 employment on UI |
| 3 | Predecessor 20-49 employment on UI |
| 4 | Predecessor 50-99 employment on UI |
| 5 | Predecessor 100-249 employment on UI |
| 6 | Predecessor 250-499 employment on UI |
| 7 | Predecessor 500+ employment on UI |

**succ_size_class_ui Variable:**

| Code | Label |
|------|-------|
| 1 | Successor <5 employment on UI |
| 2 | Successor 5-19 employment on UI |
| 3 | Successor 20-49 employment on UI |
| 4 | Successor 50-99 employment on UI |
| 5 | Successor 100-249 employment on UI |
| 6 | Successor 250-499 employment on UI |
| 7 | Successor 500+ employment on UI |

**source Variable:**

| Code | Label |
|------|-------|
| ES | Link provided on QCEW |
| UI | Sufficient flow found in UI earnings data |
| UIES | Sufficient flow found in UI earnings data and QCEW link |
