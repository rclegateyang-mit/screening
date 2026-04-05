# Census RDC Disclosure Avoidance Rules

Rules for getting results out of the RDC. **Nothing leaves without disclosure review.**

Sources: FSRDC Disclosure Avoidance Methods Handbook v5.0 (Oct 2024),
FSRDC DA Procedures Handbook, FSRDC Researcher Handbook (Feb 2026).

---

## Output Review Process

### Timeline

- **~6 weeks** from output production to cleared results (typical).
- **~1 week** for simple/expedited requests (sign-and-significance only, small volume).
- Longer for complex output or DRB escalation.

### Review chain

1. Prepare output in `/projects/<project_id>/disclosure/YYYYMMDD/`.
2. Include: output, programs that produced it, clearance request memo, disclosure statistics.
3. Set permissions to group read/write/execute.
4. Notify RDCA.
5. **RDCA/DAR review** (initial check).
6. **DAO review** (Disclosure Avoidance Officer): approves or escalates to DRB.
7. **DRB review** (Disclosure Review Board): for complex cases, high volume, or delegated-authority exceptions.
8. If approved, results emailed to you electronically. No paper output.

### Output format requirements

- **Plain text only**: tab-delimited `.txt` or `.csv`.
- Must be readable by Windows Notepad.
- **Other file formats are rejected.**
- Graphical output: `.png`, `.tif`, or `.jpeg` only.

### What to include with a request

- All support files: disclosure statistics, variable/sample definitions.
- Unweighted AND unrounded counts (so reviewers can verify rounding).
- Description of samples/subsamples and output type.
- Clear documentation that all rules were followed.

---

## Rounding Rules

### The 4-significant-digit rule

- **All weighted estimates** must be rounded to **4 significant digits**.
- Coefficients, standard errors, means, proportions, aggregates -- all 4 sig digits.
- Include both rounded and unrounded in support files for verification.

### Unweighted counts

- Subject to coarseness-dependent rounding rules.
- If unweighted count < 15: record as "N < 15" and suppress associated statistics.

### Exception

- If using **modern disclosure avoidance methods** (noise injection, formal privacy), rounding is NOT required.

---

## Cell Size Thresholds

| Context | Minimum cell size |
|---|---|
| General: unweighted counts | **15** (below this, suppress as "N < 15") |
| Frequency tables | **5** per category (collapse if any category < 5) |
| LEHD/admin records | **3** entities (households, persons, jobs, establishments) |
| Pseudo-quantile calculation | **11** observations minimum |
| Kernel density bins | **3** unique entities per bin |

---

## What CAN Be Released

### Regression output (generally safe)

- Regression coefficients and standard errors.
- t-statistics, F-statistics, R-squared, other model fit statistics.
- Sample size N (rounded per rules above).
- **Regression coefficients are considered safe** -- Census considers it impossible to use them
  to reveal individual respondent information.

### Sign and significance (expedited review)

- Just the sign (+/-) and significance level (*, **, ***).
- No numeric estimates needed.
- **Fastest review path** -- use for robustness checks.

### Other releasable statistics

- Weighted means, proportions, aggregates (rounded to 4 sig digits).
- Pseudo-quantiles (see below).
- Kernel density plots (with restrictions, see below).

---

## What CANNOT Be Released

- **Individual or record-level data** (never).
- **True quantiles, percentiles, or medians** -- these correspond to actual confidential values.
  Includes 0th, 1st, 99th, 100th percentiles.
- **Minima and maxima** unless >= 11 unique entities share the same extreme value.
- **Modes**.
- **Scatterplots** of individual-level data (generally not permitted).
- **Intermediate output** not intended for publication.
- **Output you could generate outside the RDC** (public-data stats, paper text, variable definitions).
- **Sub-state geographic tabulations from LEHD** without DRB approval.
- **Coefficients on sub-state geographic controls** from LEHD without DRB approval.

---

## Pseudo-Quantiles (Workaround for Percentiles)

Since true quantiles cannot be released, use pseudo-quantiles:

1. Find the observation at the desired percentile.
2. Take the **mean** of that observation and at least **5 observations on each side** (minimum 11 total).
3. If computing multiple quantiles, the 11-observation windows **must not overlap**.

Stata: use the `pseudop` command (pre-installed).
R: use the `discloseR` package.
Python: available rounding code tutorial on the RDC intranet.

---

## Graphs and Plots

### Kernel density plots (permitted with restrictions)

1. Each bin must contain **>= 3 unique entities**.
2. **Cut off 5% from each tail**.
3. Limit axis detail.
4. **Suppress bandwidth** values (some software prints these; an intruder could estimate true max).
5. Round underlying data to 4 significant digits before computing.
6. **Each grid point counts as 1 estimate** toward the volume limit.

### Histograms

- Less preferred than kernel density plots.
- Raw count histograms are problematic (reveal cell-level information).

### Other graphical output

- Only `.png`, `.tif`, or `.jpeg` formats.
- Graphical output depicting regression coefficients: include underlying regression in support files.
- All graphical output counts toward volume-of-output threshold.

---

## Business/Economic Data: Concentration Ratios

Required for economic/business data (LBD, ASM, CMF, etc.):

- **(n,k) rule**: Statistics cannot be dominated by a small number of firms.
- **p-percent rule**: No single firm's contribution can be closely estimated from the published data.
- **Establishment counts** are NOT considered disclosures and can be published freely.
- Magnitude data (dollar amounts, payroll) has distinct rules from frequency data.

---

## Volume of Output Limits

- **>5,000 total estimates** from a sample: triggers additional DRB review.
- **>1 estimate per 30 unique entities**: triggers additional review.
- Tracked **cumulatively across all requests** from the same sample.
- Cannot circumvent by splitting into multiple requests.
- Kernel density grid points count toward estimate total.

---

## LEHD-Specific Rules

- **National or state-level tabulations**: may be released without prior DRB approval.
- **Sub-state geographic tabulations**: NOT allowed unless using QWI noise infusion process.
  DRB will not approve sub-state geography tables from LEHD infrastructure files.
- **Sub-state geographic controls in models**: coefficients cannot be released without DRB approval.
- LEHD microdata counts: minimum cell size of **3** entities.
- Different LEHD components require different agency approvals:
  - State-level tables: state agency approval per MOU.
  - Person-level demographics (ICF): SSA approval.
  - Employer Title 26 data (T26): IRS approval.
  - Job-level tables: no additional federal approvals.

---

## Code Release Rules

Code can be released after abbreviated review. Requirements:

- **Remove full pathnames** (use relative paths only).
- **Remove user IDs** (Census calls these "James Bond IDs").
- **Remove data values or confidential information** from code and comments.
- **Comments must not reveal data characteristics** (e.g., "Only a few observations were dropped"
  is problematic -- reveals something about the data).
- Submit with a **Program Review Checklist**.
- Timeline: **~1 week** for code release.

---

## Practical Tips for Faster Disclosure Review

1. **Design tables and figures before coding.** Know exactly what you need released.
2. **Use sign-and-significance** for robustness checks (expedited review, no volume concerns).
3. **Aggregate geographic controls** -- don't report coefficients on individual sub-state dummies.
4. **Only submit results for publication.** No intermediate output.
5. **Don't generate inside the RDC what you can generate outside** (paper text, public-data stats).
6. **Build disclosure statistics into your code** -- output cell counts, rounding, and support
   documentation alongside main results automatically.
7. **Track cumulative estimate count** across requests to stay under 5,000-estimate threshold.
8. **Modularize code** (functions, macros) -- makes code faster for RDCA to review.
9. **Coordinate with RDCA well before deadlines** (conferences, journal submissions).
10. **Keep requests short and simple.**

---

## Disclosure Statistics to Compute Alongside Results

For every table/regression, automatically generate:

```python
# Alongside every result, output:
# 1. Unweighted N for each sample/subsample
# 2. Unweighted N per cell (for tables)
# 3. Min cell size flag (< 15 → suppress)
# 4. Frequency check (any category < 5 → collapse)
# 5. For economic data: concentration ratios
# 6. All counts rounded AND unrounded (both needed)
```

Stata: `dadiscstat.ado` and `rounddig.ado` (pre-installed).
R: `discloseR` package.
Python: check `/data/support/researcher/codelib/` and RDC intranet for rounding tutorials.
