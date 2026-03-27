# LEHD Snapshot Documentation: Individual Characteristics Files — Residential Geography (Section 4.2)

Source URL: https://lehd.ces.census.gov/data/lehd-snapshot-doc/S2024/sections/person_level/icfres.html

Release: S2024_R2025Q4

---

## 4.2. Individual Characteristics Files - Residential Geography

## Overview

This section documents residential geography data for wage-earning individuals, with records available from 1999 onward. The target reference date for residence information is April 1 of each year.

Two distinct data sources are used depending on the time period:

- **CPR-Sourced Residence (1999-2011)**: Derives from the Composite Person Record file, with 2011 data sourced from an alternate file after CPR discontinuation
- **RCF-Sourced Residence (2012 onward)**: Sourced from the Residence Candidates File

Both employ a composite key of PIK (Protected Identification Key) and ADDRESS_YEAR, with national scope.

## Composite Person Record (CPR)

The CPR provided the original linkage between individuals and residences. Production continued through 2010, after which the MAF-ARF (Master Address File-Auxiliary Reference File) replaced it for 2011 data. Key distinction: "The MAF-ARF was found to differ from the CPR in a number of ways, including a difference in coverage and a lack of deduplication among PIKs."

County codes reflect geography contemporaneous to each data year, as county boundaries and designations shift over time.

## Residence Candidates File (RCF)

The RCF consolidates multiple federal administrative sources into a preference-weighted file for each person/location/period. This extract contains only the highest-preference location per PIK-year, ensuring one record per combination.

Geography derives from MAFID references mapped to current geography via the Block Map File (BMF). The flag_rcf variable indicates geographic precision levels. Note: "Not all MAFIDs provide tract level precision, and some cannot be mapped into current geography."

## Connecticut County Changes (2023)

Historic Connecticut counties were replaced with nine Councils of Governments (COGs) in 2023 Census geography. Researchers needing historic county assignments may consult MAFX and GRF-C or TIGER data.

## ICF_US_RESIDENCE_CPR Variables

| Variable | Type | Length | Description |
|----------|------|--------|-------------|
| pik | char | 9 | Protected Identification Key |
| address_year | num | 4 | Year address sourced |
| geocodefull | char | 15 | FIPS state, county, tract, block concatenation |
| latitude_live | num | 8 | Residence latitude (6 decimal places) |
| longitude_live | num | 8 | Residence longitude (6 decimal places) |
| flag_latlong | num | 3 | Latitude/longitude quality indicator |

**flag_latlong codes:**

| Code | Label |
|------|-------|
| -1 | Not available for 2011 |
| 1 | Interpolated from house number |
| 2 | Snapped to range end |
| 3 | Complex house number |
| 4 | Missing house number |
| 5 | Road segment midpoint |
| 6 | ZCTA match |
| 7 | County match |

## ICF_US_RESIDENCE_RCF Variables

| Variable | Type | Length | Description |
|----------|------|--------|-------------|
| pik | char | 9 | Protected Identification Key |
| address_year | num | 4 | Year address sourced |
| county_live | char | 5 | Current state/county FIPS code |
| tract | char | 6 | Current Census Tract |
| mafid | char | 9 | Master Address File identifier |
| latitude_live | num | 8 | Internal point latitude (6 decimals) |
| longitude_live | num | 8 | Internal point longitude (6 decimals) |
| flag_rcf | num | 3 | Geographic precision flag |

**flag_rcf codes:**

| Code | Label |
|------|-------|
| 11 | BMF tract in current geography |
| 12 | Current MAF, no precise GEOID |
| 13 | Earlier tab geography, valid current county |
| 14 | Earlier tab, invalid current county |
| 15 | Earlier tab no GEOID, valid MAF county |
| 16 | Earlier tab no GEOID, invalid MAF county |

Both tables require state, IRS, and SSA approval for access and are available in SAS and Parquet formats.
