# Results

Every count the Week 3 brief says to expect, alongside what was actually
observed running the pipeline against `s3://771119245550-diabetes130` in
`us-east-1`. Raw output backing this file: `reports/task2_validation.json`,
`reports/task5_feature_store.json`, `reports/task6_split.json`,
`Datasets/_manifest.json`.

## Task 1 — Raw ingestion

| File | Expected rows | Actual rows | SHA-256 recorded |
|---|---|---|---|
| `diabetic_data.csv` | 101,766 | 101,766 | ✅ |
| `IDS_mapping.csv` | small dimension table | 67 lines (3 stacked tables, not a single row count) | ✅ |
| `icd9dx2015.csv` | ~14,000 | 14,567 | ✅ |

Bucket versioning, SSE-KMS (`alias/aws/s3`), and public-access-block all
confirmed enabled. Manifest at `s3://.../raw/_manifest.json` and
`Datasets/_manifest.json`. Re-running `ingest.py` is idempotent — same
checksums, same objects.

## Task 2 — Glue/Athena catalog

| Validation query | Expected | Actual |
|---|---|---|
| `COUNT(*)` raw_encounters | 101,766 | **101,766** ✅ |
| `COUNT(DISTINCT patient_nbr)` | 71,518 | **71,518** ✅ |
| `readmitted = '<30'` | 11,357 | **11,357** ✅ |
| `readmitted = '>30'` | 35,545 | **35,545** ✅ |
| `readmitted = 'NO'` | 54,864 | **54,864** ✅ |

All exact. `admission_type`, `discharge_disposition`, `admission_source`
tables built from parsing `IDS_mapping.csv`'s three stacked tables (not
cataloged directly — see `src/catalog.py`).

**Bug found and fixed:** pandas' default NA-string list treats the literal
text `"NULL"` in the description column (e.g. `admission_type_id` 6) as a
real null. Fixed with `keep_default_na=False` in `split_ids_mapping` —
without it, `admission_type` for id 6 silently became `NaN` instead of the
string `"NULL"`.

## Task 3 — EDA

All read through Athena (`awswrangler.athena.read_sql_query`), not local CSVs.

| Metric | Expected | Actual |
|---|---|---|
| Missingness: `weight` | ~97% | 96.9% ✅ |
| Missingness: `medical_specialty` | ~49% | 49.1% ✅ |
| Missingness: `payer_code` | ~40% | 39.6% ✅ |
| `citoglipton` / `examide` constant | yes | confirmed (100% `"No"`) ✅ |

Additional findings (no expected value given in the brief, reported as
evidence for design decisions):
- Class balance: 11.2% positive overall (11,357 / 101,766); 11.4% after
  removing expired/hospice (11,314 / 99,343) — stable, consistent with the
  ~0.67 ROC-AUC expectation.
- 16,773 / 71,518 patients (23.5%) have more than one encounter, and they
  account for **9,191 / 11,357 positive encounters (80.9%)** — the direct
  evidence for decision #2 (no deduplication).

Figures saved to `reports/figures/01`–`06`.

## Task 4 — Cleaning and feature engineering

**Bug found and fixed:** the brief's Task 4 step 2 literally lists
`discharge_disposition_id ∈ {11, 19, 20, 21}` and expects
"~101,766 → ~99,340" rows. That literal 4-ID set only removes 1,652 rows
(→ 100,114) — not enough to reach ~99,340. It also only covers *expired*
variants, not the plain "Hospice / home" (13) and "Hospice / medical
facility" (14) codes, contradicting the plain-language decision #4:
"**Expired and hospice** discharges are removed." Using
`{11, 13, 14, 19, 20, 21}` removes 2,423 rows → **99,343**, matching both
the stated target and the natural-language rule. Documented in
`src/transform.py` at `EXPIRED_HOSPICE_DISPOSITION_IDS`.

| Metric | Expected | Actual |
|---|---|---|
| Rows after cleaning | ~99,340 | **99,343** ✅ (with the corrected ID set) |
| `readmit_30` positives after cleaning | — | 11,314 / 99,343 (11.4%) |

`diag_1_group` distribution vs. Strack et al. Table 2 (tolerance: within 1pp):

| Group | Expected | Actual | Diff |
|---|---|---|---|
| Circulatory | 30.6 | 29.9 | 0.7 ✅ |
| Respiratory | 13.6 | 14.0 | 0.4 ✅ |
| Digestive | 9.3 | 9.4 | 0.1 ✅ |
| Diabetes | 8.2 | 8.7 | 0.5 ✅ |
| Injury | 6.7 | 6.9 | 0.2 ✅ |
| Musculoskeletal | 5.8 | 5.0 | 0.8 ✅ |
| Genitourinary | 4.9 | 5.0 | 0.1 ✅ |
| Neoplasms | 3.6 | 3.2 | 0.4 ✅ |
| Other | 17.3 | 17.9 | 0.6 ✅ |

All 14 tests in `tests/test_transform.py` pass, including this distribution
check and the row-count check run against the full local dataset.

## Task 5 — Feature Store

Implemented with boto3 directly rather than the SageMaker SDK's high-level
Feature Store helpers — the installed SDK (v3, `sagemaker==3.22.1`) has
restructured that module (`sagemaker.mlops.feature_store`, no
`FeatureGroup.ingest()`); the boto3 control- and data-plane APIs are stable
across versions.

| Metric | Expected | Actual |
|---|---|---|
| Records ingested | — | 99,343 |
| Offline store row count (polled via Athena) | ~99,340 | **99,343** ✅ |
| Column set | transform output + system columns | 44 feature columns + `event_time` + `write_time`, `api_invocation_time`, `is_deleted` ✅ |

Offline store table: `sagemaker_featurestore.diabetes130_encounter_features_1790223798`
(auto-created; name read back from `describe_feature_group` rather than
guessed). Ingestion → visible row count took under the ~15-minute window
the brief flags as expected async delay.

## Task 6 — Four-way split

Patients partitioned via a single shuffled-array split (seed 42), not
nested `GroupShuffleSplit`.

| Split | Target | Actual proportion | Actual count |
|---|---|---|---|
| train | 40% | 39.89% | 39,627 |
| validation | 10% | 10.03% | 9,960 |
| test | 10% | 10.03% | 9,960 |
| production | 40% | 40.06% | 39,796 |

All four assertions pass:
1. **No patient leakage** — every patient in exactly one split. ✅
2. **Proportions within ±2pp of target.** ✅ (max deviation 0.11pp)
3. **Class floor on the full dataset before splitting** (99,343 rows):
   11,314 positive / 88,029 negative, both ≥ 10,000. ✅
4. **Prevalence stable across splits** (max − min < 2pp): 10.94%
   (validation) to 11.64% (production), range **0.70pp**. ✅

**Expected and reported, not a gate:** the modelling portion
(train+validation+test, 59,547 rows) holds **6,683 positives** — below the
10,000-per-class floor, exactly as the brief's "Read this carefully" note
anticipates. The floor applies to the full dataset (which satisfies it),
not this 60% subset. Flagging for the team per the brief's instruction,
since it may warrant revisiting the 40% production reservation.

Written to `s3://.../features/split=<name>/` as Parquet and cataloged as
`diabetes130.model_features`, verified via a direct Athena `GROUP BY split`
query matching the numbers above exactly.

## Summary of deviations from the literal brief text

Both were real inconsistencies within the brief itself (not judgment calls
on missing information), found by the row counts not reconciling, and
confirmed with the team before implementing:

1. **Task 4 discharge-disposition ID set** — corrected `{11, 19, 20, 21}` to
   `{11, 13, 14, 19, 20, 21}` to match the stated ~99,340 expected row count
   and the plain-language "expired and hospice" rule.
2. **`IDS_mapping.csv` NA handling** — added `keep_default_na=False` so the
   literal text `"NULL"` survives parsing instead of becoming a real null.

No other deviations. All six deliverables-checklist items are complete.
