# Implementation brief: Diabetes 130 readmission pipeline

Build the data layer for an ML system that predicts 30-day hospital readmission.
This covers ingestion through to a split, Feature Store-backed dataset. Model
training is out of scope for this brief.

---

## Decisions already made — do not revisit

These were settled during dataset selection. Changing any of them invalidates
downstream work.

1. **Target is binary.** `readmit_30 = (readmitted == "<30")`. Both `>30` and
   `NO` are negative. Do not build a 3-class model.
2. **Do NOT deduplicate to one encounter per patient.** 101,766 encounters come
   from 71,518 patients. Deduplicating shrinks the positive class below the
   10,000-per-class floor this project requires. Keep every encounter and handle
   leakage by splitting on patient instead.
3. **Every split is a patient-level group split.** No random row split anywhere,
   at any stage, for any reason.
4. **Expired and hospice discharges are removed** before anything else. Those
   patients cannot be readmitted.
5. **Expected model performance is ~0.67 ROC-AUC.** If anything in validation
   substantially exceeds this, treat it as a leakage bug and investigate.

---

## Configuration

Put these in `config.yaml` and read them everywhere. No hardcoded bucket names
or paths in scripts.

```yaml
project: diabetes130-readmission
region: us-east-1
bucket: <account>-diabetes130            # single bucket, prefixes below
prefixes:
  raw: raw/
  processed: processed/
  features: features/
  athena_results: athena-results/
glue_database: diabetes130
feature_group: diabetes130-encounter-features
splits:
  train: 0.40
  validation: 0.10
  test: 0.10
  production: 0.40
seed: 42
```

## Repo layout

```
.
├── config.yaml
├── requirements.txt
├── src/
│   ├── config.py            # loads config.yaml
│   ├── ingest.py            # Task 1
│   ├── catalog.py           # Task 2
│   ├── transform.py         # Task 4 — cleaning + feature engineering
│   ├── feature_store.py     # Task 5
│   └── split.py             # Task 6
├── notebooks/
│   └── 01_eda.ipynb         # Task 3
└── tests/
    └── test_transform.py
```

---

## Task 1 — Raw data into the S3 datalake

**Sources.** Three files.

| File | Origin | Approx records |
|---|---|---|
| `diabetic_data.csv` | UCI dataset 296 zip | 101,766 |
| `IDS_mapping.csv` | same zip | small dimension table |
| `icd9dx2015.csv` | `https://data.nber.org/data/ICD9ProviderDiagnosticCodes/2015/icd9dx2015.csv` | ~14,000 |

Download the UCI zip manually and commit nothing but a checksum. Do **not** use
the `ucimlrepo` package: it returns `.data.features` / `.data.targets` and
`patient_nbr` may not survive into the feature frame, which breaks the grouped
split that everything else depends on.

**Write to** `s3://<bucket>/raw/<source>/<filename>`, one prefix per source file
so Athena can catalog each independently:

```
raw/encounters/diabetic_data.csv
raw/id_mapping/IDS_mapping.csv
raw/icd9/icd9dx2015.csv
```

**Requirements.**
- Enable bucket versioning and SSE-KMS encryption. Block all public access.
- The raw zone is immutable. Nothing in this project ever writes back to it.
- Record SHA-256 of each file in `raw/_manifest.json` alongside row count and
  ingestion timestamp. This is the provenance record.

**Watch out.** `IDS_mapping.csv` is not a normal CSV. It contains three stacked
tables separated by blank lines, one each for `admission_type_id`,
`discharge_disposition_id`, and `admission_source_id`. Parse it into three
separate frames. Do not `pd.read_csv` it and assume you got a table.

**Done when** all three objects exist in S3, the manifest records matching row
counts, and re-running the script is idempotent.

---

## Task 2 — Glue catalog and Athena tables

Create Glue database `diabetes130`. Define external tables over the raw prefixes
so the data is queryable before any processing.

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS diabetes130.raw_encounters (
  encounter_id bigint,
  patient_nbr bigint,
  race string,
  gender string,
  age string,
  -- ... remaining 45 columns, all string unless clearly numeric
  readmitted string
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar'=',', 'quoteChar'='"')
LOCATION 's3://<bucket>/raw/encounters/'
TBLPROPERTIES ('skip.header.line.count'='1');
```

**Notes.**
- Use `OpenCSVSerde`, not `LazySimpleSerDe` — the data contains quoted fields
  with embedded commas in `medical_specialty` and the diagnosis descriptions.
- `OpenCSVSerde` reads every column as string regardless of declared type. Cast
  in the query, or declare everything string and cast downstream. Do not fight
  this.
- Set the Athena query result location to `s3://<bucket>/athena-results/`.

Create equivalent tables for `raw_icd9` and the three ID mapping tables.

**Validation queries** — run these and save the output, they go in the report:

```sql
SELECT COUNT(*) FROM diabetes130.raw_encounters;
-- expect 101766

SELECT COUNT(DISTINCT patient_nbr) FROM diabetes130.raw_encounters;
-- expect 71518

SELECT readmitted, COUNT(*) FROM diabetes130.raw_encounters GROUP BY readmitted;
-- expect roughly <30: 11357, >30: 35545, NO: 54864
```

**Done when** all three counts match. If they don't, the SerDe or header setting
is wrong — fix it before moving on.

---

## Task 3 — EDA notebook

`notebooks/01_eda.ipynb`, reading through Athena (via `awswrangler.athena.read_sql_query`)
rather than pulling CSVs, so the notebook demonstrates the catalog is working.

Cover, in this order:

1. **Class balance.** Overall, and the count after removing expired/hospice.
2. **Encounters per patient.** Histogram. Explicitly compute how many patients
   have >1 encounter and what fraction of positives they account for. This is
   the evidence for the no-deduplication decision — it goes in the report.
3. **Missingness profile.** Percent null per column after `"?"` → null.
   Expect `weight` ~97%, `medical_specialty` ~49%, `payer_code` ~40%.
4. **Readmission rate by subgroup.** By race, gender, age band, primary
   diagnosis group, and discharge disposition. The demographic cuts are the
   baseline for the fairness section.
5. **HbA1c relationship.** Readmission rate by `A1Cresult` level including
   not-tested. This replicates the original paper's research question.
6. **Prior utilization.** Distribution and readmission rate by
   `number_inpatient`, `number_emergency`, `number_outpatient`. Expect these to
   be the strongest signals and heavily right-skewed.
7. **Zero-variance check.** Confirm `citoglipton` and `examide` are constant.

Save every figure to `reports/figures/`. Do not leave plots trapped in notebook
output.

---

## Task 4 — Cleaning and feature engineering

`src/transform.py`. Pure functions, no S3 calls inside the transform logic, so
it's unit-testable.

### Cleaning, in order

1. Replace the literal string `"?"` with null across all columns.
2. Drop rows where `discharge_disposition_id` ∈ {11, 19, 20, 21}. Expect
   ~101,766 → ~99,340.
3. Build `readmit_30` from `readmitted == "<30"`, then drop `readmitted`.
4. Drop `weight` (~97% null), `citoglipton`, `examide` (zero variance).
5. **Keep `encounter_id`** — Feature Store needs it as the record identifier.
   **Keep `patient_nbr`** — the split needs it. Both are excluded from the model
   feature list later, but they must survive to Feature Store.

### ICD-9 grouping

Applied to `diag_1`, `diag_2`, `diag_3`, producing `diag_1_group` etc. Ranges
from Strack et al. (2014) Table 2:

| Group | Codes |
|---|---|
| Diabetes | 250.xx |
| Circulatory | 390–459, 785 |
| Respiratory | 460–519, 786 |
| Digestive | 520–579, 787 |
| Genitourinary | 580–629, 788 |
| Injury | 800–999 |
| Musculoskeletal | 710–739 |
| Neoplasms | 140–239 |
| Other | everything else, including all V and E codes |

```python
def icd9_group(code):
    if pd.isna(code):
        return "Missing"
    code = str(code).strip()
    if code.startswith(("V", "E")):
        return "Other"
    try:
        v = float(code)
    except ValueError:
        return "Other"
    if 250 <= v < 251:                    return "Diabetes"
    if 390 <= v <= 459 or int(v) == 785:  return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:  return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:  return "Digestive"
    if 580 <= v <= 629 or int(v) == 788:  return "Genitourinary"
    if 800 <= v <= 999:                   return "Injury"
    if 710 <= v <= 739:                   return "Musculoskeletal"
    if 140 <= v <= 239:                   return "Neoplasms"
    return "Other"
```

**Acceptance criterion.** The resulting `diag_1_group` distribution must be
within one percentage point of: Circulatory 30.6, Respiratory 13.6, Digestive
9.3, Diabetes 8.2, Injury 6.7, Musculoskeletal 5.8, Genitourinary 4.9,
Neoplasms 3.6, Other 17.3. If it isn't, the mapping is wrong. Assert this in
the test suite.

### Derived features

| Feature | Derivation |
|---|---|
| `age_ordinal` | `age` band `[0-10)`…`[90-100)` → integer 0–9 |
| `n_med_changes` | count of the 23 drug columns with value `Up` or `Down` |
| `n_diabetes_meds` | count of the 23 drug columns not equal to `No` |
| `insulin`, `metformin` | kept as individual categoricals — high enough prevalence to matter |
| `a1c_tested` | `A1Cresult` not null |
| `a1c_level` | `A1Cresult` value, `NotTested` where null |
| `glu_tested`, `glu_level` | same pattern on `max_glu_serum` |
| `specialty_missing` | indicator on `medical_specialty` |
| `payer_missing` | indicator on `payer_code` |
| `medical_specialty`, `payer_code` | nulls → `"Unknown"`, kept as categoricals |
| `admission_type`, `discharge_disposition`, `admission_source` | decoded via the IDS mapping tables |
| `prior_visits_total` | `number_inpatient + number_emergency + number_outpatient` |

Drop the 21 remaining individual drug columns after deriving the counts.

**Do not** scale, encode, or impute here. Those are model-stage operations fit
on training data only. This stage produces a clean, typed, human-readable
feature table.

---

## Task 5 — SageMaker Feature Store

Feature group `diabetes130-encounter-features`, offline store only. There is no
online serving requirement — this is a batch scoring system.

**Required schema fields.**
- Record identifier: `encounter_id`. This is why it survived cleaning.
- Event time: `event_time`, float64 epoch seconds. The dataset has no real
  timestamps, so use ingestion time. Note this limitation in the report — it
  means Feature Store's time-travel capability is not meaningfully exercised.

**Gotchas that will cost you an afternoon each.**

1. Feature Store accepts only `String`, `Integral`, and `Fractional`. Cast every
   pandas `object` column to `str` and every `bool` to `int` before ingesting.
   `category` dtype is not accepted.
2. Feature names must match `[a-zA-Z0-9_-]+` and cannot start with a digit.
   Check your derived names.
3. Nulls in string columns cause ingestion failures. Fill with an explicit
   sentinel (`"Unknown"`, `"Missing"`) before ingesting. Nulls in numeric
   columns are permitted.
4. Offline store ingestion is asynchronous. Data can take up to ~15 minutes to
   appear in the S3 offline store and the auto-created Glue table. Poll for the
   expected row count rather than assuming it landed; do not chain the split
   step directly onto ingestion without waiting.
5. The offline store creates its own Glue table automatically. Query it through
   Athena to verify, and expect the extra system columns `write_time`,
   `api_invocation_time`, and `is_deleted`. Exclude them downstream.

**Done when** an Athena query against the offline store table returns ~99,340
rows and the column set matches the transform output plus the system columns.

---

## Task 6 — The four-way split

This is the highest-risk task in the brief. Read it fully before writing code.

**Requirement:** train 40%, validation 10%, test 10%, production holdout 40%.

**The production holdout** is not a test set. It is held back to simulate
incoming production traffic for the monitoring and drift work later in the
project. It never touches training or evaluation.

### Method

Split on **patients**, not encounters.

```python
rng = np.random.default_rng(seed)
patients = df["patient_nbr"].unique()
rng.shuffle(patients)

n = len(patients)
bounds = np.cumsum([0.40, 0.10, 0.10, 0.40])[:-1] * n
train_p, val_p, test_p, prod_p = np.split(patients, bounds.astype(int))

assignment = {p: "train" for p in train_p}
assignment.update({p: "validation" for p in val_p})
assignment.update({p: "test" for p in test_p})
assignment.update({p: "production" for p in prod_p})
df["split"] = df["patient_nbr"].map(assignment)
```

Do not use nested `GroupShuffleSplit` calls for a four-way split. It works but
the nesting makes the proportions hard to reason about and hard to review.
Partitioning the shuffled patient array is explicit and obviously correct.

### Assertions — all must fail the build, not warn

```python
# 1. No patient in more than one split.
counts = df.groupby("patient_nbr")["split"].nunique()
assert (counts == 1).all(), "patient leakage across splits"

# 2. Encounter proportions within tolerance of target.
#    They will NOT be exact — patients have varying encounter counts.
#    ±2 percentage points is acceptable; wider means investigate.
for name, target in [("train", .40), ("validation", .10),
                     ("test", .10), ("production", .40)]:
    actual = (df.split == name).mean()
    assert abs(actual - target) < 0.02, f"{name}: {actual:.3f} vs {target}"

# 3. Both classes above 10,000 in the modelling portion (train+val+test).
model_df = df[df.split != "production"]
pos = int(model_df.readmit_30.sum())
assert pos >= 10_000, f"positive class {pos} below project floor"
assert len(model_df) - pos >= 10_000

# 4. Prevalence stable across splits.
prev = df.groupby("split")["readmit_30"].mean()
assert prev.max() - prev.min() < 0.02, f"prevalence drift:\n{prev}"
```

**Expected outcome.** ~99,340 encounters total, ~11% positive. The modelling
portion (60%) should hold roughly 6,600 positives — **which is below 10,000**.

> **Read this carefully.** The 10,000-per-class requirement applies to the raw
> dataset, which satisfies it at ~11,400 positives across all 101,766 records.
> Assertion 3 above will fail on the 60% modelling subset. That is expected, not
> a bug. Change the assertion to check the **full dataset** before splitting,
> and record the post-split counts as a reported statistic rather than a gate.
> Do not "fix" this by shrinking the production holdout or by rebalancing —
> raise it with the team instead, since it may warrant revisiting the 40%
> production reservation.

**Write** each split to `s3://<bucket>/features/split=<name>/` as Parquet,
partitioned so Athena can read them independently.

---

## Tests

`tests/test_transform.py`, runnable without AWS credentials against a small
fixture.

- `icd9_group` on known codes: `250.83` → Diabetes, `428.0` → Circulatory,
  `V58.49` → Other, `E885.9` → Other, `nan` → Missing, `"abc"` → Other.
- `diag_1_group` distribution within 1pp of the published figures.
- Expired/hospice rows removed; no `discharge_disposition_id` in {11,19,20,21}
  survives.
- `"?"` fully eliminated from the output frame.
- Target construction: `<30` → 1, `>30` → 0, `NO` → 0.
- Split assertions 1 and 4 above run as tests on a fixture.

---

## Deliverables checklist

- [ ] Three raw objects in S3 with a manifest recording checksums and row counts
- [ ] Glue database with queryable Athena tables; three validation queries pass
- [ ] EDA notebook with all seven analyses, figures saved to disk
- [ ] `transform.py` passing the full test suite
- [ ] Feature group ingested, verified by Athena row count against the offline store
- [ ] Four splits in S3, all four assertions documented with actual values
- [ ] A short `RESULTS.md` recording every count this brief says to expect
      alongside what was actually observed
