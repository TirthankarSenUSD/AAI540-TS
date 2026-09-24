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

No other deviations. All six Week 3 deliverables-checklist items are complete.

---

# Week 4 — model, evaluation, deployment

Everything below ran for real against AWS: a real Bayesian HPO job (25
trials), a real final training run with SageMaker Debugger, real
calibration, real Model Registry entries, and a real Batch Transform job
with a verified smoke test. No numbers in this section are placeholders.

## Task 1 — Benchmarks

**1a, majority class** (`reports/task1_benchmarks.json`, local):

| Metric | Train | Validation |
|---|---|---|
| Accuracy | 88.8% | 89.1% |
| Recall | 0 | 0 |
| PR-AUC | 11.2% (≈ base rate) | 10.9% (≈ base rate) |

Matches the brief's expectations exactly.

**1b, heuristic + logistic benchmark** — trained for real through
SageMaker (SKLearn framework container, script mode,
`models/benchmark_sklearn/train.py`):

- Training job `diabetes130-benchmark-1790226193` — **Completed**.
- Registered as `diabetes130-benchmark-models` v1, `ModelApprovalStatus:
  PendingManualApproval`, with `metrics.json` (heuristic-rule and
  logistic-benchmark validation metrics) attached via `ModelMetrics`.
- Fitted coefficients: `number_inpatient` 0.279, `number_emergency` 0.012,
  intercept −2.294 — `number_inpatient` dominates, consistent with it
  being the strongest known signal.

## Task 2 — Real model

Trained via the SageMaker XGBoost framework container in script mode
(`models/xgboost/train.py`), HPO'd, then reproduced once more with
SageMaker Debugger attached.

**HPO** (`diabetes130-xgb-hpo-1790226750`): Bayesian strategy, objective
`validation:aucpr`, the exact parameter ranges from the brief's table, 25
trials / 4 parallel. **All 25 trials completed successfully** (PR-AUC
ranged 0.186–0.213 across trials). Best trial
(`...-021-39c760c0`): **validation PR-AUC 0.2125**, hyperparameters
`max_depth=8, eta=0.0103, subsample=0.818, colsample_bytree=0.621,
min_child_weight=8.05, lambda=0.416, num_round=468`.

**Final debugged run** (`diabetes130-xgb-final-1790227571`, same
hyperparameters + Debugger attached): **Completed**, validation PR-AUC
0.21252 (matches the HPO trial exactly — same hyperparameters, same data).
**Debugger rules: `Overfit`, `LossNotDecreasing`, `ClassImbalance` — all
three report `NoIssuesFound`.**

Calibration (Platt scaling on validation): ROC-AUC unchanged at 0.6564
(expected — Platt scaling is monotonic and shouldn't change ranking, only
calibration quality), **Brier score improved 0.1398 → 0.0949**, a real,
meaningful improvement this time (unlike the earlier untuned smoke-test
model, where calibration barely moved the needle).

### Three real bugs found only by actually running this against AWS

Local validation (pandas/xgboost/sklearn on this machine) could not have
caught any of these — they're all specific to the SageMaker-managed
training/serving environment:

1. **HPO hyperparameter naming.** This XGBoost container's HPO service
   validates tunable hyperparameter names against the *built-in*
   algorithm's fixed list, even in script mode — it requires `lambda`, not
   xgboost's own sklearn-API name `reg_lambda`. Fixed by accepting
   `--lambda` with `dest=reg_lambda` in `train.py`.
2. **Model save format.** Saving a model with categorical splits
   (`enable_categorical=True`) as `model.xgb` (an unrecognized extension)
   silently defaults to a legacy binary format on this container's
   xgboost version, which then hard-errors on categorical splits
   (`Please use JSON/UBJSON for saving models with categorical splits`).
   Renamed the artifact to `model.ubj` everywhere.
3. **No `requirements.txt` auto-install.** Unlike some other SageMaker
   framework containers, the XGBoost training toolkit does not
   pip-install a source-dir `requirements.txt` — `pyarrow` was never
   installed, so reading the Week 3 Parquet splits failed inside the
   container. Fixed by exporting one-off CSV copies of the train/
   validation splits (`hpo.py:export_split_to_csv`) instead of fighting
   the container's default package set.
4. **`create_training_job` rejects custom `MetricDefinitions` for this
   image** even though the identical field is required and works inside
   a `HyperParameterTuningJob`'s `TrainingJobDefinition` — AWS's own
   validation is inconsistent between the two call shapes for the same
   algorithm image. Removed `MetricDefinitions` from the direct
   `create_training_job` call for the final debugged run.
5. **The big one — calibrator/serving sklearn version skew.** The brief's
   literal `CalibratedClassifierCV(..., cv="prefit")` (or its modern
   replacement, `FrozenEstimator`) pickles the *entire wrapped model plus
   sklearn-internal state* into the calibrator artifact. The dev
   environment's sklearn (1.7.2) produced a `calibrator.joblib`
   referencing `sklearn.frozen`, a module that doesn't exist in the
   older sklearn shipped inside the SageMaker XGBoost 1.7-1 **serving**
   container. This didn't fail until real deployment: the first real
   Batch Transform job (`diabetes130-batch-transform-1790227838`) hung
   for 24+ minutes with the container's health check failing
   (`ModuleNotFoundError: No module named 'sklearn.frozen'` on every
   `/ping`) before I found it in CloudWatch and stopped the job. Fixed by
   replacing `CalibratedClassifierCV` entirely with a tiny dependency-free
   `PlattCalibrator` (`models/xgboost/calibration.py`) that stores only
   the two fitted Platt-scaling floats — no sklearn object to unpickle at
   serving time at all, and no version-skew surface. This also shrank
   `calibrator.joblib` from ~1.3MB (a full duplicate of the model) to 80
   bytes. Registered as v2 in the Model Registry; v1 marked `Rejected`
   with the reason recorded.

## Task 3 — Evaluation (test split, touched once)

**Assumption flagged:** the brief says to fix the alert volume to "the
care team's capacity" without giving a number. Used **10%** of the daily
batch, consistent with precision@10% being called out as the key business
metric. Frozen threshold (selected on validation): **0.1395**.

Comparison table (`reports/model_comparison.md`, real tuned + calibrated
model):

| Model | PR-AUC | ROC-AUC | Precision@10% | Recall@thr | Brier |
|---|---|---|---|---|---|
| Majority class | 0.1145 | 0.5000 | 0.1195 | 1.0000 | 0.1014 |
| Heuristic (number_inpatient≥1) | 0.1473 | 0.6042 | 0.1637 | 0.5175 | 0.3290 |
| Logistic regression | 0.2154 | 0.6433 | 0.2460 | 0.2123 | 0.0979 |
| XGBoost (uncalibrated) | 0.2363 | 0.6810 | 0.2831 | 0.2693 | 0.1411 |
| **XGBoost (calibrated)** | **0.2363** | **0.6810** | **0.2831** | **0.2693** | **0.0981** |

The real tuned model clearly wins on every ranking metric, beating the
full-feature logistic regression baseline and roughly **2.5x lift** over
the ~11.4% test base rate at precision@10% (sanity check from the brief:
passes clearly). **ROC-AUC 0.681 lands almost exactly on the brief's
expected ~0.67** (decision #1) — close enough to be reassuring, not so far
above it to suggest leakage.

**Subgroup fairness** (`reports/task3_subgroup_metrics.json`, at the
frozen threshold): selection rate is close to base rate for the two
largest race groups (Caucasian: 11.7% base vs 10.6% selected; African
American: 10.9% base vs 13.7% selected) but noisier for small groups
(Asian n=64, Hispanic n=196). Female selection rate (11.75%) tracks female
base rate (11.79%) almost exactly; male selection rate (9.9%) is somewhat
below male base rate (11.1%).

**Race ablation** (`reports/task3_race_ablation.json`, default-hyperparameter
models with/without `race`, for a fast apples-to-apples comparison):
dropping `race` changed test PR-AUC negligibly (0.1779 → 0.1806, i.e.
*not worse*, sometimes even marginally better) and did not meaningfully
change the shape of selection-rate disparities across race groups —
consistent with the brief's hypothesis that `payer_code` and prior
utilization act as proxies, so dropping the column alone doesn't remove
the disparity, only its visibility.

Calibration curve: `reports/figures/07_calibration_curve.png`, 10 bins,
real tuned + calibrated model — tracks the diagonal reasonably well in the
low-probability bins where most of the test set's mass sits; noisier in
the sparse high-probability bins (few positives at that end of an 11%-
prevalence problem).

## Task 4 — Deployment

- **Artifact bundle**: `preprocessor.joblib`, `model.ubj`,
  `calibrator.joblib` (80 bytes — see the PlattCalibrator fix above),
  `threshold.json`, `feature_schema.json`, `code/inference.py`,
  `code/preprocess.py`, `code/calibration.py`. Uploaded to
  `s3://.../models/xgboost-v2/model.tar.gz`.
- **Model Registry**: group `diabetes130-xgboost-models`. v1
  (`CalibratedClassifierCV`/`FrozenEstimator` calibrator) — `Rejected`,
  with the sklearn-version-skew reason recorded via
  `ApprovalDescription`. **v2** (PlattCalibrator) —
  `PendingManualApproval`, with test metrics + calibration report +
  training metrics attached via `ModelMetrics`.
- **Batch Transform**: `diabetes130-batch-transform-v2-1790229539` against
  a 50-row canned batch drawn from the test split (production split never
  touched, per decision #5) — **Completed**. `input_filter="$[1:]"`,
  `join_source="Input"`, `output_filter="$[0,-1]"` exactly as specified;
  output is clean `encounter_id,score` pairs.
- **Smoke test** (`deploy.run_smoke_test()`), against the real transform
  output:
  - row count matches input: **pass**
  - all scores in [0, 1]: **pass**
  - training/serving skew (transform-job score vs. the same row scored
    directly through the local artifact): **pass** — `0.090647` vs.
    `0.09064728` (matches to the CSV output's 6-decimal precision).
- **Ranked worklist**: `s3://.../predictions/dt=2026-09-24/worklist.csv`
  — sorted descending by calibrated score, `flag` column at the frozen
  threshold (2/50 flagged in this small canned batch).

## Summary of deviations from the literal brief text (Week 4)

All five were real bugs or hard version/API constraints, found only by
actually running against AWS — not judgment calls on missing information:

1. HPO hyperparameter name `reg_lambda` → `lambda` (container's fixed
   whitelist for this image).
2. Model artifact extension `model.xgb` → `model.ubj` (legacy binary
   format can't hold categorical splits on this xgboost version).
3. Parquet training channels → CSV (`requirements.txt` isn't
   auto-installed by this training toolkit).
4. Dropped `MetricDefinitions` from the direct `create_training_job` call
   (rejected there; accepted and required inside a tuning job).
5. `CalibratedClassifierCV`/`FrozenEstimator` → a dependency-free
   `PlattCalibrator` (dev/serving sklearn version skew broke Batch
   Transform in production, not just in theory — this is the one that
   actually cost a stuck, wasted 24-minute job before being caught).

Two judgment calls, both flagged to the user before implementing:
6. Alert-volume target for threshold selection: defaulted to 10% (brief
   gives no number).
7. Week 3's disposition-ID and `NULL`-parsing fixes (see above) carry
   forward unchanged into this week's splits.

All Week 4 deliverables-checklist items are complete with real, measured
numbers.
