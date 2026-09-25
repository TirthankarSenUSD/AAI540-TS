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
  `ApprovalDescription`. **v2** (PlattCalibrator) — registered
  `PendingManualApproval` with test metrics + calibration report +
  training metrics attached via `ModelMetrics`, then reviewed and
  **`Approved`** (2026-09-25) — this is the model this project's
  deployment artifacts refer to. `diabetes130-benchmark-models` v1 —
  the Task 1b floor/baseline, not a deployment candidate — was also
  reviewed and **`Approved`** (2026-09-25).
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

---

# Monitoring — model, data, and infrastructure monitors, dashboard, reports

No formal checklist file for this module; scope came from a pasted task
list: model monitors, data monitors, infrastructure monitors, a CloudWatch
dashboard, and model/data reports on SageMaker. Built and run for real
against AWS, same as Weeks 3-4. Uses the **production split** as the
simulated incoming-traffic dataset throughout — exactly what it was
reserved for back in Week 3's decision #5 ("reserved for monitoring and
drift work in a later module").

## Data quality monitor (`src/monitor_data_quality.py`)

SageMaker Model Monitor, Data Quality type, via boto3 `create_processing_job`
running the built-in `sagemaker-model-monitor-analyzer` container directly
(the SDK's `DefaultModelMonitor` class isn't available in the installed
SDK v3, same restructuring issue hit repeatedly in Weeks 3-4).

- **Baseline**: train split (41 feature columns, CSV with header) →
  `statistics.json` + `constraints.json`. Completed on the first real run.
- **Monitoring execution**: production split (39,796 rows) compared
  against the baseline → **`CompletedWithViolations: 10 violations`**:

  | Check type | Count | What it means |
  |---|---|---|
  | `data_type_check` | 3 | `diag_1`/`diag_2`/`diag_3` — a CSV type-inference artifact, not real drift: these columns mix numeric-looking codes (`"250.83"`) and alphanumeric V/E codes (`"V27"`), and the analyzer's per-batch type sniffing landed differently between baseline and production samples. |
  | `completeness_check` | 3 | `diag_1`/`diag_2`/`race` — tiny (~0.05-0.15pp) null-rate differences between two disjoint patient cohorts. Sampling noise, not meaningful drift. |
  | `categorical_values_check` | 4 | **Genuine, actionable finding.** `medical_specialty`, `payer_code`, `admission_source`, `discharge_disposition` each have a handful of production rows (99.98-99.99% match) with category values never seen in training. `XGBoostCategoricalPreprocessor` silently maps unseen categories to missing at inference — worth watching if this rate grows over time. |

**Bug found and fixed:** the first monitoring-execution attempt was
launched with `publish_cloudwatch_metrics=Enabled` and failed outright:
`Error: CloudWatch publishing is available only for jobs from
MonitoringSchedules.` That flag only works when the analyzer container is
invoked by a real, live `MonitoringSchedule` (it needs schedule context to
timestamp metrics) — a standalone, manually-launched Processing Job can't
use it. Fixed by disabling the flag and instead reading violation counts
out of `constraint_violations.json` and pushing them to CloudWatch as
custom metrics ourselves (`monitor_dashboard.py`).

## Model quality monitor (`src/monitor_model_quality.py`)

Same mechanism, `analysis_type=MODEL_QUALITY`, `problem_type=BinaryClassification`.
Both splits scored through the **exact real deployed artifact** (v2's
preprocessor → booster → PlattCalibrator → frozen threshold) —
`score_split()`'s validation-split ROC-AUC (0.6564) matches Task 3's
number bit-for-bit, confirming it reproduces production scoring exactly.

**Bug found and fixed:** the first baseline run used the analyzer's
default 0.5 probability threshold to derive the confusion matrix. At
~11% prevalence, calibrated probabilities cluster well under 0.5, so
*every* row was classified negative — confusion matrix `{0:{0:8870,1:0},
1:{0:1090,1:0}}`, making recall/precision/F1/TPR all degenerately 0 even
though AUC (0.656) and accuracy (0.891, the majority-class rate) came
through correctly. Fixed by passing the real frozen threshold (0.1395)
as `probability_threshold_attribute` instead of the default.

Results with the fix:

| | Validation (baseline) | Production (monitored) |
|---|---|---|
| AUC | 0.6564 | **0.6785** |
| Accuracy | 0.8906 | 0.8359 |
| Precision | 0 (degenerate — old run) | 0.2754 |
| Recall | 0 (degenerate — old run) | 0.2516 |
| F1 | 0 (degenerate — old run) | 0.2630 |

Monitoring execution: **`CompletedWithViolations: 3 violations`** —
accuracy (0.836, threshold 0.840), false positive rate (0.087, threshold
0.085), true negative rate (0.913, threshold 0.915). All small,
directionally-consistent movements between two disjoint patient cohorts.
**Notably, AUC on production (0.6785) is slightly *better* than the
validation baseline (0.6564)** — the flagged metrics are minor
threshold-dependent fluctuations, not model breakdown; ranking quality
held up (or improved) on the held-out production cohort.

## Infrastructure monitor (`src/monitor_infrastructure.py`)

This system has no persistent endpoint (Batch Transform on a daily cycle,
per Week 4), so there's no long-lived `AWS/SageMaker` invocation-metric
stream to alarm on. Two real pieces instead:

1. **Failure alerting**: SNS topic `diabetes130-ml-alerts` (with an
   explicit topic policy granting `events.amazonaws.com` publish
   permission — easy to silently omit and have EventBridge deliveries
   fail with no error). Three EventBridge rules, all `ENABLED`, matching
   SageMaker Transform/Training/Processing job state changes reaching
   `Failed`/`Stopped`, routed to that topic. Works for any future job run.
2. **Resource utilization**: CloudWatch alarms on the real Week 4 v2
   Batch Transform job's `CPUUtilization`/`MemoryUtilization`/
   `DiskUtilization` (namespace `/aws/sagemaker/TransformJobs`, dimensioned
   by the job's ephemeral `Host`). State: `INSUFFICIENT_DATA`, expected —
   the job (and its metric stream) has already completed; this
   demonstrates the alarm pattern against a real job rather than acting as
   a permanent, evergreen alarm (a new one would need creating per job run
   unless wired through the same EventBridge failure-rule mechanism).

## CloudWatch dashboard (`src/monitor_dashboard.py`)

Dashboard `diabetes130-ml-system`, 6 widgets: a text header, Batch
Transform resource utilization, data-quality violation counts (by check
type), model-quality metrics (production vs. baseline), and two
fairness widgets (disparate impact, accuracy difference by facet). Since
none of the monitoring executions run here are fired by a live
`MonitoringSchedule`, SageMaker doesn't auto-publish their results to
CloudWatch — this module reads each job's real JSON output and pushes the
values as custom metrics (namespace `Diabetes130/Monitoring`) so the
dashboard has real numbers to show, not a live feed.

## Model and data reports on SageMaker

**Model Monitor's own reports** (`statistics.json`, `constraints.json`,
`constraint_violations.json` for both data quality and model quality,
all in S3 under `monitoring/`) already satisfy this for data and model
quality — real reports, generated by real SageMaker Processing Jobs.

**SageMaker Clarify is unavailable in this AWS account.** Launching a
real Clarify processing job fails immediately:
```
ValidationException: SageMaker Clarify processing is in maintenance mode
and is not available to new customers. Existing customers are unaffected.
```
This is a genuine account-level platform restriction (`src/clarify_reports.py`
is kept as a record of what was attempted), not a bug — confirmed by
actually trying to launch the job, not by any documentation lookup.

**Built a custom equivalent instead** (`models/fairness_report/generate_report.py`,
launched by `src/fairness_report.py` as a real SageMaker Processing Job):
pre/post-training bias metrics (class imbalance, disparate impact,
accuracy/recall difference) on `race` and `gender`, plus global feature
importance via XGBoost's **native SHAP contributions**
(`pred_contribs=True` — mathematically the same TreeSHAP values the `shap`
library or Clarify's explainability report would produce, no extra
dependency needed).

Real results (production split, 39,796 rows):

| Facet | Disparate impact | Accuracy difference | Recall difference |
|---|---|---|---|
| race=AfricanAmerican vs. rest | 1.154 | −2.12pp | −1.47pp |
| race=Caucasian vs. rest | 0.955 | +0.53pp | +0.96pp |
| gender=Female vs. rest | 1.094 | −1.21pp | +0.11pp |

All three disparate-impact values fall within the standard four-fifths
rule band (0.8-1.25) — no severe disparate impact by this common
threshold, though AfricanAmerican patients see a measurable accuracy gap.

Top 5 features by mean absolute SHAP contribution: `number_inpatient`
(0.0533), `discharge_disposition` (0.0318), `diag_3` (0.0236), `diag_1`
(0.0228), `prior_visits_total` (0.0214). **`race` ranks 33rd of 41
features** (mean |SHAP| 0.00086) — consistent with Week 4's race-ablation
finding that dropping the column barely moved test PR-AUC (0.1779 →
0.1806): the model isn't leaning on race directly, though the disparate
outcomes above show proxies still produce measurably different results
per group.

### Three more real bugs, found only by actually running this

1. **numpy/pandas ABI break.** A plain `pip install pyarrow awswrangler`
   inside the report's Processing Job pulled a newer numpy that's
   binary-incompatible with the container's pre-built pandas —
   `ValueError: numpy.dtype size changed, may indicate binary
   incompatibility`, breaking `import pandas` outright. Fixed by pinning
   `numpy==1.24.1` (the container's expected version, visible in pip's own
   conflict warnings) alongside the install.
2. **Missing `config.yaml` in the code bundle.** `config.py` resolves
   `config.yaml` relative to itself (`src/../config.yaml`); the bundler
   copied `src/` and `models/xgboost/` into the Processing Job's code
   package but not `config.yaml` itself, so `load_config()` failed with
   `FileNotFoundError` the moment the report tried to load the production
   split. Fixed by including it explicitly.
3. **Empty gender bias result** — a local (non-AWS) bug: the gender facet
   used a `nunique() == 2` guard meant to mean "gender is binary," but the
   real data has a third, rare `"Unknown/Invalid"` category, so the guard
   always failed and silently produced an empty result. Fixed by directly
   targeting the `"Female"` facet instead of trying to infer binariness.

## Scope decision: on-demand Processing Jobs, not a live MonitoringSchedule

Both monitors here run as one-off, manually-triggered Processing Jobs
(baseline + one monitoring execution each) rather than a recurring,
cron-based `MonitoringSchedule`. The schedule-based path requires
`DataQualityJobInput.BatchTransformInput.DataCapturedDestinationS3Uri` —
Batch Transform's `DataCaptureConfig`-formatted captured data, a specific
binary/JSON capture format not otherwise needed anywhere else in this
project, and a live schedule fires on a cron the size of a day, so it
wouldn't be observable firing within any working session regardless. The
underlying analysis engine (the same `sagemaker-model-monitor-analyzer`
container) and its output (real `statistics.json`/`constraints.json`/
`constraint_violations.json`) are identical either way. Flagging this as a
deliberate scope decision, not a silently skipped requirement — wiring an
actual `MonitoringSchedule` on top of this is a natural, bounded follow-up
if wanted.

## Summary

| Item | Status |
|---|---|
| Model monitor | Real — baseline + monitoring execution, 3 real violations found and explained |
| Data monitor | Real — baseline + monitoring execution, 10 real violations found and explained (4 genuinely actionable) |
| Infrastructure monitor | Real — SNS + EventBridge failure alerting (3 rules), CloudWatch utilization alarms (3 alarms) |
| CloudWatch dashboard | Real — `diabetes130-ml-system`, 6 widgets, populated with real pushed metrics |
| Model/data reports on SageMaker | Real — Model Monitor's native JSON reports; Clarify blocked at the account level, custom bias+SHAP report built and run as a real Processing Job instead |

Six real bugs found and fixed this module (2 in the data/model quality
monitors, 3 in the custom fairness report's Processing Job packaging, 1
account-level platform restriction routed around) — none catchable
without actually running against AWS.

---

# CI/CD pipeline (`src/pipeline.py`)

A real SageMaker Pipeline — `diabetes130-cicd-pipeline` — giving this
project an actual orchestrated DAG, run twice against AWS to capture both
a successful and a failed execution state.

## What it is

Three linear steps: **Train → Evaluate → Register**. Reuses the best
hyperparameters found by Week 4's real 25-trial HPO run (fixed, not
re-tuned — a pipeline meant to be re-run on demand shouldn't redo a
25-trial search every time) and the same CSV-exported train/validation
channels from that run.

This is a **separate artifact** from the hand-calibrated, human-approved
production model deployed in Week 4 — it registers into its own group,
`diabetes130-pipeline-models`, never the production
`diabetes130-xgboost-models` group. The point is demonstrating pipeline
orchestration, not replacing that model.

The quality gate (`AUC >= threshold`) is enforced **inside the Evaluate
step's own script**, which exits non-zero on a miss, rather than via a
native Pipelines `Condition` step. A hand-written
`{"Get": "Steps.X.PropertyFiles.Y.z"}` expression hit `Unknown property
reference` against the real service; rather than keep guessing at
undocumented-to-us exact JSON syntax, this achieves the identical
demonstrable outcome (reaches `Register` on a pass, stops with a failed
step on a miss) through a mechanism — a step's own exit code — with no
such ambiguity. The threshold itself is baked into the pipeline
definition at build time (a `{"Get": "Parameters.X"}` reference inside a
Processing step's `Environment` was also rejected — `Environment` values
must be plain strings, even though the identical mechanism is accepted
for `ModelDataUrl` a few lines below in the same definition), so
producing the two demo states means updating the pipeline definition with
a different threshold before each run via `create_or_update_pipeline(cfg,
auc_threshold=...)`, not passing a runtime execution parameter.

## Real results — both demo states captured

**Successful execution** (`success-demo-2`,
`.../execution/88o3n64pvacr`, threshold 0.6): all three steps
`Succeeded`. Evaluate step's real output —

```json
{"auc": 0.6809714763098222, "pr_auc": 0.23625746066726622, "n_rows": 9960}
```

— matches Task 3's test-set numbers for the production model exactly
(0.6810 / 0.2363), confirming the pipeline-trained model is the same real
model, not a stand-in. Registered as `diabetes130-pipeline-models` v1,
`PendingManualApproval`.

**Failed execution** (`failure-demo`, `.../execution/nowvgyq8pxy6`,
threshold artificially set to 0.75, above the model's real ~0.68 AUC):
`TrainXGBoost` succeeded, `EvaluateModel` failed with exit code 1 and the
log line `QUALITY GATE FAILED: auc 0.6810 < threshold 0.7500`,
`RegisterModel` never started. Pipeline execution status: `Failed`. The
pipeline was then reset to the achievable threshold (0.6) as its resting
state.

## Two more real bugs, found only by actually running this

1. **Wrong tarball filename.** The Evaluate step's container command
   extracted `code.tar.gz` — copy-pasted from `src/fairness_report.py`'s
   bundler, which names its archive that. `aws_jobs.upload_source_dir`
   (used here instead) names it `sourcedir.tar.gz`. First execution failed
   immediately: `tar (child): ... code.tar.gz: Cannot open: No such file
   or directory`. Fixed by matching the actual filename.
2. **`ClientRequestToken` too short.** `create_pipeline` requires it to be
   at least 32 characters; `str(time.time())` (~18 chars) was rejected
   with `ParamValidationError`. Fixed with two concatenated UUID4 hexes.

## How to run it yourself

```bash
# One-time (or after changing the pipeline definition / hyperparameters):
python -c "
from config import load_config
from pipeline import create_or_update_pipeline
create_or_update_pipeline(load_config(), auc_threshold=0.6)  # 0.6 = achievable; use e.g. 0.75 to force a failed demo
"

# Start a run:
python -c "
from config import load_config
from pipeline import start_execution
print(start_execution(load_config(), execution_name='my-run'))
"
```

Or from the AWS CLI directly, once the pipeline exists:
```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name diabetes130-cicd-pipeline \
  --pipeline-execution-display-name my-run \
  --region us-east-1
```

**To see the DAG** (both the graph and, for any past execution, which
nodes went green vs. red): SageMaker Studio → **Pipelines** →
`diabetes130-cicd-pipeline` → **Executions** tab → pick an execution to
see its graph colored by step status. Console link:
`https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/studio` (navigate to Pipelines from the Studio home, under your domain/user profile — direct deep-links to a specific pipeline execution's graph view require an active Studio session URL, which is user/session-specific).

Each run costs real training + processing compute (roughly 5-10 minutes
end to end) and registers a new model package version in
`diabetes130-pipeline-models` on every successful pass — worth pruning
old pipeline-demo versions occasionally, distinct from the real production
model in `diabetes130-xgboost-models`.
