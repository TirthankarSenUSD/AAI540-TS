# Diabetes 130 — 30-day readmission

An ML system that predicts whether a hospital encounter will be followed by
a readmission within 30 days, built on the UCI
["Diabetes 130-US hospitals" dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
[`Week 3 Checklist.md`](Week%203%20Checklist.md) covers ingestion through a
split, Feature Store-backed dataset. [`Week 4 Checklist.md`](Week%204%20Checklist.md)
covers benchmark + real model training, calibration, evaluation, and Batch
Transform deployment.

## Decisions already made

1. **Target is binary**: `readmit_30 = (readmitted == "<30")`.
2. **No deduplication to one encounter per patient.** Every encounter is
   kept; leakage is handled by splitting on patient instead. See the EDA
   notebook's encounters-per-patient analysis for why: 80.9% of positive
   encounters come from patients with more than one visit.
3. **Every split is a patient-level group split.** No random row split
   anywhere.
4. **Expired and hospice discharges are removed** before anything else —
   see the note in [`src/transform.py`](src/transform.py) on the corrected
   discharge-disposition ID set (`{11, 13, 14, 19, 20, 21}`).
5. **Expected model performance is ~0.67 ROC-AUC.** A materially higher
   number on the real (HPO-tuned) model means leakage, not success.
6. **No resampling, no deep learning, accuracy is not a metric.** PR-AUC
   and precision@k are primary; boosted trees, not neural nets, per prior
   published work on this exact dataset.
7. **The production split is untouched** — reserved for monitoring/drift
   work in a later module.

## Repo layout

```
.
├── config.yaml              # single source of truth for bucket/paths — no hardcoded values in scripts
├── requirements.txt
├── src/
│   ├── config.py             # loads config.yaml
│   ├── ingest.py              # Week 3 Task 1 — raw CSVs -> S3 datalake + provenance manifest
│   ├── catalog.py             # Week 3 Task 2 — Glue database + Athena tables over raw data
│   ├── transform.py           # Week 3 Task 4 — cleaning + feature engineering (pure functions, unit-tested)
│   ├── feature_store.py       # Week 3 Task 5 — SageMaker Feature Store ingestion (offline store)
│   ├── split.py                # Week 3 Task 6 — four-way patient-level split
│   ├── model_data.py          # shared split-loading + feature-exclusion helper
│   ├── benchmark.py            # Week 4 Task 1 — majority-class floor + heuristic/logistic benchmark
│   ├── calibrate.py            # Week 4 Task 2 — Platt scaling on the validation split
│   ├── evaluate.py             # Week 4 Task 3 — metrics, precision@k, subgroup fairness, comparison table
│   ├── deploy.py               # Week 4 Task 4 — artifact bundling, Model Registry, Batch Transform, smoke test
│   ├── hpo.py                  # Week 4 Task 2 — Bayesian HPO launcher (boto3)
│   └── aws_jobs.py             # shared boto3 helpers for training/tuning job launches
├── models/
│   ├── benchmark_sklearn/train.py   # SageMaker SKLearn script-mode entry point (Task 1b)
│   └── xgboost/                      # SageMaker XGBoost script-mode entry point (Task 2)
│       ├── train.py, preprocess.py, inference.py, calibration.py
├── notebooks/
│   └── 01_eda.ipynb           # Week 3 Task 3 — EDA, read entirely through Athena
├── tests/
│   └── test_transform.py      # unit tests, no AWS credentials required
├── reports/
│   ├── figures/                # every figure from the EDA notebook + evaluation, saved to disk
│   ├── model_comparison.{md,csv}    # the Week 4 deliverable comparison table
│   └── task*.json              # validation output from every task, both weeks
├── docs/
│   └── reference_smart_grid_structure.md   # folder-structure reference from a prior, unrelated project
└── Datasets/                  # local staging for the 3 raw source files (gitignored — see below)
```

## Data

Raw source data is **not** committed to this repo. The pipeline downloads
nothing automatically — the three source files are staged locally under
`Datasets/` and `src/ingest.py` uploads them to S3, recording a SHA-256 +
row-count manifest as the provenance record:

| File | Origin |
|---|---|
| `diabetic_data.csv` | UCI dataset 296 zip |
| `IDS_mapping.csv` | same zip |
| `icd9dx2015.csv` | NBER ICD-9 provider diagnostic codes, 2015 |

```
s3://<bucket>/raw/encounters/diabetic_data.csv
s3://<bucket>/raw/id_mapping/IDS_mapping.csv
s3://<bucket>/raw/icd9/icd9dx2015.csv
s3://<bucket>/raw/_manifest.json          # checksums, row counts, ingestion timestamp
```

The bucket has versioning, SSE-KMS encryption, and public access blocked.
The raw zone is immutable — nothing downstream writes back to it.

## Pipeline

Run in order, each reading only from `config.yaml`:

```bash
pip install -r requirements.txt

python src/ingest.py         # Task 1 — S3 datalake + manifest
python src/catalog.py        # Task 2 — Glue/Athena tables + validation queries
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb  # Task 3
python -m pytest tests/      # Task 4 tests (transform.py is exercised directly by feature_store.py / split.py below)
python src/feature_store.py  # Task 5 — Feature Store ingestion (offline store)
python src/split.py          # Task 6 — four-way patient-level split

# Week 4 — model, evaluation, deployment
python src/benchmark.py               # Task 1a/1b — local benchmark metrics (see also launch_benchmark_training_job)
python src/hpo.py                     # Task 2 — launches real HPO (multi-job, costs real compute — run deliberately)
python src/calibrate.py --model-dir <best-job-artifact-dir>   # Task 2 — Platt scaling
python src/evaluate.py --xgboost-model-dir <dir-with-model+calibrator>  # Task 3 — full evaluation + comparison table
python src/deploy.py                  # Task 4 — bundle, register, Batch Transform (see module for individual steps)
```

Each script writes its validation output to `reports/` so results are
reviewable without re-running anything. Every Week 4 step above has been
run for real against AWS: a 25-trial Bayesian HPO job, a final training
run with SageMaker Debugger attached, real calibration, real Model
Registry entries, and a real Batch Transform job with a passing smoke
test (including the training/serving-skew regression check). Along the
way this surfaced five real bugs/API constraints that only showed up
under actual SageMaker execution — see `RESULTS.md` for the full list,
including one that genuinely broke a deployed Batch Transform job
(a calibrator pickle incompatible with the serving container's older
sklearn) before being caught and fixed.

## Results

See [`RESULTS.md`](RESULTS.md) for every count this brief said to expect,
alongside what was actually observed.
