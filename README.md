# Diabetes 130 — 30-day readmission

Data layer for an ML system that predicts whether a hospital encounter will
be followed by a readmission within 30 days, built on the UCI
["Diabetes 130-US hospitals" dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
This covers ingestion through to a split, Feature Store-backed dataset.
Model training is out of scope for this stage — see
[`Week 3 Checklist.md`](Week%203%20Checklist.md) for the full implementation
brief this repo follows.

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
5. **Expected model performance is ~0.67 ROC-AUC.**

## Repo layout

```
.
├── config.yaml              # single source of truth for bucket/paths — no hardcoded values in scripts
├── requirements.txt
├── src/
│   ├── config.py             # loads config.yaml
│   ├── ingest.py              # Task 1 — raw CSVs -> S3 datalake + provenance manifest
│   ├── catalog.py             # Task 2 — Glue database + Athena tables over raw data
│   ├── transform.py           # Task 4 — cleaning + feature engineering (pure functions, unit-tested)
│   ├── feature_store.py       # Task 5 — SageMaker Feature Store ingestion (offline store)
│   └── split.py                # Task 6 — four-way patient-level split
├── notebooks/
│   └── 01_eda.ipynb           # Task 3 — EDA, read entirely through Athena
├── tests/
│   └── test_transform.py      # unit tests for transform.py and split.py, no AWS credentials required
├── reports/
│   ├── figures/                # every figure from the EDA notebook, saved to disk
│   ├── task2_validation.json
│   ├── task6_split.json
│   └── task5_feature_store.json
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
```

Each script writes its validation output to `reports/` so results are
reviewable without re-running anything.

## Results

See [`RESULTS.md`](RESULTS.md) for every count this brief said to expect,
alongside what was actually observed.
