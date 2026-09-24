# Reference: Smart_Grid-main folder structure

This documents the folder structure of `Smart_Grid-main`, a separate/unrelated
prior AAI-540 project (AMI smart-meter load forecasting). It is kept here only
as a structural reference for organizing this repo (`AAI540-TS`, the diabetes
130 readmission project) — none of its code or data applies to this project.

## Layout

```
Smart_Grid-main/
├── README.md                          # Project overview / problem statement
├── .gitignore
│
├── 1_Data_Exploration.ipynb           # EDA on raw AMI + weather + holiday data
├── 2_Load_Processed_Data.ipynb        # Load cleaned data for downstream steps
├── 3_Create_Athena_Database_fp.ipynb  # Create Glue/Athena database
├── 4_Register_CSV_with_Athena.ipynb   # Register raw CSVs as Athena tables
├── 5_Convert_S3_csv_to_parquet.ipynb  # CSV -> Parquet conversion in S3
├── 6_Create_Feature_Store.ipynb       # SageMaker Feature Store ingestion
├── 7a_XGBT_model_training.ipynb       # XGBoost model training
├── 7b_LGBM_model_training.ipynb       # LightGBM model training (alt model)
├── 8_Model_Monitoring_Normal.ipynb    # SageMaker Model Monitor — baseline
├── 8a_Model_Monitoring_Alarm.ipynb    # Model Monitor — drift/alarm scenario
├── 9_CI-CD_Pipelines.ipynb            # CodePipeline/CodeBuild CI-CD setup
│
├── pipe_tools.py                      # Shared helpers for the CI/CD notebook
│                                       #   (SageMaker image/model URIs, Feature
│                                       #    Group helpers, pipeline utilities)
│
├── code/
│   └── preprocessor.py                # Model Monitor preprocessing handler
│                                       #   (flattens endpoint output + ground
│                                       #    truth records for monitoring jobs)
│
├── model/
│   └── model.tar.gz                   # Packaged trained model artifact
│
├── test_data/
│   ├── batch_data.csv                 # Batch transform / scoring input
│   ├── validation_data.csv
│   └── validation_with_predictions.csv
│
├── train_data.csv                     # Top-level copies of the train/val/test
├── validation_data.csv                #   splits (also duplicated under
├── test_data.csv                      #   test_data/) used directly by the
├── batch_data.csv                     #   notebooks
│
└── WebApp/                            # Demo front end for serving predictions
    ├── app.py                         # FastAPI backend (calls SageMaker
    │                                   #   endpoint, serves predictions/data)
    ├── modified_test_data.csv         # Sample data for the demo UI
    └── frontend/                      # React app (create-react-app layout)
        ├── package.json / package-lock.json
        ├── public/                    # Static assets, index.html, manifest
        └── src/
            ├── App.js / App.css / App.test.js
            ├── index.js / index.css
            ├── components/
            │   └── chart.js           # Chart component for predictions
            └── reportWebVitals.js / setupTests.js / logo.svg
```

## Structural pattern worth reusing

The notebooks are numbered in pipeline order and map 1:1 onto pipeline
stages, each stage owning its own notebook rather than one monolithic
notebook:

1. Raw data exploration
2. Load/clean processed data
3–4. Glue/Athena database + table registration over raw data
5. Format conversion (CSV → Parquet) for query efficiency
6. Feature Store ingestion
7. Model training (one notebook per candidate algorithm, `a`/`b` suffixes)
8. Model monitoring (baseline + alarm/drift scenario, `a` suffix)
9. CI/CD pipeline setup

Supporting code lives outside the notebooks in small, single-purpose modules
(`pipe_tools.py`, `code/preprocessor.py`) rather than inline in notebook
cells, and a `WebApp/` directory holds an optional demo front end (FastAPI +
React) that is decoupled from the modeling pipeline.

## Where this project (AAI540-TS) diverges

The Week 3 Checklist for this project specifies a different, more
script-driven layout — see [`Week 3 Checklist.md`](../Week%203%20Checklist.md):

```
.
├── config.yaml
├── requirements.txt
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── catalog.py
│   ├── transform.py
│   ├── feature_store.py
│   └── split.py
├── notebooks/
│   └── 01_eda.ipynb
└── tests/
    └── test_transform.py
```

Key differences from Smart_Grid-main to carry forward intentionally:
- Pipeline logic lives in testable `src/*.py` modules, not directly in
  notebooks. Only EDA stays as a notebook.
- Configuration is centralized in `config.yaml` (no hardcoded bucket names),
  where Smart_Grid-main hardcodes buckets/paths inline in notebooks.
- A `tests/` directory with real unit tests is required; Smart_Grid-main has
  none.
