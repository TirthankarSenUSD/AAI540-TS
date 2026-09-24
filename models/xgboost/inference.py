"""SageMaker inference hooks for the final bundled model.tar.gz.

Used by Batch Transform. Loads preprocessor + booster + calibrator and
emits a single calibrated probability per row — the frozen threshold is
deliberately **not** applied here. Per the brief's Batch Transform config,
the identifier column is stripped before reaching the model
(`input_filter="$[1:]"`) and rejoined by SageMaker afterward
(`join_source="Input"`, `output_filter="$[0,-1]"`), so this only ever sees
feature columns, in `feature_schema.json`'s `feature_order`, and only ever
emits a score. Sorting, thresholding, and flagging happen in
`src/deploy.py`'s post-processing step against the raw
identifier-plus-score output.
"""
import json
import os
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# So joblib.load(...) resolves these classes from this same file's
# directory when the model.tar.gz layout puts these modules alongside this
# script under code/.
import preprocess  # noqa: F401  (needed for unpickling XGBoostCategoricalPreprocessor)
import calibration  # noqa: F401  (needed for unpickling PlattCalibrator)


def model_fn(model_dir):
    with open(os.path.join(model_dir, "feature_schema.json")) as f:
        schema = json.load(f)

    booster_model = xgb.XGBClassifier()
    booster_model.load_model(os.path.join(model_dir, "model.ubj"))

    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    # calibrator.joblib is a PlattCalibrator (see calibration.py): just two
    # floats, no sklearn object — avoids any dev/serving sklearn-version
    # pickle skew for this artifact.
    calibrator = joblib.load(os.path.join(model_dir, "calibrator.joblib"))

    return {
        "model": booster_model,
        "preprocessor": preprocessor,
        "calibrator": calibrator,
        "schema": schema,
    }


def input_fn(request_body, content_type):
    if content_type != "text/csv":
        raise ValueError(f"unsupported content type: {content_type}")
    return request_body


def predict_fn(request_body, model_bundle):
    schema = model_bundle["schema"]
    feature_order = schema["feature_order"]

    df = pd.read_csv(StringIO(request_body), header=None, names=feature_order)

    for col in schema["categorical_columns"]:
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    for col in schema["numeric_columns"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X_t = model_bundle["preprocessor"].transform(df)
    raw_scores = model_bundle["model"].predict_proba(X_t)[:, 1]
    scores = model_bundle["calibrator"].predict_proba(raw_scores)[:, 1]
    return scores


# NOTE: `input_fn` is a pass-through (returns the raw CSV text) and all
# parsing happens in `predict_fn` because the number of rows in a Batch
# Transform mini-batch isn't known until parsed, and keeping parse+predict
# together avoids passing schema-dependent state through the toolkit twice.


def output_fn(prediction, accept):
    if accept != "text/csv":
        raise ValueError(f"unsupported accept type: {accept}")
    return "\n".join(f"{p:.6f}" for p in np.asarray(prediction)), accept
