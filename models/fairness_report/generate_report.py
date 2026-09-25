"""Custom fairness + explainability report -- runs as a SageMaker
Processing Job.

Stands in for SageMaker Clarify's bias/explainability reports, which are
in maintenance mode and unavailable to this AWS account
("SageMaker Clarify processing is in maintenance mode and is not
available to new customers" -- a real ValidationException hit when
actually launching a Clarify processing job for this project). This
produces the same substance -- pre/post-training bias metrics on `race`,
per-subgroup accuracy/selection-rate, and global feature importance via
XGBoost's native SHAP contributions -- as a real Processing Job output
in S3, just not Clarify-branded.

Expects src/ and models/xgboost/ bundled alongside this script under
/opt/ml/processing/input/code (see src/fairness_report.py's launcher).
"""
import json
import os
import sys

CODE_DIR = "/opt/ml/processing/input/code"
sys.path.insert(0, os.path.join(CODE_DIR, "src"))
sys.path.insert(0, os.path.join(CODE_DIR, "models", "xgboost"))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_DIR = os.path.join(CODE_DIR, "deployed_model")
OUTPUT_DIR = "/opt/ml/processing/output"


def disparate_impact(selection_rate_minority: float, selection_rate_majority: float) -> float:
    if selection_rate_majority == 0:
        return float("nan")
    return selection_rate_minority / selection_rate_majority


def bias_metrics(df: pd.DataFrame, facet_col: str, facet_value: str, label_col: str, pred_col: str) -> dict:
    """Pre-training bias (class imbalance, difference in label proportions)
    and post-training bias (disparate impact, accuracy/recall difference),
    facet_value vs. everyone else -- the same substance Clarify's
    pretrain_bias/posttrain_bias metrics report."""
    facet_mask = df[facet_col] == facet_value
    n_facet, n_other = facet_mask.sum(), (~facet_mask).sum()

    label_rate_facet = df.loc[facet_mask, label_col].mean()
    label_rate_other = df.loc[~facet_mask, label_col].mean()

    sel_rate_facet = df.loc[facet_mask, pred_col].mean()
    sel_rate_other = df.loc[~facet_mask, pred_col].mean()

    acc_facet = (df.loc[facet_mask, pred_col] == df.loc[facet_mask, label_col]).mean()
    acc_other = (df.loc[~facet_mask, pred_col] == df.loc[~facet_mask, label_col]).mean()

    tp_facet = ((df.loc[facet_mask, pred_col] == 1) & (df.loc[facet_mask, label_col] == 1)).sum()
    pos_facet = (df.loc[facet_mask, label_col] == 1).sum()
    tp_other = ((df.loc[~facet_mask, pred_col] == 1) & (df.loc[~facet_mask, label_col] == 1)).sum()
    pos_other = (df.loc[~facet_mask, label_col] == 1).sum()
    recall_facet = tp_facet / pos_facet if pos_facet else float("nan")
    recall_other = tp_other / pos_other if pos_other else float("nan")

    return {
        "facet": f"{facet_col}={facet_value}",
        "n_facet": int(n_facet), "n_other": int(n_other),
        "pretraining": {
            "class_imbalance": float((n_other - n_facet) / (n_facet + n_other)),
            "difference_in_label_proportion": float(label_rate_facet - label_rate_other),
        },
        "posttraining": {
            "disparate_impact": float(disparate_impact(sel_rate_facet, sel_rate_other)),
            "difference_in_selection_rate": float(sel_rate_facet - sel_rate_other),
            "accuracy_difference": float(acc_facet - acc_other),
            "recall_difference": float(recall_facet - recall_other),
            "selection_rate_facet": float(sel_rate_facet),
            "selection_rate_other": float(sel_rate_other),
            "accuracy_facet": float(acc_facet),
            "accuracy_other": float(acc_other),
            "recall_facet": float(recall_facet),
            "recall_other": float(recall_other),
        },
    }


def feature_importance(preprocessor, booster: xgb.XGBClassifier, X: pd.DataFrame, sample_n: int = 5000) -> list:
    """Global feature importance via XGBoost's native SHAP contributions
    (pred_contribs=True) -- mathematically the same TreeSHAP values the
    `shap` library / Clarify's explainability report would produce."""
    X_sample = X.sample(n=min(sample_n, len(X)), random_state=42)
    X_t = preprocessor.transform(X_sample)
    dmat = xgb.DMatrix(X_t, enable_categorical=True)
    contribs = booster.get_booster().predict(dmat, pred_contribs=True)

    feature_names = list(X_t.columns) + ["bias"]
    mean_abs = np.abs(contribs).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [{"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])} for i in order]


def main():
    from model_data import features_and_target, load_split

    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
    calibrator = joblib.load(os.path.join(MODEL_DIR, "calibrator.joblib"))
    booster = xgb.XGBClassifier()
    booster.load_model(os.path.join(MODEL_DIR, "model.ubj"))
    threshold = json.loads(open(os.path.join(MODEL_DIR, "threshold.json")).read())["threshold"]

    prod_df = load_split("production")
    X_prod, y_prod = features_and_target(prod_df)

    X_t = preprocessor.transform(X_prod)
    raw_scores = booster.predict_proba(X_t)[:, 1]
    calibrated = calibrator.predict_proba(raw_scores)[:, 1]
    prediction = (calibrated >= threshold).astype(int)

    scored = pd.DataFrame({
        "race": X_prod["race"].astype(object).where(X_prod["race"].notna(), "Missing").reset_index(drop=True),
        "gender": X_prod["gender"].reset_index(drop=True),
        "label": y_prod.reset_index(drop=True),
        "prediction": prediction,
        "probability": calibrated,
    })

    report = {
        "dataset": "production split (Week 3 decision #5's reserved monitoring/drift dataset)",
        "n_rows": len(scored),
        "bias_by_race": [
            bias_metrics(scored, "race", value, "label", "prediction")
            for value in ["AfricanAmerican", "Caucasian"]
            if (scored["race"] == value).sum() >= 30
        ],
        "bias_by_gender": [
            bias_metrics(scored, "gender", "Female", "label", "prediction")
        ] if (scored["gender"] == "Female").sum() >= 30 else [],
        "feature_importance": feature_importance(preprocessor, booster, X_prod),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "fairness_explainability_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
