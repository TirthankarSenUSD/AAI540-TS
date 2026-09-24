"""Task 3 — evaluation.

Run on the **test** split, touched once. Validation is for tuning/
calibration/threshold selection only.

Includes the full-feature logistic regression baseline (one-hot +
standardization, per Task 2's preprocessing note) that anchors the
"Logistic regression" row of the comparison table — a separate thing from
Task 1b's 2-feature heuristic benchmark.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS_XGBOOST_DIR = Path(__file__).resolve().parent.parent / "models" / "xgboost"
SUBGROUP_COLUMNS = ["race", "gender", "age_ordinal"]


# ---------------------------------------------------------------------------
# Full-feature logistic regression baseline (comparison-table row)
# ---------------------------------------------------------------------------

def build_logistic_pipeline(X_train: pd.DataFrame) -> Pipeline:
    categorical_cols = [c for c in X_train.columns if X_train[c].dtype in ("object",) or str(X_train[c].dtype) == "string"]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight=None)),
    ])


def fit_logistic_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    # X may contain pandas nullable Int64 / string dtypes with <NA>; sklearn's
    # OneHotEncoder/StandardScaler want plain numpy-compatible objects.
    X_train = _to_sklearn_friendly(X_train)
    pipe = build_logistic_pipeline(X_train)
    pipe.fit(X_train, y_train)
    return pipe


def _to_sklearn_friendly(X: pd.DataFrame) -> pd.DataFrame:
    """sklearn's OneHotEncoder/StandardScaler don't handle pandas nullable
    dtypes or a mix of `pd.NA` and str in one column — fill categorical
    nulls with an explicit sentinel and cast numerics to plain float64."""
    X = X.copy()
    for col in X.columns:
        if str(X[col].dtype) == "string":
            X[col] = X[col].astype(object)
            X[col] = X[col].where(X[col].notna(), "Missing")
        elif str(X[col].dtype) == "Int64":
            X[col] = X[col].astype("float64")
    return X


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_fraction: float) -> float:
    n = len(scores)
    top_k = max(1, int(np.ceil(n * k_fraction)))
    order = np.argsort(-scores)
    top_idx = order[:top_k]
    return float(np.mean(np.asarray(y_true)[top_idx]))


def select_threshold_by_alert_volume(val_scores: np.ndarray, target_alert_fraction: float) -> float:
    """Fix the alert volume to the care team's capacity: the threshold is
    the score at the (1 - target_alert_fraction) quantile of validation
    scores, so exactly ~target_alert_fraction of the batch is flagged."""
    return float(np.quantile(val_scores, 1 - target_alert_fraction))


def full_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "precision_at_5pct": precision_at_k(y_true, scores, 0.05),
        "precision_at_10pct": precision_at_k(y_true, scores, 0.10),
        "precision_at_20pct": precision_at_k(y_true, scores, 0.20),
        "base_rate": float(np.mean(y_true)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
    }


def calibration_curve_data(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> dict:
    frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=n_bins, strategy="uniform")
    return {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()}


# ---------------------------------------------------------------------------
# Subgroup evaluation
# ---------------------------------------------------------------------------

def subgroup_metrics(df: pd.DataFrame, y_true: np.ndarray, scores: np.ndarray, threshold: float,
                      subgroup_cols: list[str] = SUBGROUP_COLUMNS, min_group_size: int = 30) -> dict:
    results = {}
    y_pred = (scores >= threshold).astype(int)
    for col in subgroup_cols:
        group_col = df[col]
        if str(group_col.dtype) == "string":
            # pandas' groupby(dropna=False) chokes on a nullable `string`
            # dtype containing actual nulls — fill with a sentinel first.
            group_col = group_col.astype(object).where(group_col.notna(), "Missing")
        group_results = {}
        for group_value, idx in df.groupby(group_col, dropna=False).groups.items():
            idx = np.asarray(idx)
            if len(idx) < min_group_size:
                continue
            yt = np.asarray(y_true)[idx]
            yp = y_pred[idx]
            sc = scores[idx]
            group_results[str(group_value)] = {
                "n": int(len(idx)),
                "base_rate": float(yt.mean()),
                "selection_rate": float(yp.mean()),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "brier_score": float(brier_score_loss(yt, sc)) if len(set(yt)) > 1 else None,
            }
        results[col] = group_results
    return results


def race_ablation_report(X_train, y_train, X_test, y_test, df_test: pd.DataFrame, threshold_fraction: float) -> dict:
    """Does dropping `race` from the feature set reduce disparity, or does a
    proxy (payer_code, prior utilization) just hide it? Trains two quick
    default-hyperparameter XGBoost models (with/without race) and compares
    subgroup selection rates."""
    sys.path.insert(0, str(MODELS_XGBOOST_DIR))
    import xgboost as xgb

    from preprocess import XGBoostCategoricalPreprocessor

    def fit_and_score(X_tr, X_te):
        prep = XGBoostCategoricalPreprocessor().fit(X_tr)
        model = xgb.XGBClassifier(
            max_depth=6, eta=0.1, subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            enable_categorical=True, n_estimators=200,
        )
        model.fit(prep.transform(X_tr), y_train)
        return model.predict_proba(prep.transform(X_te))[:, 1]

    scores_with_race = fit_and_score(X_train, X_test)
    scores_without_race = fit_and_score(X_train.drop(columns=["race"]), X_test.drop(columns=["race"]))

    thr_with = select_threshold_by_alert_volume(scores_with_race, threshold_fraction)
    thr_without = select_threshold_by_alert_volume(scores_without_race, threshold_fraction)

    return {
        "with_race": {
            "test_pr_auc": float(average_precision_score(y_test, scores_with_race)),
            "selection_rate_by_race": subgroup_metrics(df_test, y_test, scores_with_race, thr_with, ["race"]),
        },
        "without_race": {
            "test_pr_auc": float(average_precision_score(y_test, scores_without_race)),
            "selection_rate_by_race": subgroup_metrics(df_test, y_test, scores_without_race, thr_without, ["race"]),
        },
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(rows: dict[str, dict]) -> pd.DataFrame:
    """rows: {model_name: metrics_dict from full_metrics()}"""
    records = []
    for name, m in rows.items():
        records.append({
            "Model": name,
            "PR-AUC": round(m["pr_auc"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "Precision@10%": round(m["precision_at_10pct"], 4),
            "Recall@thr": round(m["recall_at_threshold"], 4),
            "Brier": round(m["brier_score"], 4),
        })
    return pd.DataFrame.from_records(records)


def save_comparison_table(table: pd.DataFrame, reports_dir: Path) -> None:
    reports_dir.mkdir(exist_ok=True)
    table.to_csv(reports_dir / "model_comparison.csv", index=False)
    with open(reports_dir / "model_comparison.md", "w") as f:
        f.write("# Model comparison — test split\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n")


def main(xgboost_model_dir: str, target_alert_fraction: float = 0.10) -> None:
    """Run the full Task 3 pipeline. `xgboost_model_dir` should hold
    model.ubj, preprocessor.joblib, calibrator.joblib (from
    models/xgboost/train.py + src/calibrate.py output).
    """
    import json

    import joblib
    import matplotlib.pyplot as plt
    import xgboost as xgb

    from benchmark import (
        fit_logistic_benchmark,
        heuristic_predict,
        logistic_benchmark_metrics,
    )
    from model_data import features_and_target, load_split

    repo_root = Path(__file__).resolve().parent.parent
    reports_dir = repo_root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_split("train"), load_split("validation"), load_split("test")
    X_train, y_train = features_and_target(train_df)
    X_val, y_val = features_and_target(val_df)
    X_test, y_test = features_and_target(test_df)

    def threshold_and_metrics(val_scores, test_scores, y_val_, y_test_):
        thr = select_threshold_by_alert_volume(np.asarray(val_scores), target_alert_fraction)
        return full_metrics(np.asarray(y_test_), np.asarray(test_scores), thr)

    comparison_rows = {}

    # Majority class: constant score = base rate everywhere.
    const_val = np.full(len(y_val), y_train.mean())
    const_test = np.full(len(y_test), y_train.mean())
    comparison_rows["Majority class"] = threshold_and_metrics(const_val, const_test, y_val, y_test)

    # Heuristic rule: binary 0/1 "score".
    heur_val = heuristic_predict(val_df).astype(float)
    heur_test = heuristic_predict(test_df).astype(float)
    comparison_rows["Heuristic (number_inpatient>=1)"] = threshold_and_metrics(heur_val, heur_test, y_val, y_test)

    # Full-feature logistic regression.
    logreg_pipe = fit_logistic_pipeline(X_train, y_train)
    logreg_val = logreg_pipe.predict_proba(_to_sklearn_friendly(X_val))[:, 1]
    logreg_test = logreg_pipe.predict_proba(_to_sklearn_friendly(X_test))[:, 1]
    comparison_rows["Logistic regression"] = threshold_and_metrics(logreg_val, logreg_test, y_val, y_test)

    # XGBoost — uncalibrated and calibrated.
    sys.path.insert(0, str(MODELS_XGBOOST_DIR))
    model_dir = Path(xgboost_model_dir)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(model_dir / "model.ubj"))
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    calibrator = joblib.load(model_dir / "calibrator.joblib")

    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    xgb_val = xgb_model.predict_proba(X_val_t)[:, 1]
    xgb_test = xgb_model.predict_proba(X_test_t)[:, 1]
    comparison_rows["XGBoost (uncalibrated)"] = threshold_and_metrics(xgb_val, xgb_test, y_val, y_test)

    cal_val = calibrator.predict_proba(xgb_val)[:, 1]
    cal_test = calibrator.predict_proba(xgb_test)[:, 1]
    comparison_rows["XGBoost (calibrated)"] = threshold_and_metrics(cal_val, cal_test, y_val, y_test)

    table = build_comparison_table(comparison_rows)
    print(table.to_string(index=False))
    save_comparison_table(table, reports_dir)
    (reports_dir / "task3_full_metrics.json").write_text(json.dumps(comparison_rows, indent=2))

    # Subgroup evaluation for the final (calibrated) model.
    final_threshold = select_threshold_by_alert_volume(cal_val, target_alert_fraction)
    subgroups = subgroup_metrics(test_df, np.asarray(y_test), cal_test, final_threshold)
    (reports_dir / "task3_subgroup_metrics.json").write_text(json.dumps(subgroups, indent=2))

    # Calibration curve, 10 bins, plotted.
    curve = calibration_curve_data(np.asarray(y_test), cal_test, n_bins=10)
    (reports_dir / "task3_calibration_curve.json").write_text(json.dumps(curve, indent=2))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(curve["mean_predicted"], curve["fraction_positive"], "o-", label="XGBoost (calibrated)")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve (test, 10 bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "07_calibration_curve.png")

    # Race ablation.
    ablation = race_ablation_report(X_train, y_train, X_test, y_test, test_df, target_alert_fraction)
    (reports_dir / "task3_race_ablation.json").write_text(json.dumps(ablation, indent=2))

    print(f"\nfrozen threshold @ {target_alert_fraction:.0%} alert volume: {final_threshold:.4f}")
    print("done — see reports/task3_*.json, reports/model_comparison.{md,csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--xgboost-model-dir", required=True)
    parser.add_argument("--target-alert-fraction", type=float, default=0.10)
    args = parser.parse_args()
    main(args.xgboost_model_dir, args.target_alert_fraction)
