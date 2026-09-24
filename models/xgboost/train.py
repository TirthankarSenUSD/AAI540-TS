"""SageMaker XGBoost framework container, script mode — Task 2's real model.

Reads train/validation channels as CSV — the SageMaker XGBoost training
toolkit does not auto-install a source-dir requirements.txt the way some
other framework containers do, so Parquet (which needs pyarrow) isn't
usable here without a lot more image-customization than this warrants; the
Week 3 splits are exported to CSV once (see src/hpo.py's
`export_split_to_csv`) rather than fighting the container.

Fits the categorical preprocessor on train only, trains an XGBoost
classifier with early stopping on validation, and saves the artifact bundle
(minus calibrator/threshold, which are fit downstream in src/calibrate.py
and src/evaluate.py once this job's output is chosen by HPO).

Objective metric for Automatic Model Tuning: `validation:aucpr`, printed to
stdout in SageMaker's regex-scrapeable metric format.
"""
import argparse
import json
import os

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from preprocess import XGBoostCategoricalPreprocessor

TARGET = "readmit_30"
EXCLUDED_COLUMNS = [
    "encounter_id", "patient_nbr", "split",
    "event_time", "write_time", "api_invocation_time", "is_deleted",
]


def load_split(channel_dir: str) -> pd.DataFrame:
    # A channel directory may contain one or more CSV part-files.
    files = [f for f in os.listdir(channel_dir) if f.endswith(".csv")]
    frames = [pd.read_csv(os.path.join(channel_dir, f)) for f in files]
    return pd.concat(frames, ignore_index=True)


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [c for c in EXCLUDED_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols + [TARGET])
    y = df[TARGET].astype(int)
    leaked = [c for c in X.columns if c in EXCLUDED_COLUMNS or c == TARGET]
    assert not leaked, f"identifier/target columns leaked into feature list: {leaked}"
    return X, y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "."))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "."))

    # HPO-tunable hyperparameters — ranges per the brief's Task 2 table.
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    # SageMaker HPO's tunable-hyperparameter whitelist for this image uses
    # the built-in algorithm's name "lambda", not xgboost's sklearn-API
    # name "reg_lambda" — accept the CLI flag SageMaker actually sends.
    parser.add_argument("--lambda", dest="reg_lambda", type=float, default=1.0)
    parser.add_argument("--num_round", type=int, default=500)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)
    parser.add_argument("--scale_pos_weight", type=float, default=1.0)
    return parser.parse_args()


def build_debugger_hook():
    """Best-effort SageMaker Debugger hook — Overfit / LossNotDecreasing /
    ClassImbalance rules are configured on the launcher side (boto3
    DebugRuleConfigurations); this just needs to emit the tensors they
    analyze. Falls back to no hook if smdebug isn't available (e.g. running
    this script locally for a smoke test, outside the SageMaker container).
    """
    try:
        from smdebug.xgboost import Hook
        return Hook.create_from_json_file()
    except Exception:
        return None


def main():
    args = parse_args()

    train_df = load_split(args.train)
    val_df = load_split(args.validation)
    X_train, y_train = features_and_target(train_df)
    X_val, y_val = features_and_target(val_df)

    preprocessor = XGBoostCategoricalPreprocessor().fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.reg_lambda,
        "scale_pos_weight": args.scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "enable_categorical": True,
    }

    model = xgb.XGBClassifier(
        n_estimators=args.num_round,
        early_stopping_rounds=args.early_stopping_rounds,
        **params,
    )

    hook = build_debugger_hook()
    fit_kwargs = {}
    if hook is not None:
        hook.set_mode(mode=hook.mode.TRAIN)  # type: ignore[union-attr]
        fit_kwargs["callbacks"] = [hook]

    model.fit(
        X_train_t, y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False,
        **fit_kwargs,
    )

    val_scores = model.predict_proba(X_val_t)[:, 1]
    val_pr_auc = average_precision_score(y_val, val_scores)

    print(f"best_iteration={model.best_iteration}")
    print(f"validation:aucpr={val_pr_auc}")

    model.get_booster().save_model(os.path.join(args.model_dir, "model.ubj"))
    joblib.dump(preprocessor, os.path.join(args.model_dir, "preprocessor.joblib"))

    feature_schema = {
        "categorical_columns": preprocessor.categorical_columns_,
        "numeric_columns": preprocessor.numeric_columns_,
        "feature_order": preprocessor.feature_names_,
    }
    with open(os.path.join(args.model_dir, "feature_schema.json"), "w") as f:
        json.dump(feature_schema, f, indent=2)

    with open(os.path.join(args.model_dir, "training_metrics.json"), "w") as f:
        json.dump({
            "validation_pr_auc": float(val_pr_auc),
            "best_iteration": int(model.best_iteration),
            "hyperparameters": params | {"num_round": args.num_round},
        }, f, indent=2)


if __name__ == "__main__":
    main()
