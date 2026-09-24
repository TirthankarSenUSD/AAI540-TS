"""SageMaker script-mode entry point for the Task 1b heuristic benchmark.

Runs inside the SKLearn framework container. Trains a 2-feature logistic
regression on `number_inpatient` / `number_emergency` (the continuous-score
half of the benchmark) and records both its metrics and the parameter-free
heuristic rule's metrics (`number_inpatient >= 1`) to `metrics.json` in the
model directory, so the registry can attach real validation numbers to this
model package.

This benchmark is deliberately trivial — the point of running it through a
real SageMaker training job, model artifact, and registry entry is to prove
out that path on something that can't fail, before Task 2 does the same
with a model that can.

Expects two data channels, each a single CSV with a header:
  train.csv / validation.csv, columns: number_inpatient, number_emergency, readmit_30
"""
import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

FEATURES = ["number_inpatient", "number_emergency"]
TARGET = "readmit_30"


def heuristic_metrics(df: pd.DataFrame) -> dict:
    y = df[TARGET]
    y_pred = (df["number_inpatient"] >= 1).astype(int)
    return {
        "accuracy": float((y_pred == y).mean()),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_pred)),
    }


def logistic_metrics(model: LogisticRegression, df: pd.DataFrame) -> dict:
    y = df[TARGET]
    scores = model.predict_proba(df[FEATURES])[:, 1]
    y_pred = (scores >= 0.5).astype(int)
    return {
        "accuracy": float((y_pred == y).mean()),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "."))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "."))
    return parser.parse_args()


def main():
    args = parse_args()

    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.validation, "validation.csv"))

    model = LogisticRegression(max_iter=1000)
    model.fit(train_df[FEATURES], train_df[TARGET])

    metrics = {
        "heuristic_rule": {
            "train": heuristic_metrics(train_df),
            "validation": heuristic_metrics(val_df),
        },
        "logistic_benchmark": {
            "train": logistic_metrics(model, train_df),
            "validation": logistic_metrics(model, val_df),
        },
    }
    print(json.dumps(metrics, indent=2))
    # SageMaker Training job metrics are scraped from stdout via regex when a
    # MetricDefinitions config is attached to the Estimator (see src/benchmark.py).
    print(f"validation:pr_auc={metrics['logistic_benchmark']['validation']['pr_auc']}")
    print(f"validation:roc_auc={metrics['logistic_benchmark']['validation']['roc_auc']}")

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# --- SageMaker SKLearn container inference hooks (used if deployed/batch-transformed) ---

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type):
    if content_type == "text/csv":
        from io import StringIO
        return pd.read_csv(StringIO(request_body), header=None, names=FEATURES)
    raise ValueError(f"unsupported content type: {content_type}")


def predict_fn(input_data, model):
    return model.predict_proba(input_data[FEATURES])[:, 1]


def output_fn(prediction, accept):
    if accept == "text/csv":
        return "\n".join(str(p) for p in prediction), accept
    raise ValueError(f"unsupported accept type: {accept}")


if __name__ == "__main__":
    main()
