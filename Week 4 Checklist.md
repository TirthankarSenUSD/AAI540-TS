# Implementation brief 2: model, evaluation, deployment

Follows on from `CLAUDE_CODE_BRIEF.md`. Assumes the Feature Store group is
populated and the four splits exist in S3.

---

## Decisions already made — do not revisit

1. **Expected performance is ~0.67 ROC-AUC.** Three independent published
   studies on this dataset and target converge there. A materially higher number
   means patient leakage, not success. Stop and investigate.
2. **No resampling.** No SMOTE, no undersampling, no class rebalancing. This is
   a ranking problem and resampling destroys the probability calibration it
   depends on. Use `scale_pos_weight` if weighting is needed, then recalibrate.
3. **Accuracy is not a metric here.** At ~11% prevalence a majority-class
   predictor scores ~89%. Primary metrics are PR-AUC and precision@k.
4. **No deep learning.** Published work on this exact dataset found neural nets
   the weakest method tried. Boosted trees on 100k rows of mixed tabular data.
5. **The production split is untouched.** It is reserved for monitoring and drift
   work in a later module. Do not train on it, evaluate on it, or peek at it.

---

## Task 1 — Benchmark model

The point is a defensible floor to measure against, not a good model. Build two.

### 1a. Trivial floor

Predict the majority class for everything. Report accuracy (~89%), recall (0),
and PR-AUC (≈ base rate). Three lines of code. It exists to make the
accuracy-is-useless argument concretely in the report rather than rhetorically.

### 1b. Heuristic benchmark — the real baseline

A single-feature rule on prior inpatient admissions:

```
predict positive if number_inpatient >= 1
```

Prior utilization is the strongest known signal in this dataset, so this is the
rule a hospital analyst would write without any ML at all. That makes it the
honest thing to beat. Also fit a logistic regression on just
`number_inpatient` and `number_emergency` to get a continuous score, so the
benchmark can be compared on ranking metrics and not only at a fixed threshold.

**Deploy 1b through SageMaker**, not as a local function. Use script mode with
the SKLearn container. It is trivial to train but it forces you to build the
train job, the model artifact, and the registry entry once on something simple,
before doing it with a model that can actually fail. Everything you learn here
applies to Task 2.

**Done when** the benchmark is a registered model package with recorded
validation metrics.

---

## Task 2 — The real model

### Use script mode, not the built-in algorithm

SageMaker's built-in XGBoost is simpler, but it can't do what this project
needs: calibration wrapping, custom evaluation metrics, and a preprocessor
bundled into the same artifact. Use the **SageMaker XGBoost framework
container in script mode** with your own `train.py`. You keep managed
infrastructure and get full control of the fit.

If you do use the built-in algorithm anyway, note its input contract: CSV with
**the target as the first column and no header row**. This trips up everyone
once.

### Preprocessing at model stage

`transform.py` from brief 1 produced a clean, typed, human-readable table. It
did not encode or scale, deliberately. That happens here, and it happens **fit
on training data only**.

- Categoricals: native categorical handling in XGBoost (`enable_categorical=True`),
  or ordinal encoding. Do not one-hot the high-cardinality columns.
- For the logistic baseline: one-hot plus standardization.
- Persist the fitted preprocessor with `joblib` into the same `model.tar.gz` as
  the model. Not beside it. Not in a separate bucket.

**Exclude from features:** `encounter_id`, `patient_nbr`, `split`, `event_time`,
and the Feature Store system columns `write_time`, `api_invocation_time`,
`is_deleted`. Assert the feature list does not contain any of these before
fitting — an identifier that leaks into training is the classic way to get a
suspiciously good model.

### Hyperparameter tuning

SageMaker Automatic Model Tuning, Bayesian strategy, objective metric
`validation:aucpr` (**not** `auc` — PR-AUC is what matters at this prevalence).

| Parameter | Range |
|---|---|
| `max_depth` | 3–8 |
| `eta` | 0.01–0.3, log scale |
| `subsample` | 0.6–1.0 |
| `colsample_bytree` | 0.6–1.0 |
| `min_child_weight` | 1–10 |
| `lambda` | 0–10 |
| `num_round` | 100–1000, with early stopping on validation |

20–30 trials is plenty. Do not run 200 — you are tuning into noise on a dataset
with a hard performance ceiling, and it burns budget.

### Calibration

Fit Platt scaling (`CalibratedClassifierCV` with `method="sigmoid"`, `cv="prefit"`)
on the **validation** split after the model is trained on train. Published work
on this dataset reports ROC-AUC improving from 0.664 to 0.688 through
calibration, so expect this to matter.

The calibrator is a separate object that must be serialized into the same
artifact bundle. A model shipped without its calibrator produces confidently
wrong probabilities.

### Debugging

Enable SageMaker Debugger with the built-in rules `Overfit`,
`LossNotDecreasing`, and `ClassImbalance`. They cost nothing and the report has
a debugging section to fill.

---

## Task 3 — Evaluation

Run on the **test** split. Validation is for tuning and calibration; test is
touched once at the end.

### Metrics

| Metric | Why |
|---|---|
| PR-AUC | Primary. Informative at 11% prevalence where ROC-AUC is flattering. |
| precision@k | The business metric. See below. |
| ROC-AUC | Secondary, for comparison against published ~0.67. |
| Brier score | Calibration quality. |
| Calibration curve | Plot it. Ten bins, predicted vs observed frequency. |
| Confusion matrix | At the chosen operating threshold only. |
| Recall at threshold | How many genuine readmissions the worklist catches. |

### precision@k

The care team can follow up with a fixed number of patients. Report precision at
k = top 5%, 10%, and 20% of the scored batch, and state the base rate (~11%)
next to each so lift is visible.

**Sanity check:** if precision@10% is not meaningfully above the base rate, the
model is not ranking and something is wrong. A model that scores 0.67 ROC-AUC
should show clear lift in the top decile.

### Threshold selection

Do not use 0.5. Choose the threshold on **validation** by fixing the alert
volume to the care team's capacity, then apply that frozen threshold to test.
The threshold is a deployment artifact: version it with the model.

### Subgroup evaluation

Compute precision, recall, and calibration separately across `race`, `gender`,
and `age_ordinal` bands, plus selection rate at the operating threshold. Report
the disparities found, not just the aggregate. This feeds the fairness section
of the design document and it is the part graders find thin.

Also test whether dropping `race` from the feature set reduces disparity or
merely hides it — `payer_code` and prior utilization are both plausible proxies.

### Comparison table

Produce one table, benchmark versus model, on identical test data:

| Model | PR-AUC | ROC-AUC | Precision@10% | Recall@thr | Brier |
|---|---|---|---|---|---|
| Majority class | | | | | |
| Heuristic (`number_inpatient >= 1`) | | | | | |
| Logistic regression | | | | | |
| XGBoost (uncalibrated) | | | | | |
| XGBoost (calibrated) | | | | | |

This table is the deliverable. Save it to `reports/model_comparison.md` and as
a CSV.

---

## Task 4 — Deployment

### Batch Transform, not a real-time endpoint

Discharge cohorts finalize on a daily cycle and the care team works the list the
next morning. Nothing in this workflow needs sub-second inference, and a
persistent endpoint bills continuously for zero operational benefit. Use Batch
Transform on an EventBridge daily schedule.

State this rationale explicitly in the design document — the assignment asks you
to choose, and "we picked batch because it's cheaper and matches the decision
cadence" is a better answer than deploying an endpoint because it demos well.

### The artifact bundle

The deployable unit is not the model file. It is:

```
model.tar.gz
├── preprocessor.joblib     # fitted on train only
├── model.xgb               # the booster
├── calibrator.joblib       # Platt scaling
├── threshold.json          # the frozen operating threshold
└── feature_schema.json     # ordered feature names + dtypes
```

Ship these separately and they will drift apart. Promote, version, and roll back
as one unit.

### Model Registry

Register into a `ModelPackageGroup` with `ModelApprovalStatus=PendingManualApproval`.
Attach validation metrics to the model package metadata so the registry shows
performance alongside the version. Approval is the gate before any promotion.

### Batch Transform configuration

Instance `ml.m5.xlarge`. The daily cohort is hundreds to low thousands of rows
against a model of a few megabytes, so this is oversized and leaves room for a
full-corpus backfill.

**The join-back problem.** Batch Transform input must not contain the target,
but the output must be traceable to a patient or the worklist is useless.
Configure:

```python
input_filter  = "$[1:]"        # drop the identifier column from what the model sees
join_source   = "Input"        # attach input to output
output_filter = "$[0,-1]"      # keep identifier + prediction
```

Get this wrong and you produce a column of probabilities nobody can act on.

### Output

Write ranked predictions to `s3://<bucket>/predictions/dt=<date>/`, sorted
descending by calibrated score, with the identifier, the score, and the flag at
the frozen threshold.

### Smoke test

After deployment, score a fixed canned batch and assert: output row count equals
input row count, all scores fall in [0, 1], and the predictions for a known
input row match what the local model produces. That last assertion is your
training/serving skew regression test and it belongs in CI.

---

## Deliverables checklist

- [ ] Majority-class floor with metrics recorded
- [ ] Heuristic benchmark trained and registered through SageMaker
- [ ] XGBoost trained in script mode with HPO, bundled with its preprocessor
- [ ] Calibrator fit on validation, serialized into the artifact
- [ ] Threshold selected on validation at target alert volume, frozen to JSON
- [ ] Test-set evaluation with all metrics, run exactly once
- [ ] Subgroup metrics across race, gender, age, with disparities stated
- [ ] Comparison table, benchmark versus model, saved to `reports/`
- [ ] Model package registered as PendingManualApproval with metrics attached
- [ ] Batch Transform job producing a ranked, identifier-joined worklist
- [ ] Smoke test passing, including the skew regression assertion
- [ ] `RESULTS.md` updated with expected versus actual for every number above
