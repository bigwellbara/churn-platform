# How the model is trained

This document explains every step of `src/churn/training/` and the reasoning behind it.
It is written so that you can defend each decision in an interview.

---

## 1. The problem, framed as ML

- **Task:** binary classification. Will this customer churn (1) or stay (0)?
- **Output:** a probability between 0 and 1, plus a yes/no decision from a threshold.
- **Imbalance:** only ~26.5% of customers churn. A model that always says "stays" is 73.5%
  accurate and completely useless. That is why **accuracy is not the headline metric here**.

## 2. Data cleaning (`data.py`)

| Issue | Fix | Why |
|---|---|---|
| `TotalCharges` is text, with 11 blank values | Convert to number; blanks become 0 | All 11 have `tenure == 0`: new customers not billed yet, so 0 is the *true* value, not a guess |
| `Churn` is "Yes"/"No" | Map to 1/0 | Models need numbers |
| `customerID` | Drop | It's unique per row. A model could memorise it, which is pure leakage with no real signal |
| Stray whitespace | Strip | Stops "Yes" and "Yes " becoming two categories |

The function also **fails loudly** if columns are missing or the target has unexpected
values. Bad data should stop training, not silently produce a bad model.

## 3. Splitting: train / validation / test = 60 / 20 / 20

```
full data ──┬── test (20%)        locked away; used ONCE at the very end
            └── rest ──┬── validation (20%)  pick threshold, compare models
                       └── train (60%)       fit models, cross-validation
```

- **Stratified:** every split keeps the same churn rate (~26.5%).
- **Why three sets?** Using a dataset to make a decision (pick a model, pick a threshold)
  makes scores on that set optimistic. The test set is never used for any decision, so its
  score is an honest estimate of real-world performance.

## 4. Feature engineering (`features.py`)

Two domain features are added to the 19 raw ones:

- **`avg_monthly_spend` = TotalCharges / tenure.** If `MonthlyCharges` is well above the
  historical average, the bill recently went up, which is a classic churn trigger.
- **`num_services`:** count of add-ons (phone, security, backup, streaming…). More
  services means it's harder to leave.

Then preprocessing:

- **Numeric:** median imputation, then **standardisation** (mean 0, std 1). Logistic
  regression needs comparable scales. Tree models don't care, but it doesn't hurt them.
- **Categorical:** most-frequent imputation, then **one-hot encoding**
  (`Contract=Month-to-month` becomes its own 0/1 column). `handle_unknown="ignore"` means
  an unseen value at prediction time doesn't crash the API.

### Why everything lives in one `Pipeline`

```
Pipeline: FeatureEngineer → ColumnTransformer(impute/scale/one-hot) → model
```

The *whole* pipeline is saved as the model. The API sends raw customer fields and gets
the identical transformations used in training. Re-implementing preprocessing in the API
by hand ("training/serving skew") is one of the most common production ML bugs, and this
design rules it out.

It also prevents **leakage** during cross-validation: the scaler and imputer are re-fit
inside each fold, so validation data never influences their statistics.

## 5. The models (`models.py`)

| Model | What it is | Why it's in the comparison |
|---|---|---|
| **Dummy baseline** | Predicts the churn rate for everyone | The floor. Every real model must beat it, otherwise something's broken |
| **Logistic regression** | Weighted sum of features passed through a sigmoid | Fast, interpretable (coefficients show direction), strong on this kind of tabular data. A good model often barely beats it |
| **Random forest** | Hundreds of decision trees on random subsets of rows and features, averaged | Captures non-linear effects and interactions (e.g. fibre *and* month-to-month); robust with little tuning |
| **XGBoost** | Gradient-boosted trees: each new tree corrects the errors of the previous ones | Usually the strongest model on tabular data; industry standard |

### Handling class imbalance

Without help, models lean towards the majority class and miss churners. Each model is
re-weighted:

- Logistic regression and random forest: `class_weight="balanced"`. Errors on churners
  count ~2.8× more.
- XGBoost: `scale_pos_weight = #stayers / #churners` (≈ 2.8), which does the same thing.

The alternative, resampling (e.g. SMOTE, which creates synthetic churners), adds complexity
and leakage risk for usually no gain over weighting.

### Key hyperparameters and what they control

- **Random forest:** `n_estimators=300` trees; `max_depth=10` and `min_samples_leaf=5` stop
  trees memorising individual customers (overfitting control).
- **XGBoost:** `learning_rate=0.05` (small steps, more stable), `max_depth=4` (shallow
  trees generalise better), `subsample=0.8` and `colsample_bytree=0.8` (each tree sees 80%
  of rows and columns, a regularisation trick), `reg_lambda` (L2 penalty on leaf weights).

### Optional tuning (`--tune`)

`RandomizedSearchCV` tries 25 random XGBoost settings from `XGB_SEARCH_SPACE`, each scored
by 5-fold cross-validated PR-AUC on the training set. The best becomes `xgboost_tuned` and
competes with the others. Random search is used instead of grid search because it finds
good settings in far fewer trials. The time cap is deliberate: on this dataset, tuning
usually adds around +0.005 AUC, and the engineering around the model matters more.

## 6. Evaluation (`evaluate.py`)

For each model, `train.py` does:

1. **5-fold stratified cross-validation** on train. The mean is the expected performance;
   the **std** shows stability. A high std means the model is sensitive to which rows it
   sees.
2. **Fit on the full train set**, predict probabilities for train and validation.
3. **Tune the decision threshold** on validation to maximise F1 (see below).
4. **Overfitting check:** `train_roc_auc - val_roc_auc`. A gap above 0.05 logs a warning.
5. Log everything to MLflow.

### The metrics

| Metric | Question it answers | Notes |
|---|---|---|
| **Precision** | Of customers we flagged, how many really churn? | Low means wasted retention offers |
| **Recall** | Of customers who churn, how many did we catch? | Low means lost customers we never contacted |
| **F1** | Balance of precision and recall | Harmonic mean, so it punishes ignoring either |
| **ROC-AUC** | How well does the model rank churners above stayers? | Threshold-free; 0.5 = random, 1.0 = perfect |
| **PR-AUC** | Ranking quality focused on the churn class | **Used to pick the winner.** More honest than ROC-AUC under imbalance; baseline = churn rate (~0.27) |
| **Confusion matrix** | Raw counts of TN / FP / FN / TP | Saved as an image per model in MLflow |

### Why tune the threshold?

A model outputs probabilities; 0.5 is an arbitrary cut-off. An *unweighted* model on 26%
churners usually needs a lower cut-off (often 0.3 to 0.4) to catch enough churners. Our
models use class weighting, which pushes all scores *upward*, so the best cut-off came out
*higher* (0.606). Tuning the threshold corrects for whatever the model does, which is why
we tune it instead of assuming 0.5. The threshold is chosen on **validation**, then stored with the
registered model (`decision_threshold` tag) so the API uses exactly the same value.

In a real business you would pick it from costs: if a missed churner costs R2,000 and a
retention offer costs R100, you'd accept many false positives (low threshold).

## 7. Model selection, test and registry

1. The winner is the model with the highest **validation PR-AUC**.
2. It is scored **once** on the test set. Those are the numbers quoted in the README and
   the model card.
3. The full pipeline is logged with a **signature** (expected input columns and types) and
   an **input example**, then registered as `churn-model`. Each registration gets a new
   version number (v1, v2…).
4. The new version gets the alias **`@champion`**. The API loads
   `models:/churn-model@champion`, so promoting or rolling back a model means moving the
   alias. No code change, no redeploy of the API image needed.

## 8. MLflow: what gets tracked

Per run (one run per model): all hyperparameters, CV mean and std, train and validation
metrics, threshold, overfit gap, confusion-matrix image. The winning run also gets test
metrics and the model artifact.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000, select the runs, click "Compare"
```

Storage: a local SQLite file (`mlflow.db`) for run metadata and the registry, and
`mlartifacts/` for files. In the cloud phase this becomes Postgres plus object storage.

## 9. Results on the real data (run of 2026-11-25)

| Model | CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Val F1 | Train-val gap |
|---|---|---|---|---|---|
| Dummy baseline | 0.500 ± 0.000 | 0.500 | 0.265 | 0.420 | 0.000 |
| Logistic regression | 0.848 ± 0.012 | 0.836 | 0.643 | 0.630 | 0.017 |
| Random forest | 0.847 ± 0.011 | 0.837 | 0.640 | 0.636 | 0.089 ⚠ |
| XGBoost (defaults) | 0.838 ± 0.010 | 0.829 | 0.634 | 0.617 | 0.114 ⚠ |
| **XGBoost (tuned)** | **0.849 ± 0.012** | **0.839** | **0.648** | **0.644** | 0.036 |

Winner on the test set (threshold 0.606): ROC-AUC **0.847**, PR-AUC **0.663**,
recall **0.725**, precision **0.579**, F1 **0.644**.

What to take from it:

- **Every real model gives about 2.4× lift** over the dummy's PR-AUC (0.265 = churn rate).
- **Default XGBoost overfits** (gap 0.114). Tuning chose shallower trees, a lower learning
  rate and a larger `min_child_weight`, which cut the gap to 0.036 *and* improved
  validation scores. That is regularisation working as intended.
- **The top three models are within one CV standard deviation** (±0.012). Logistic
  regression is statistically tied with tuned XGBoost while being simpler and directly
  interpretable. That is a legitimate argument for serving it instead.
- **Permutation importance** (see `notebooks/01_eda.ipynb`): tenure, Contract and
  InternetService carry most of the signal.

See [MODEL_CARD.md](MODEL_CARD.md) for limitations and the fairness evaluation.
