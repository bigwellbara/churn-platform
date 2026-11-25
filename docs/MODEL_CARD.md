# Model card: `churn-model`

| | |
|---|---|
| **Model** | `churn-model` v1, alias `@champion`, in the MLflow registry |
| **Type** | Gradient-boosted trees (XGBoost 3.2), tuned, inside a scikit-learn Pipeline |
| **Output** | Churn probability in [0, 1] and a churn flag at decision threshold **0.606** |
| **Owner** | Bigwell Baramasimbe |
| **Last trained** | 2026-11-25 |
| **Code** | `src/churn/training/`, reproducible with `uv run churn-train --tune` |

## Intended use

- **Purpose:** rank existing telecom customers by risk of cancelling, so a retention
  team can prioritise outreach (calls, discounts, contract offers).
- **Users:** customer-retention and marketing analysts, via the prediction API and
  dashboard.
- **Decision type:** supports a human-run campaign; the model does not take action itself.

### Out of scope (do not use for)

- Credit, pricing, or eligibility decisions, or anything that *withholds* service.
- Treating customers differently based on age, gender or family status.
- Customers or markets that are unlike the training data (different country, product
  mix, or business customers). Retrain and re-validate first.

## Training data

- **Source:** IBM "Telco Customer Churn" sample dataset (public, via Kaggle). Fictional
  customers of a US telecom provider; single snapshot, no timestamps.
- **Size:** 7,043 customers, 19 input features, 26.5% churn.
- **Features:** account (tenure, contract, billing, payment method, charges), services
  (phone, internet type, add-ons), demographics (gender, senior citizen, partner,
  dependents).
- **Preprocessing:** 11 blank `TotalCharges` for brand-new customers set to 0; two
  engineered features (`avg_monthly_spend`, `num_services`); median/mode imputation,
  scaling and one-hot encoding are inside the saved pipeline.
- **Split:** stratified 60% train / 20% validation / 20% test (`random_state=42`).

## Model selection

Five candidates were compared with 5-fold cross-validation and a validation set, and
selected by validation PR-AUC:

| Model | CV ROC-AUC | Val PR-AUC | Train-val gap |
|---|---|---|---|
| Dummy (predict base rate) | 0.500 | 0.265 | 0.000 |
| Logistic regression | 0.848 ± 0.012 | 0.643 | 0.017 |
| Random forest | 0.847 ± 0.011 | 0.640 | 0.089 |
| XGBoost (defaults) | 0.838 ± 0.010 | 0.634 | 0.114 |
| **XGBoost (tuned)** | **0.849 ± 0.012** | **0.648** | **0.036** |

Tuning (25-trial randomised search) reduced overfitting by choosing shallower trees
(`max_depth=3`), a lower learning rate (0.01, 600 trees) and `min_child_weight=3`.
Logistic regression is within noise of the winner; see *Limitations*.

## Performance (held-out test set, n = 1,409, evaluated once)

| ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| 0.847 | 0.663 | 0.725 | 0.579 | 0.644 | 0.787 |

Confusion matrix at threshold 0.606: **271** churners caught, **103** missed,
**197** false alarms, **838** correct "stays".

The threshold maximises F1 on the validation set. If the cost of a missed churner versus a
wasted offer is known, the threshold should be re-chosen from those costs.

## Fairness evaluation (test set)

| Group | n | Actual churn | Flagged | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Female | 687 | 28.1% | 34.4% | 0.715 | 0.849 |
| Male | 722 | 25.1% | 32.1% | 0.735 | 0.846 |
| Not senior | 1,187 | 23.3% | 28.6% | 0.692 | 0.848 |
| **Senior** | 222 | 44.1% | **57.7%** | 0.816 | **0.786** |
| No dependents | 978 | 31.0% | 41.3% | 0.772 | 0.835 |
| **Has dependents** | 431 | 16.5% | 14.8% | **0.521** | 0.848 |

- **Gender:** no meaningful gap.
- **Senior citizens:** over-flagged relative to their (already high) churn rate, and
  ranked less accurately. If flagged customers receive better offers, seniors benefit;
  if flagging led to any negative treatment, this would be a harm. Small sample, so
  monitor rather than over-correct.
- **Customers with dependents:** about half of churners are missed. Retention campaigns
  relying only on this model will under-serve families.

## Explainability

Permutation importance (drop in validation PR-AUC when a column is shuffled): **tenure
(0.124)**, **Contract (0.093)** and **InternetService (0.036)** dominate; ~10 features
contribute almost nothing. Details are in `notebooks/01_eda.ipynb`. Per-customer
explanations (SHAP) are a planned stretch goal.

## Limitations and risks

- **Synthetic, dated, US data.** Patterns may not transfer to a real or South African
  operator without retraining.
- **No time dimension.** A single snapshot: the model cannot see recent changes
  (bill increases, complaints, outages), which are often the real churn triggers.
- **Probabilities are not calibrated.** Class weighting pushes scores upward. Use them
  for ranking; a 0.7 score does not mean a 70% chance.
- **Simpler alternative.** Logistic regression performs within cross-validation noise
  of the winner and is more interpretable; it should be reconsidered if
  explainability requirements grow.
- **Drift.** Prices, products and customer mix change. Input drift is monitored with a
  PSI check (Week 3), and the model should be retrained when drift is detected.

## Versioning and rollback

Each training run that registers a model creates a new version; the API serves whatever
version holds the `@champion` alias. Rolling back means moving the alias to the previous
version. No code change is needed.
