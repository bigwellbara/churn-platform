# Data

This folder is git-ignored (except this file). Datasets are never committed.

## Telco Customer Churn

- Source: Kaggle, "Telco Customer Churn" (IBM sample dataset), file `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- ~7,043 customers, 21 columns, target column `Churn` (Yes/No), roughly 26.5% churners.

Place the file at:

```
data/raw/telco_churn.csv
```

Then train with:

```
uv run churn-train
```
