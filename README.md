# PCA (Principal Component Analysis) — sklearn

## What this is

PCA using `sklearn.decomposition.PCA` directly, with the evaluation needed in practice: variance
explained, reconstruction error, and a downstream-model comparison to check
whether reducing dimensions helped or hurt.

## Dataset

The real **Telco Customer Churn** dataset (`data/Telco-Customer-Churn.csv`,
7,043 rows, 21 columns — 26.5% churn rate). `load_data()` in
`train_model.py` handles the real-world cleanup this dataset needs:
- drops `customerID` (an identifier, not a feature)
- `TotalCharges` is stored as text in the raw file because 11 rows (all
  customers with `tenure=0`) have a blank value instead of a number —
  coerced to numeric and filled with 0
- `Churn` (Yes/No) encoded to 1/0
- all categorical columns (`gender`, `Contract`, `PaymentMethod`,
  `InternetService`, etc.) one-hot encoded with `drop_first=True`, which
  expands the dataset to 30 numeric features — this is what PCA actually
  runs on

## Run it

```bash
pip install -r requirements.txt
python train_model.py
```

## What `src/pca_analysis.py` gives you

- `standardize(X)` — scale features before PCA (required, PCA is
  scale-sensitive)
- `fit_full_pca(X_scaled)` — fit keeping all components, for inspecting
  variance explained per component
- `n_components_for_variance(pca_full, target=0.90)` — smallest k that
  reaches a variance target
- `reconstruction_error_by_k(X_scaled, k_values)` — mean squared error
  between original and reconstructed data at each k (how much information
  is lost)
- `compare_downstream_performance(X_scaled, y, k)` — cross-validated
  Logistic Regression accuracy on full features vs. PCA-reduced features
- `get_loadings(pca, feature_names)` — which original features drive each
  principal component

## Results from this run (real Telco Customer Churn data)

| Check | Result |
|---|---|
| Features after one-hot encoding | 30 |
| Components for 90% variance | 15 of 30 |
| Reconstruction MSE at k=15 | 0.082 (down from 0.668 at k=1) |
| Downstream accuracy, all 30 features | 0.8038 |
| Downstream accuracy, top 15 PCA components | 0.8001 |
| Accuracy difference | -0.0037 |

**Takeaway:** halving the feature count (30 → 15) cost only 0.37 points of
accuracy. PC1 is dominated by the "no internet service" family of dummy
columns (`OnlineSecurity_No internet service`, `TechSupport_No internet
service`, etc.) plus `MonthlyCharges` — these all move together because a
customer with no internet service triggers the same "No internet service"
category across six different columns simultaneously, which is exactly the
kind of one-hot-encoding-induced redundancy PCA is good at collapsing.
PC2 is led by `tenure`, `TotalCharges`, and `Contract_Two year` — a
"long-standing, committed customer" axis.
