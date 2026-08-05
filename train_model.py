"""
PCA — sklearn-based training script, run on the real
Telco Customer Churn dataset (data/Telco-Customer-Churn.csv).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.pca_analysis import (
    standardize,
    fit_full_pca,
    n_components_for_variance,
    reconstruction_error_by_k,
    compare_downstream_performance,
    get_loadings,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "data/Telco-Customer-Churn.csv"


def load_data(path=DATA_PATH):
    """
    Load and prepare the real Telco Customer Churn dataset for PCA.

    Cleaning steps:
    - Drop customerID (an identifier, not a feature -- carries no signal
      and would just add a meaningless high-cardinality dummy column)
    - TotalCharges is loaded as a string in the raw CSV because 11 rows
      (all customers with tenure=0, i.e. brand new that month) have a
      blank value instead of a number. Coerce to numeric and fill those
      with 0, since 0 tenure logically means 0 total charges so far.
    - Encode Churn (Yes/No) as 1/0 for the downstream classifier
    - One-hot encode all categorical columns (gender, Contract,
      PaymentMethod, etc.) -- PCA and StandardScaler only accept numeric
      input, drop_first=True avoids the dummy-variable trap (one
      redundant column per categorical feature)
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    y = (df["Churn"] == "Yes").astype(int).values

    X_df = df.drop(columns=["Churn"])
    X_df = pd.get_dummies(X_df, drop_first=True)
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(float)

    return X, y, feature_names


def main():
    print("=" * 60)
    print("PCA (sklearn) — Telco Customer Churn (real data)")
    print("=" * 60)

    X, y, feature_names = load_data()
    print(f"\nDataset shape: {X.shape[0]} rows x {X.shape[1]} features (after one-hot encoding)")
    print(f"Churn rate: {y.mean():.1%}")

    # 1. Standardize
    X_scaled, scaler = standardize(X)

    # 2. Fit full PCA to inspect variance explained per component
    pca_full = fit_full_pca(X_scaled)
    print("\nExplained variance ratio (first 10 components):")
    for i, ratio in enumerate(pca_full.explained_variance_ratio_[:10], start=1):
        print(f"  PC{i}: {ratio:.4f}")

    n_90, cum_var = n_components_for_variance(pca_full, target=0.90)
    print(f"\nComponents needed for >=90% variance: {n_90} (of {X.shape[1]} original features)")

    # 3. Scree plot
    plt.figure(figsize=(9, 5))
    components = range(1, len(pca_full.explained_variance_ratio_) + 1)
    plt.bar(components, pca_full.explained_variance_ratio_, alpha=0.6, label="Individual")
    plt.plot(components, cum_var, marker="o", color="red", markersize=3, label="Cumulative")
    plt.axhline(y=0.90, color="gray", linestyle="--", label="90% threshold")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.title("Scree Plot: Variance Explained per Component (Telco Churn)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/scree_plot.png", dpi=150)
    plt.close()
    print("Saved: outputs/scree_plot.png")

    # 4. Reconstruction error by k
    k_values = list(range(1, min(X.shape[1], 30) + 1))
    errors = reconstruction_error_by_k(X_scaled, k_values)
    print(f"\nReconstruction MSE at k={n_90}: {errors[n_90 - 1]:.4f} (k=1: {errors[0]:.4f}, k={k_values[-1]}: {errors[-1]:.4f})")

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, errors, marker="o", markersize=3)
    plt.xlabel("Number of Components (k)")
    plt.ylabel("Mean Squared Reconstruction Error")
    plt.title("Reconstruction Error vs. Number of Components")
    plt.tight_layout()
    plt.savefig("outputs/reconstruction_error.png", dpi=150)
    plt.close()
    print("Saved: outputs/reconstruction_error.png")

    # 5. Downstream task comparison: does PCA help or hurt a real model?
    score_full, score_pca = compare_downstream_performance(
        X_scaled, y, k=n_90, cv=5
    )
    print(f"\n--- Downstream comparison (Logistic Regression, 5-fold CV) ---")
    print(f"All {X.shape[1]} original features -> accuracy: {score_full:.4f}")
    print(f"Top {n_90} PCA components           -> accuracy: {score_pca:.4f}")
    print(f"Difference: {score_pca - score_full:+.4f}")

    # 6. 2D projection for visualization, colored by churn
    X_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1], c=y, cmap="coolwarm", alpha=0.4, s=10
    )
    plt.colorbar(scatter, label="Churn (0=No, 1=Yes)")
    plt.xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("2D PCA Projection of Telco Customer Churn, colored by churn")
    plt.tight_layout()
    plt.savefig("outputs/pca_2d_projection.png", dpi=150)
    plt.close()
    print("Saved: outputs/pca_2d_projection.png")

    # 7. Component loadings (top drivers of PC1/PC2 among original columns)
    loadings = get_loadings(pca_full, feature_names, n_components=2)
    print("\nTop 8 loadings on PC1:")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(8))
    print("\nTop 8 loadings on PC2:")
    print(loadings["PC2"].abs().sort_values(ascending=False).head(8))

    print("\nDone. See outputs/ for plots.")


if __name__ == "__main__":
    main()