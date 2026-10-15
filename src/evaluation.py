
"""
Evaluation utilities for customer clustering.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score


def calculate_silhouette(X, labels, sample_size=2000, random_state=42):
    """Calculate silhouette score, excluding DBSCAN noise points."""

    labels = np.asarray(labels)

    # DBSCAN assigns -1 to noise points
    non_noise_mask = labels != -1
    X_non_noise = X.iloc[np.where(non_noise_mask)[0]]
    labels_non_noise = labels[non_noise_mask]

    # Silhouette requires at least two clusters
    if len(np.unique(labels_non_noise)) < 2:
        return np.nan

    # Use a sample when the dataset is large
    actual_sample_size = min(sample_size, len(labels_non_noise))

    return silhouette_score(
        X_non_noise,
        labels_non_noise,
        sample_size=actual_sample_size,
        random_state=random_state
    )


def summarize_clusters(metadata, labels):
    """Summarize cluster sizes and observed churn rates."""

    results = metadata[["Churn"]].copy()
    results["Cluster"] = np.asarray(labels)

    summary = results.groupby("Cluster").agg(
        Customers=("Churn", "size"),
        Churned=("Churn", lambda values: (values == "Yes").sum())
    )

    summary["Churn Rate (%)"] = (
        summary["Churned"] / summary["Customers"] * 100
    ).round(2)

    return summary
