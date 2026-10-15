
"""
Clustering utilities for customer segmentation.
"""

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


def run_kmeans(X, n_clusters=3, random_state=42):
    """Fit K-Means and return the model and cluster labels."""

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    labels = model.fit_predict(X)

    return model, labels


def run_hierarchical(X, n_clusters=3):
    """Run hierarchical clustering and return labels."""

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )

    labels = model.fit_predict(X)

    return model, labels


def run_dbscan(X, eps=2.5, min_samples=10):
    """Run DBSCAN and return labels.

    DBSCAN uses -1 for observations classified as noise.
    """

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=-1
    )

    labels = model.fit_predict(X)

    return model, labels
