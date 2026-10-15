
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project root available for importing src modules
PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_customers
from src.clustering import (
    run_kmeans,
    run_hierarchical,
    run_dbscan
)
from src.evaluation import (
    calculate_silhouette,
    summarize_clusters
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Segmentation")
st.write(
    "Explore customer segments using unsupervised machine learning "
    "and compare their observed churn behavior."
)


# --------------------------------------------------
# Load and preprocess data
# --------------------------------------------------

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


@st.cache_data
def load_customer_data(path):
    """Load the original customer dataset."""
    return pd.read_csv(path)


@st.cache_data
def prepare_customer_data(data):
    """Preprocess customer features for clustering."""
    features, metadata, _ = preprocess_customers(data)
    return features, metadata


try:
    raw_data = load_customer_data(DATA_PATH)
    X_processed, metadata = prepare_customer_data(raw_data)
except Exception as error:
    st.error(f"Could not load or preprocess the dataset: {error}")
    st.stop()


# --------------------------------------------------
# Sidebar: algorithm selection
# --------------------------------------------------

st.sidebar.header("Clustering Settings")

algorithm = st.sidebar.selectbox(
    "Choose an algorithm",
    ["K-Means", "Hierarchical", "DBSCAN"]
)

if algorithm in ["K-Means", "Hierarchical"]:
    n_clusters = st.sidebar.slider(
        "Number of clusters",
        min_value=2,
        max_value=8,
        value=3
    )

elif algorithm == "DBSCAN":
    eps = st.sidebar.slider(
        "Epsilon (eps)",
        min_value=1.0,
        max_value=4.0,
        value=2.5,
        step=0.1
    )

    min_samples = st.sidebar.slider(
        "Minimum samples",
        min_value=3,
        max_value=20,
        value=10
    )


# --------------------------------------------------
# Run clustering
# --------------------------------------------------

if st.button("Run Customer Segmentation", type="primary"):

    with st.spinner("Clustering customers..."):

        # Run the selected clustering algorithm
        if algorithm == "K-Means":
            model, labels = run_kmeans(
                X_processed,
                n_clusters=n_clusters
            )

        elif algorithm == "Hierarchical":
            model, labels = run_hierarchical(
                X_processed,
                n_clusters=n_clusters
            )

        else:
            model, labels = run_dbscan(
                X_processed,
                eps=eps,
                min_samples=min_samples
            )

        # Evaluate clustering results
        results = summarize_clusters(metadata, labels)

        silhouette = calculate_silhouette(
            X_processed,
            labels,
            sample_size=2000
        )

        noise_count = int((labels == -1).sum())
        cluster_count = len(set(labels) - {-1})

        # Display key metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Customers", f"{len(labels):,}")
        col2.metric("Clusters Found", cluster_count)
        col3.metric("Noise Customers", f"{noise_count:,}")

        # Clustering evaluation
        st.subheader("Clustering Evaluation")

        if pd.isna(silhouette):
            st.info("Silhouette score could not be calculated.")
        else:
            st.metric("Silhouette Score", f"{silhouette:.4f}")

        # Customer segment summary
        st.subheader("Customer Segment Summary")
        st.caption(
            "Cluster labels are arbitrary. For DBSCAN, -1 represents noise."
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        # Churn rate visualization
        st.subheader("Observed Churn Rate by Cluster")

        churn_chart = results[["Churn Rate (%)"]]
        st.bar_chart(churn_chart)

        # --------------------------------------------------
        # Cluster characteristics and business insights
        # --------------------------------------------------

        st.subheader("Cluster Characteristics")

        # Combine original customer data with cluster assignments
        profile = raw_data.copy()
        profile["Cluster"] = labels

        # Convert TotalCharges to numeric and handle missing values
        profile["TotalCharges"] = pd.to_numeric(
            profile["TotalCharges"],
            errors="coerce"
        ).fillna(0)

        # Calculate characteristics for each cluster
        profile_summary = profile.groupby("Cluster").agg(
            Customers=("customerID", "count"),
            Average_Tenure_Months=("tenure", "mean"),
            Average_Monthly_Charges=("MonthlyCharges", "mean"),
            Average_Total_Charges=("TotalCharges", "mean")
        )

        # Add observed churn rate for each cluster
        profile_summary["Churn_Rate_Percent"] = results[
            "Churn Rate (%)"
        ]

        profile_summary = profile_summary.round(2)

        st.dataframe(
            profile_summary,
            use_container_width=True
        )

        # Identify the segment with the highest observed churn rate
        valid_clusters = results.drop(index=-1, errors="ignore")

        if not valid_clusters.empty:
            highest_churn_cluster = valid_clusters[
                "Churn Rate (%)"
            ].idxmax()

            highest_churn_rate = valid_clusters.loc[
                highest_churn_cluster,
                "Churn Rate (%)"
            ]

            highest_churn_customers = int(
                valid_clusters.loc[
                    highest_churn_cluster,
                    "Customers"
                ]
            )

            st.subheader("Business Insight")

            st.warning(
                f"Cluster {highest_churn_cluster} has the highest "
                f"observed churn rate at {highest_churn_rate:.2f}%, "
                f"covering {highest_churn_customers:,} customers. "
                "This segment may deserve further investigation "
                "for customer-retention opportunities."
            )

        # Explain how to interpret the results
        st.info(
            "Churn rates describe customers in each discovered segment. "
            "They do not establish why customers churn or predict an "
            "individual customer's future churn. Cluster labels are "
            "algorithm-specific and should not be compared directly "
            "between different algorithms."
        )