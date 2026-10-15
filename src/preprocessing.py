
"""
Preprocessing utilities for the customer churn segmentation project.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_customers(df):
    """
    Clean customer data and transform features for clustering.

    Returns:
        X_processed: Numeric feature DataFrame for clustering.
        metadata: Customer IDs and churn labels for interpretation.
        preprocessor: Fitted preprocessing pipeline.
    """

    # Work on a copy so the original data remains unchanged
    df_clean = df.copy()

    # Convert blank TotalCharges values to missing values
    df_clean["TotalCharges"] = df_clean["TotalCharges"].replace(
        r"^\s*$", float("nan"), regex=True
    )

    # Convert TotalCharges to numeric
    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"],
        errors="coerce"
    )

    # These missing values belong to customers with zero tenure
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0)

    # Keep identifiers and churn labels outside the clustering features
    metadata = df_clean[["customerID", "Churn"]].copy()

    X = df_clean.drop(columns=["customerID", "Churn"])

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()

    # Scale numerical features and one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            )
        ]
    )

    # Fit the transformer and create the numeric feature matrix
    transformed_data = preprocessor.fit_transform(X)

    # Recover readable feature names
    feature_names = preprocessor.get_feature_names_out()

    X_processed = pd.DataFrame(
        transformed_data,
        columns=feature_names,
        index=df_clean.index
    )

    return X_processed, metadata, preprocessor
