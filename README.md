 # Customer Churn Segmentation Using Unsupervised Machine Learning

An end-to-end data science project that applies unsupervised machine learning to segment telecommunications customers and analyze their observed churn behavior. The project compares K-Means, Agglomerative Hierarchical Clustering, and DBSCAN, and provides an interactive Streamlit dashboard for exploring customer segments.

## Project Overview

Customer churn is an important business challenge for subscription-based services. Understanding customer groups can help businesses investigate patterns in customer behavior and identify segments that may deserve retention-focused analysis.

This project uses customer demographic, account, service, and billing information to discover customer segments without using the churn label during clustering. After clustering, the actual churn labels are used to interpret and compare the resulting segments.

**Project type:** Data Science / Unsupervised Machine Learning

**Dataset:** IBM Telco Customer Churn

**Algorithms:** K-Means, Agglomerative Hierarchical Clustering, DBSCAN

**Dashboard:** Streamlit

## Objectives

- Explore customer characteristics and churn patterns.
- Clean and preprocess numerical and categorical data.
- Apply feature scaling and one-hot encoding.
- Segment customers using three clustering algorithms.
- Evaluate cluster separation using the silhouette score.
- Compare cluster sizes and observed churn rates.
- Interpret segments using customer tenure and billing characteristics.
- Build an interactive dashboard to explore clustering results.

## Project Structure

```text
customer-churn-segmentation/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_kmeans.ipynb
│   ├── 04_hierarchical.ipynb
│   ├── 05_dbscan.ipynb
│   └── 06_comparison.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── clustering.py
│   └── evaluation.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The project uses the IBM Telco Customer Churn dataset, containing **7,043 customer records and 21 original columns**.

The data includes customer demographics, subscribed services, contract types, payment methods, tenure, charges, and churn labels.

The `customerID` column is excluded from clustering because it is an identifier. The `Churn` column is also excluded from the clustering features and retained separately for post-clustering interpretation.

Before publishing the dataset in a public repository, check its applicable usage and redistribution terms. If redistribution is not permitted, provide instructions for obtaining the dataset instead of committing the raw CSV.

## Methodology

### 1. Exploratory Data Analysis

- Examined dataset dimensions, data types, and missing values.
- Investigated overall churn distribution.
- Compared churn rates across contract types, internet services, and payment methods.
- Explored relationships among tenure and billing variables.

### 2. Data Preprocessing

- Converted `TotalCharges` into a numerical feature.
- Handled blank `TotalCharges` values, which occurred for customers with zero tenure.
- Standardized numerical features using `StandardScaler`.
- Encoded categorical features using `OneHotEncoder`.
- Kept customer identifiers and churn labels separate from the clustering features.

The resulting feature matrix contains **7,043 customers and 45 encoded features**.

### 3. Clustering Algorithms

**K-Means**

Groups customers into a specified number of clusters by minimizing within-cluster distances. The project evaluated different cluster counts and selected three clusters for the main comparison.

**Agglomerative Hierarchical Clustering**

Builds customer groups using a bottom-up clustering approach. Ward linkage was used with three clusters in the main comparison.

**DBSCAN**

Identifies dense groups of customers and marks points that do not meet the density requirements as noise. The main configuration uses `eps=2.5` and `min_samples=10`.

### 4. Evaluation and Interpretation

The algorithms were compared using:

- Silhouette score
- Number and size of clusters
- Noise points identified by DBSCAN
- Observed churn rate by cluster
- Average tenure and customer charges

Churn labels are used to interpret the clusters after they have been formed, not as input features for the clustering algorithms.

## Results

The main comparison produced the following results:

| Algorithm | Clusters | Noise customers | Silhouette score |
|---|---:|---:|---:|
| K-Means | 3 | 0 | 0.2383 |
| Hierarchical | 3 | 0 | 0.1996 |
| DBSCAN | 4 | 108 | 0.1736 |

For the comparison, silhouette scores were estimated using a common sample of 2,000 customers. DBSCAN's score was calculated on the non-noise customers in that sample.

### Key Findings

- The overall observed churn rate in the dataset is **26.54%**.
- K-Means produced the highest silhouette score among the three evaluated configurations.
- K-Means identified a segment with an observed churn rate of **44.03%**.
- Hierarchical clustering identified a segment with an observed churn rate of **42.61%**.
- DBSCAN identified a segment with an observed churn rate of **46.02%** and classified 108 customers as noise in the full dataset.
- Contract type, tenure, and billing characteristics helped distinguish the discovered segments.

The silhouette scores are relatively modest, indicating that the customer groups are not cleanly separated. K-Means is the provisional choice for this project based on its score and interpretability, not proof that it is universally the best clustering algorithm.

## Streamlit Dashboard

The interactive dashboard allows users to:

- Choose K-Means, Hierarchical Clustering, or DBSCAN.
- Adjust clustering parameters.
- View the number of customers, clusters, and noise points.
- Examine the silhouette score.
- Compare observed churn rates across clusters.
- Explore cluster characteristics such as average tenure and charges.
- View a business insight highlighting the segment with the highest observed churn rate.

## Installation and Usage

### Prerequisites

- Python 3.11 recommended
- Git
- pip
- Jupyter Notebook for the notebooks

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd customer-churn-segmentation
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the dataset CSV at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Ensure that the dataset is available before running the dashboard or notebooks.

### 5. Launch the dashboard

Run the following command from the project root:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`, where the dashboard can be opened in a browser.

### 6. Run the notebooks

Open Jupyter from the project root:

```bash
jupyter notebook
```

Run the notebooks in numerical order, from exploratory data analysis through the final algorithm comparison.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Git and GitHub

## Limitations

- The silhouette scores indicate limited cluster separation.
- Results depend on feature engineering, scaling, and algorithm parameters.
- Cluster labels are arbitrary and may differ between algorithms.
- Observed churn rates describe existing groups; they do not establish causation.
- This is a segmentation project, not a supervised model for predicting individual customer churn.
- The current analysis does not demonstrate that any retention intervention will reduce churn.

## Future Improvements

- Add downloadable customer-to-cluster assignments.
- Investigate additional clustering parameters and feature representations.
- Evaluate cluster stability across samples and parameter settings.
- Develop clearer segment profiles and visual comparisons.
- Test whether the segments support useful business decisions with additional evidence.

## Author

**Nirjal Poudel**

BSc CSIT student interested in Data Science, Machine Learning, Deep Learning, and AI-powered product development.

## Acknowledgements

Thanks to the creators and maintainers of the IBM Telco Customer Churn dataset and the open-source Python ecosystem.