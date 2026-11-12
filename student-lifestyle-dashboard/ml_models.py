"""Machine-learning helpers: K-Means clustering and Linear Regression (GPA prediction)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "GPA",
]
# Physical activity is excluded from the regression: the five activity columns sum to exactly 24 h,
# so including all of them would make the predictors perfectly collinear.
REGRESSION_FEATURES = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
]


def kmeans_elbow(df: pd.DataFrame, k_range=range(2, 8), seed: int = 42) -> pd.DataFrame:
    """Inertia and silhouette score for several values of k."""
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def run_kmeans(df: pd.DataFrame, k: int = 3, seed: int = 42):
    """Cluster students; clusters are renumbered 1..k by ascending mean GPA so labels are stable."""
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    labels = pd.Series(km.labels_, index=df.index)
    order = df.groupby(labels)["GPA"].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order, start=1)}
    labels = labels.map(remap).map(lambda i: f"Cluster {i}")
    sil = silhouette_score(X, km.labels_)
    profile = df.groupby(labels)[CLUSTER_FEATURES].mean().round(2)
    profile["Students"] = labels.value_counts()
    return labels, sil, profile.sort_index()


def fit_gpa_regression(df: pd.DataFrame, seed: int = 42):
    """Linear regression GPA ~ lifestyle hours. Returns a dict with model, metrics and predictions."""
    X, y = df[REGRESSION_FEATURES], df["GPA"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = LinearRegression().fit(X_tr, y_tr)
    pred_te = model.predict(X_te)
    # Standardised coefficients make the features comparable
    sd = X_tr.std()
    std_coef = pd.Series(model.coef_ * sd.values, index=REGRESSION_FEATURES)
    return {
        "model": model,
        "r2": r2_score(y_te, pred_te),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred_te))),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "test": pd.DataFrame({"Actual": y_te, "Predicted": pred_te}, index=X_te.index),
        "coef": pd.Series(model.coef_, index=REGRESSION_FEATURES),
        "std_coef": std_coef,
        "intercept": float(model.intercept_),
    }
