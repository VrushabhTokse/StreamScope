"""
StreamScope — Modeling Module (Advanced v2)
KMeans + PCA visualization + RandomForest + TF-IDF Recommendation Engine.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FEATURE_COLS = ["primary_genre", "rating", "content_length_category", "release_year"]
TARGET_COL = "type"


def encode_with_encoders(df: pd.DataFrame, feature_cols: list):
    """Encode categorical features and return the encoded DF and the encoders."""
    encoded = df[feature_cols].copy()
    encoders = {}
    for col in feature_cols:
        # Check if the column is NOT a numeric type (float/int)
        if not pd.api.types.is_numeric_dtype(encoded[col]):
            le = LabelEncoder()
            # Convert to string to handle any mix of types before encoding
            encoded[col] = le.fit_transform(encoded[col].astype(str))
            encoders[col] = le
    return encoded.dropna(), encoders


def encode_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """[DEPRECATED] Use encode_with_encoders. Simple helper for clustering."""
    encoded, _ = encode_with_encoders(df, feature_cols)
    return encoded


# ── Clustering ────────────────────────────────────────────────────────────────
def run_clustering(df: pd.DataFrame, n_clusters: int = 3):
    """KMeans clustering with PCA 2D coordinates for scatter plot."""
    encoded = encode_features(df, FEATURE_COLS)
    valid_idx = encoded.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    result = df.loc[valid_idx].copy()
    result["cluster"] = cluster_labels.astype(str)
    result["pca_x"] = X_pca[:, 0]
    result["pca_y"] = X_pca[:, 1]

    explained = pca.explained_variance_ratio_
    return result, kmeans.inertia_, explained


def elbow_inertias(df: pd.DataFrame, k_range=range(2, 11)) -> dict:
    encoded = encode_features(df, FEATURE_COLS)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded)
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias[k] = km.inertia_
    return inertias


# ── Classification ────────────────────────────────────────────────────────────
def run_classification(df: pd.DataFrame):
    """RandomForest + cross-validation for Movie vs. TV Show classification."""
    encoded, encoders = encode_with_encoders(df, FEATURE_COLS)
    target_df = df.loc[encoded.index, TARGET_COL].copy()
    le_target = LabelEncoder()
    y = le_target.fit_transform(target_df.astype(str))

    scaler = StandardScaler()
    X = scaler.fit_transform(encoded.values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=le_target.classes_, output_dict=True
    )

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)

    class_names = list(le_target.classes_)

    # ROC AUC (binary only)
    try:
        if len(class_names) == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = None
    except Exception:
        auc = None

    return acc, cm, report, importance_df, class_names, cv_scores, auc, clf, encoders


def get_prediction(model, encoders, class_names, raw_features: dict) -> tuple:
    """Predict label, confidence, and feature-level probabilities."""
    numerical_features = []
    for col in FEATURE_COLS:
        val = raw_features[col]
        if col in encoders:
            try:
                # Wrap in 2D array for transform
                transformed = encoders[col].transform([str(val)])[0]
                numerical_features.append(transformed)
            except:
                numerical_features.append(0)
        else:
            try:
                numerical_features.append(float(val))
            except:
                numerical_features.append(0)
            
    # Get prediction and probabilities
    pred_idx = model.predict([numerical_features])[0]
    probs = model.predict_proba([numerical_features])[0]
    
    label = class_names[pred_idx]
    confidence = float(np.max(probs))
    
    # Create dictionary of class -> probability
    prob_dict = {name: float(p) for name, p in zip(class_names, probs)}
    
    return label, confidence, prob_dict


# ── Gradient Boosting comparison ──────────────────────────────────────────────
def run_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare RandomForest vs GradientBoosting accuracy via 5-fold CV."""
    encoded = encode_features(df, FEATURE_COLS)
    target_df = df.loc[encoded.index, TARGET_COL].copy()
    le_target = LabelEncoder()
    y = le_target.fit_transform(target_df.astype(str))

    scaler = StandardScaler()
    X = scaler.fit_transform(encoded.values)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        results.append({"Model": name, "Mean Accuracy": scores.mean(), "Std": scores.std()})

    return pd.DataFrame(results)


# ── Recommendation Engine ─────────────────────────────────────────────────────
def build_recommendation_engine(df: pd.DataFrame):
    """Build TF-IDF matrix on combined text features for content-based recs."""
    rec_df = df[["title", "description", "listed_in", "director", "cast",
                  "rating", "type"]].copy()
    rec_df["description"] = rec_df["description"].fillna("")

    # Combine features into a single text blob
    rec_df["combined"] = (
        rec_df["listed_in"].fillna("") + " " +
        rec_df["description"] + " " +
        rec_df["director"].fillna("") + " " +
        rec_df["cast"].fillna("") + " " +
        rec_df["rating"].fillna("")
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(rec_df["combined"])

    return rec_df.reset_index(drop=True), tfidf_matrix


def get_recommendations(title: str, rec_df: pd.DataFrame, tfidf_matrix, top_n: int = 8) -> pd.DataFrame:
    """Return top-N similar titles using cosine similarity."""
    title_lower = title.lower()
    matches = rec_df[rec_df["title"].str.lower().str.contains(title_lower, na=False)]

    if matches.empty:
        return pd.DataFrame(columns=["title", "type", "listed_in", "director", "similarity"])

    idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = 0  # exclude self
    top_indices = sim_scores.argsort()[-top_n:][::-1]

    result = rec_df.iloc[top_indices][["title", "type", "listed_in", "director"]].copy()
    result["similarity"] = (sim_scores[top_indices] * 100).round(1)
    return result.reset_index(drop=True)
