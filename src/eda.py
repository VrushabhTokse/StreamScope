"""
StreamScope — EDA Helper Module (Advanced v2)
Extended aggregation functions for deeper exploratory analysis.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re


def content_growth_over_time(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["year_added"] > 2000].copy()
    growth = filtered.groupby(["year_added", "type"]).size().reset_index(name="count")
    return growth


def genre_distribution(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    exploded = df.explode("genres_list")
    genre_counts = exploded["genres_list"].value_counts().head(top_n).reset_index()
    genre_counts.columns = ["genre", "count"]
    return genre_counts


def country_content_counts(df: pd.DataFrame) -> pd.DataFrame:
    country_counts = (
        df[df["primary_country"] != "Unknown"]["primary_country"]
        .value_counts().reset_index()
    )
    country_counts.columns = ["country", "count"]
    return country_counts


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rating_counts = (
        df[df["rating"] != "Unknown"]["rating"]
        .value_counts().reset_index()
    )
    rating_counts.columns = ["rating", "count"]
    return rating_counts


def content_type_split(df: pd.DataFrame) -> dict:
    return df["type"].value_counts().to_dict()


def top_genres_per_year(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    filtered = df[df["year_added"] > 2000].copy()
    exploded = filtered.explode("genres_list")
    grouped = exploded.groupby(["year_added", "genres_list"]).size().reset_index(name="count")
    top = (
        grouped.sort_values("count", ascending=False)
        .groupby("year_added").head(top_n)
        .reset_index(drop=True)
    )
    return top


def rating_group_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["rating_group"].value_counts().reset_index()
    counts.columns = ["rating_group", "count"]
    return counts


def content_length_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    breakdown = df.groupby(["type", "content_length_category"]).size().reset_index(name="count")
    return breakdown


def decade_distribution(df: pd.DataFrame) -> pd.DataFrame:
    decade_counts = df["decade"].value_counts().sort_index().reset_index()
    decade_counts.columns = ["decade", "count"]
    return decade_counts


# ── NEW ADVANCED FUNCTIONS ─────────────────────────────────────────────────────

def director_leaderboard(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top N directors by number of titles (excluding Unknown)."""
    exploded = df[df["director"] != "Unknown"]["director"].str.split(",").explode().str.strip()
    counts = exploded.value_counts().head(top_n).reset_index()
    counts.columns = ["director", "count"]
    return counts


def actor_leaderboard(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top N actors by number of titles (excluding Unknown)."""
    exploded = df[df["cast"] != "Unknown"]["cast"].str.split(",").explode().str.strip()
    counts = exploded.value_counts().head(top_n).reset_index()
    counts.columns = ["actor", "count"]
    return counts


def genre_cooccurrence_matrix(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Build a genre co-occurrence matrix for the top N genres."""
    # Get top genres
    all_genres = []
    for g_list in df["genres_list"]:
        all_genres.extend(g_list)
    top_genres = [g for g, _ in Counter(all_genres).most_common(top_n)]

    co_matrix = pd.DataFrame(0, index=top_genres, columns=top_genres)

    for g_list in df["genres_list"]:
        filtered_genres = [g for g in g_list if g in top_genres]
        for i, g1 in enumerate(filtered_genres):
            for g2 in filtered_genres[i:]:
                co_matrix.loc[g1, g2] += 1
                if g1 != g2:
                    co_matrix.loc[g2, g1] += 1

    return co_matrix


def monthly_additions_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot of titles added per (year, month)."""
    filtered = df[(df["year_added"] > 2000) & (df["month_added"] > 0)].copy()
    pivot = filtered.groupby(["year_added", "month_added"]).size().reset_index(name="count")
    pivot_wide = pivot.pivot(index="month_added", columns="year_added", values="count").fillna(0)
    return pivot_wide


def content_addition_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of years between release_year and year_added."""
    valid = df[(df["year_added"] > 0) & (df["release_year"] > 0)].copy()
    valid["lag_years"] = valid["year_added"] - valid["release_year"]
    valid = valid[valid["lag_years"].between(0, 30)]
    lag_counts = valid["lag_years"].value_counts().sort_index().reset_index()
    lag_counts.columns = ["lag_years", "count"]
    return lag_counts


def description_word_frequency(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Most frequent meaningful words in Netflix descriptions."""
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "his", "her", "their", "he", "she", "they", "it",
        "as", "by", "from", "be", "are", "was", "were", "has", "have", "had",
        "that", "this", "when", "who", "which", "how", "after", "before", "into",
        "new", "one", "two", "out", "up", "its", "him", "them", "all", "more",
        "can", "not", "what", "while", "will", "than", "about", "must", "find",
        "set", "take", "get", "make", "life", "own", "goes", "soon", "also",
        "each", "then", "now", "between", "during", "through", "against", "over",
        "back", "off", "way", "world", "film"
    }
    text = " ".join(df["description"].dropna().astype(str).tolist()).lower()
    words = re.findall(r"\b[a-z]{4,}\b", text)
    freq = Counter(w for w in words if w not in STOPWORDS)
    df_words = pd.DataFrame(freq.most_common(top_n), columns=["word", "count"])
    return df_words


def country_genre_heatmap(df: pd.DataFrame, top_countries: int = 10, top_genres: int = 8) -> pd.DataFrame:
    """Cross-tabulation: top countries vs top genres."""
    top_c = df["primary_country"].value_counts().head(top_countries).index.tolist()
    top_g_series = df.explode("genres_list")["genres_list"].value_counts().head(top_genres).index.tolist()

    filtered = df[df["primary_country"].isin(top_c)].copy()
    exploded = filtered.explode("genres_list")
    exploded = exploded[exploded["genres_list"].isin(top_g_series)]

    pivot = exploded.groupby(["primary_country", "genres_list"]).size().reset_index(name="count")
    matrix = pivot.pivot(index="primary_country", columns="genres_list", values="count").fillna(0)
    return matrix


def movie_duration_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Duration distribution of movies in minutes."""
    movies = df[(df["type"] == "Movie") & df["duration_value"].notna()].copy()
    movies = movies[(movies["duration_value"] >= 20) & (movies["duration_value"] <= 300)]
    return movies[["title", "duration_value", "primary_genre", "release_year", "rating"]]


def tv_seasons_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Season count distribution of TV shows."""
    shows = df[(df["type"] == "TV Show") & df["duration_value"].notna()].copy()
    season_counts = shows["duration_value"].astype(int).value_counts().sort_index().reset_index()
    season_counts.columns = ["seasons", "count"]
    return season_counts[season_counts["seasons"] <= 15]
