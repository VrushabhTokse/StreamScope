"""
StreamScope — Data Preprocessing Module
Handles loading, cleaning, and feature engineering for the Netflix dataset.
"""

import pandas as pd
import numpy as np
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Netflix CSV dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Please download netflix_titles.csv from Kaggle and place it in the data/ folder."
        )
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the raw Netflix dataset."""
    df = df.copy()

    # ── Drop duplicates ────────────────────────────────────────────────────
    df.drop_duplicates(subset="show_id", keep="first", inplace=True)

    # ── Normalize Type (handle Tv Show vs TV Show) ───────────────────────
    df["type"] = df["type"].str.strip().str.title().replace({
        "Tv Show": "TV Show",
        "Movie": "Movie"
    })

    # ── Fill missing values ────────────────────────────────────────────────
    df["director"].fillna("Unknown", inplace=True)
    df["cast"].fillna("Unknown", inplace=True)
    df["country"].fillna("Unknown", inplace=True)
    df["rating"].fillna("Unknown", inplace=True)
    df["duration"].fillna("Unknown", inplace=True)
    df["date_added"].fillna("January 1, 2000", inplace=True)

    # ── Parse date_added ───────────────────────────────────────────────────
    # Try different formats as some datasets have ISO others have Month Day, Year
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
    
    # If date_added failed but year_added already exists in the CSV, don't overwrite with 0s
    if "year_added" in df.columns and df["year_added"].notna().any() and (df["year_added"] > 0).any():
        # Keep existing year_added if it's already there and valid
        pass
    else:
        df["year_added"] = df["date_added"].dt.year.fillna(0).astype(int)
    
    df["month_added"] = df["date_added"].dt.month.fillna(0).astype(int)

    # ── Primary genre (first listed genre) ────────────────────────────────
    df["genres_list"] = df["listed_in"].apply(
        lambda x: [g.strip() for g in str(x).split(",")] if pd.notna(x) else ["Unknown"]
    )
    df["primary_genre"] = df["genres_list"].apply(lambda x: x[0] if x else "Unknown")

    # ── Primary country ────────────────────────────────────────────────────
    df["primary_country"] = df["country"].apply(
        lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "Unknown"
    )

    # ── Decade of release ─────────────────────────────────────────────────
    df["decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"

    # ── Duration numeric & content length category ─────────────────────────
    def parse_duration(row):
        dur = str(row["duration"]).strip()
        if "min" in dur:
            try:
                return int(dur.replace(" min", ""))
            except ValueError:
                return np.nan
        elif "Season" in dur:
            try:
                return int(dur.split(" ")[0])
            except ValueError:
                return np.nan
        return np.nan

    df["duration_value"] = df.apply(parse_duration, axis=1)

    def content_length_category(row):
        if row["type"] == "Movie":
            val = row["duration_value"]
            if pd.isna(val):
                return "Unknown"
            if val < 60:
                return "Short"
            elif val <= 120:
                return "Medium"
            else:
                return "Long"
        elif row["type"] == "TV Show":
            val = row["duration_value"]
            if pd.isna(val):
                return "Unknown"
            if val == 1:
                return "Short"
            elif val <= 3:
                return "Medium"
            else:
                return "Long"
        return "Unknown"

    df["content_length_category"] = df.apply(content_length_category, axis=1)

    # ── Rating grouping ────────────────────────────────────────────────────
    mature_ratings = ["TV-MA", "R", "NC-17"]
    teen_ratings = ["TV-14", "PG-13"]
    family_ratings = ["TV-G", "TV-Y", "TV-Y7", "TV-Y7-FV", "G", "PG"]

    def rate_group(r):
        if r in mature_ratings:
            return "Mature"
        elif r in teen_ratings:
            return "Teen"
        elif r in family_ratings:
            return "Family"
        else:
            return "Other"

    df["rating_group"] = df["rating"].apply(rate_group)

    return df


def get_clean_data(filepath: str) -> pd.DataFrame:
    """One-shot: load and clean."""
    df = load_data(filepath)
    df = clean_data(df)
    return df
