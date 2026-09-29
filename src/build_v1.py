import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# =========================================
# PATHS
# =========================================
DATASETS_DIR = r"D:\MovieReccomendationSystem\datasets"
ARTIFACTS_DIR = r"D:\MovieReccomendationSystem\artifacts"

RAW_PATH = f"{DATASETS_DIR}\\TMDB_movie_dataset_v11.csv"
REDUCED_PATH = f"{DATASETS_DIR}\\tmdb_movies_25k.csv"
CONTENT_PATH = f"{DATASETS_DIR}\\tmdb_movies_25k_content.csv"

# =========================================
# PHASE 1 — LOAD + FILTER + REDUCE TO 25K
# =========================================
df = pd.read_csv(RAW_PATH, engine="python")
print("Original shape:", df.shape)

df = df[
    (df["status"] == "Released") &
    (df["adult"] == False) &
    (df["title"].notna()) &
    (df["overview"].notna())
].copy()

# Dedup on id, not title — remakes legitimately share a title.
df = df.drop_duplicates(subset="id")

df = df.sort_values(by=["vote_count", "vote_average"], ascending=[False, False])
df = df.head(25000).reset_index(drop=True)

# year + unique label — needed because titles alone can collide (remakes)
df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
df["label"] = df["title"] + " (" + df["year"].astype(str) + ")"

keep_columns = [
    "id", "label", "title", "year",
    "overview", "genres", "keywords", "tagline", "production_companies",
    "vote_average", "vote_count", "release_date", "runtime",
    "budget", "revenue", "popularity", "original_language",
    "spoken_languages", "poster_path",
]
df = df[keep_columns]

print("\nReduced shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

df.to_csv(REDUCED_PATH, index=False)
print(f"\nSaved: {REDUCED_PATH}")

# =========================================
# PHASE 2 — PREPROCESSING + CONTENT (for V1 only)
# =========================================
free_text_cols = ["title", "overview", "tagline"]
structured_cols = ["genres", "keywords", "production_companies"]
for c in free_text_cols + structured_cols:
    df[c] = df[c].fillna("").astype(str)


def clean_free_text(s: str) -> str:
    """lowercase, strip non-alphanumeric (numbers KEPT — sequel numbers/years matter)"""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens_from_list(s: str, max_items: int = None) -> str:
    """
    Turns 'Science Fiction, Drama' into 'science_fiction drama' — one token
    per list item instead of splitting into loose unigrams. Without this,
    "Science Fiction" as a concept is lost and "science" falsely matches
    any unrelated movie that says "science" in its overview.
    """
    items = [p.strip() for p in s.split(",") if p.strip()]
    if max_items:
        items = items[:max_items]
    toks = [re.sub(r"[^a-z0-9]+", "_", i.lower()).strip("_") for i in items]
    return " ".join(t for t in toks if t)


for c in free_text_cols:
    df[c + "_clean"] = df[c].map(clean_free_text)

df["genres_tok"] = df["genres"].map(tokens_from_list)
df["keywords_tok"] = df["keywords"].map(tokens_from_list)
df["companies_tok"] = df["production_companies"].map(lambda s: tokens_from_list(s, max_items=2))

df["content"] = (
    df["title_clean"] + " " + df["overview_clean"] + " " + df["tagline_clean"] + " " +
    df["genres_tok"] + " " + df["keywords_tok"] + " " + df["companies_tok"]
)
df["content"] = df["content"].str.replace(r"\s+", " ", regex=True).str.strip()

empty_content = (df["content"].str.len() == 0).sum()
print("\nRows with empty content:", empty_content)
df = df[df["content"].str.len() > 0].reset_index(drop=True)

df[["id", "label", "title", "year", "genres", "vote_average", "vote_count",
    "poster_path", "content"]].to_csv(CONTENT_PATH, index=False)
print(f"Saved: {CONTENT_PATH}")

# =========================================
# PHASE 3+4 — TF-IDF (V1, merged/unweighted) + SAVE
# =========================================
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    max_df=0.85,
)
tfidf_matrix = vectorizer.fit_transform(df["content"])
print("\nV1 TF-IDF matrix shape:", tfidf_matrix.shape)

feature_names = np.array(vectorizer.get_feature_names_out())
top_idf_idx = vectorizer.idf_.argsort()[::-1][:15]
print("Top 15 rarest terms (high IDF):", feature_names[top_idf_idx])

joblib.dump(vectorizer, f"{ARTIFACTS_DIR}\\vectorizer.pkl", compress=3)
joblib.dump(tfidf_matrix, f"{ARTIFACTS_DIR}\\tfidf_matrix.pkl", compress=3)
df[["id", "label", "title", "year", "genres", "vote_average", "vote_count",
    "poster_path"]].to_csv(f"{ARTIFACTS_DIR}\\movies_slim.csv", index=False)

print("\nbuild_v1.py completed.")
print("Artifacts saved: vectorizer.pkl, tfidf_matrix.pkl, movies_slim.csv")