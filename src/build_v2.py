import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, MinMaxScaler
import joblib

# =========================================
# PATHS
# =========================================
DATASETS_DIR = r"D:\MovieReccomendationSystem\datasets"
ARTIFACTS_DIR = r"D:\MovieReccomendationSystem\artifacts"
REDUCED_PATH = f"{DATASETS_DIR}\\tmdb_movies_25k.csv"

# =========================================
# LOAD — reuses the SAME 25k reduction build_v1.py already produced.
# Does not redo raw filtering, so V1 and V2 are guaranteed to start
# from identical rows (aligned by `id`, not row position).
# =========================================
df = pd.read_csv(REDUCED_PATH)
print("Loaded reduced dataset:", df.shape)

free_text_cols = ["title", "overview", "tagline"]
structured_cols = ["genres", "keywords", "production_companies"]
for c in free_text_cols + structured_cols:
    df[c] = df[c].fillna("").astype(str)


def clean_free_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens_from_list(s: str, max_items: int = None) -> str:
    items = [p.strip() for p in s.split(",") if p.strip()]
    if max_items:
        items = items[:max_items]
    toks = [re.sub(r"[^a-z0-9]+", "_", i.lower()).strip("_") for i in items]
    return " ".join(t for t in toks if t)


df["title_clean"] = df["title"].map(clean_free_text)
df["overview_clean"] = df["overview"].map(clean_free_text)
df["tagline_clean"] = df["tagline"].map(clean_free_text)
df["genres_tok"] = df["genres"].map(tokens_from_list)
df["keywords_tok"] = df["keywords"].map(tokens_from_list)
df["companies_tok"] = df["production_companies"].map(lambda s: tokens_from_list(s, max_items=2))

# =========================================
# PER-FIELD TF-IDF
# =========================================
overview_vec  = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 max_features=15000, min_df=2, max_df=0.85)
title_vec     = TfidfVectorizer(stop_words="english", max_features=5000, min_df=1)
tagline_vec   = TfidfVectorizer(stop_words="english", max_features=5000, min_df=1)
genres_vec    = TfidfVectorizer(max_features=500, min_df=1)
keywords_vec  = TfidfVectorizer(max_features=8000, min_df=2)
companies_vec = TfidfVectorizer(max_features=3000, min_df=1)

X_overview  = overview_vec.fit_transform(df["overview_clean"])
X_title     = title_vec.fit_transform(df["title_clean"])
X_tagline   = tagline_vec.fit_transform(df["tagline_clean"])
X_genres    = genres_vec.fit_transform(df["genres_tok"])
X_keywords  = keywords_vec.fit_transform(df["keywords_tok"])
X_companies = companies_vec.fit_transform(df["companies_tok"])

print("\nField matrix shapes:")
for name, X in [("overview", X_overview), ("title", X_title), ("tagline", X_tagline),
                 ("genres", X_genres), ("keywords", X_keywords), ("companies", X_companies)]:
    print(f"  {name}: {X.shape}")

# Final weights — tuned via genre_jaccard/title_word_overlap in evaluate.py.
# 0.578 genre_jaccard @ these weights = genre helps without dominating.
WEIGHTS = {
    "overview": 0.50, "genres": 0.18, "keywords": 0.18,
    "companies": 0.04, "title": 0.06, "tagline": 0.04,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6

X_v2 = hstack([
    X_overview  * WEIGHTS["overview"],
    X_genres    * WEIGHTS["genres"],
    X_keywords  * WEIGHTS["keywords"],
    X_companies * WEIGHTS["companies"],
    X_title     * WEIGHTS["title"],
    X_tagline   * WEIGHTS["tagline"],
]).tocsr()
# Re-normalize: movies with sparser metadata (empty tagline/companies)
# would otherwise get deflated vector norms and unfairly lower similarity.
X_v2 = normalize(X_v2, norm="l2")
print("\nV2 combined matrix shape:", X_v2.shape)

# =========================================
# QUALITY SCORE (for hybrid rerank, used at inference time in evaluate.py)
# =========================================
scaler = MinMaxScaler()
df["vote_avg_norm"] = scaler.fit_transform(df[["vote_average"]])
df["vote_conf"] = np.log1p(df["vote_count"]) / np.log1p(df["vote_count"].max())
df["quality_score"] = df["vote_avg_norm"] * df["vote_conf"]

# =========================================
# SAVE
# =========================================
joblib.dump({
    "overview": overview_vec, "title": title_vec, "tagline": tagline_vec,
    "genres": genres_vec, "keywords": keywords_vec, "companies": companies_vec,
    "weights": WEIGHTS,
}, f"{ARTIFACTS_DIR}\\vectorizers_v2.pkl", compress=3)

joblib.dump(X_v2, f"{ARTIFACTS_DIR}\\tfidf_matrix_v2.pkl", compress=3)

df[["id", "label", "title", "year", "genres", "vote_average", "vote_count",
    "quality_score", "poster_path"]].to_csv(
    f"{ARTIFACTS_DIR}\\movies_slim_v2.csv", index=False
)

print("\nbuild_v2.py completed.")
print("Artifacts saved: vectorizers_v2.pkl, tfidf_matrix_v2.pkl, movies_slim_v2.csv")