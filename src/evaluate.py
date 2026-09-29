import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import joblib

# =========================================
# PATHS — loads ONLY saved artifacts, never refits anything.
# Keeps eval honest: it tests exactly what build_v1/v2 produced.
# =========================================
ARTIFACTS_DIR = r"D:\MovieReccomendationSystem\artifacts"

vectorizer = joblib.load(f"{ARTIFACTS_DIR}\\vectorizer.pkl")
X_v1 = joblib.load(f"{ARTIFACTS_DIR}\\tfidf_matrix.pkl")
df_v1 = pd.read_csv(f"{ARTIFACTS_DIR}\\movies_slim.csv")

v2_vectorizers = joblib.load(f"{ARTIFACTS_DIR}\\vectorizers_v2.pkl")
X_v2 = joblib.load(f"{ARTIFACTS_DIR}\\tfidf_matrix_v2.pkl")
df_v2 = pd.read_csv(f"{ARTIFACTS_DIR}\\movies_slim_v2.csv")

overview_vec  = v2_vectorizers["overview"]
title_vec     = v2_vectorizers["title"]
tagline_vec   = v2_vectorizers["tagline"]
genres_vec    = v2_vectorizers["genres"]
keywords_vec  = v2_vectorizers["keywords"]
companies_vec = v2_vectorizers["companies"]

print("V1 matrix:", X_v1.shape, "| rows:", len(df_v1))
print("V2 matrix:", X_v2.shape, "| rows:", len(df_v2))

# =========================================
# ALIGN BY id, NOT ROW POSITION
# Two matrices built by two different scripts should never be assumed
# to be in the same row order — align explicitly.
# =========================================
# position-in-matrix lookup, keyed by movie id, per dataframe
v1_id_to_pos = pd.Series(df_v1.index, index=df_v1["id"])
v2_id_to_pos = pd.Series(df_v2.index, index=df_v2["id"])


def label_to_id(df, label):
    """Exact label match first, then fuzzy title contains-match fallback."""
    exact = df[df["label"] == label]
    if not exact.empty:
        return exact.iloc[0]["id"]
    fuzzy = df[df["title"].str.lower().str.contains(label.lower(), na=False)]
    if fuzzy.empty:
        return None
    return fuzzy.iloc[0]["id"]


def recommend(matrix, df, id_to_pos, label, n=10):
    movie_id = label_to_id(df, label)
    if movie_id is None:
        print(f"No match for '{label}'")
        return None
    idx = id_to_pos[movie_id]
    sims = linear_kernel(matrix[idx], matrix).ravel()
    top = sims.argsort()[::-1][1:n + 1]
    out = df.loc[top, ["label", "genres", "vote_average"]].copy()
    out["similarity"] = sims[top]
    return out.reset_index(drop=True)


def hybrid_recommend(matrix, df, id_to_pos, label, n=10, pool_size=50, alpha=0.8):
    movie_id = label_to_id(df, label)
    if movie_id is None:
        print(f"No match for '{label}'")
        return None
    idx = id_to_pos[movie_id]
    sims = linear_kernel(matrix[idx], matrix).ravel()
    pool = sims.argsort()[::-1][1:pool_size + 1]
    pool_df = df.loc[pool].copy()
    pool_df["content_sim"] = sims[pool]
    pool_df["final_score"] = alpha * pool_df["content_sim"] + (1 - alpha) * pool_df["quality_score"]
    result = pool_df.sort_values("final_score", ascending=False).head(n)
    return result[["label", "genres", "vote_average", "content_sim", "final_score"]].reset_index(drop=True)


# =========================================
# LEVEL 1 — SIDE-BY-SIDE MANUAL COMPARISON
# =========================================
test_titles = ["Interstellar", "The Dark Knight", "Toy Story",
               "The Notebook", "John Wick", "Get Out"]

for t in test_titles:
    print(f"\n{'='*70}\n{t}\n{'='*70}")
    print("\n--- V1 ---")
    print(recommend(X_v1, df_v1, v1_id_to_pos, t, 10).to_string())
    print("\n--- V2 ---")
    print(recommend(X_v2, df_v2, v2_id_to_pos, t, 10).to_string())

# =========================================
# LEVEL 2 — QUANTIFIED METRICS
# NOTE: genre_jaccard is inflated for both since genres are inside
# content — valid only as a RELATIVE comparison, not an absolute score.
# =========================================
def genre_jaccard_at_k(matrix, df, id_to_pos, movie_id, k=10):
    idx = id_to_pos[movie_id]
    sims = linear_kernel(matrix[idx], matrix).ravel()
    top = sims.argsort()[::-1][1:k + 1]
    q_genres = set(g.strip() for g in df.loc[idx, "genres"].lower().split(",") if g.strip())
    scores = []
    for i in top:
        r_genres = set(g.strip() for g in df.loc[i, "genres"].lower().split(",") if g.strip())
        union = q_genres | r_genres
        if union:
            scores.append(len(q_genres & r_genres) / len(union))
    return np.mean(scores) if scores else 0.0


def title_word_overlap_rate(matrix, df, id_to_pos, movie_id, k=10):
    idx = id_to_pos[movie_id]
    sims = linear_kernel(matrix[idx], matrix).ravel()
    top = sims.argsort()[::-1][1:k + 1]
    q_words = set(df.loc[idx, "title"].lower().split())
    hits = sum(1 for i in top if q_words & set(df.loc[i, "title"].lower().split()))
    return hits / k


test_ids_v1 = [label_to_id(df_v1, t) for t in test_titles]
test_ids_v2 = [label_to_id(df_v2, t) for t in test_titles]

print(f"\n{'='*50}\nQUANTIFIED COMPARISON\n{'='*50}")
jac_v1 = np.mean([genre_jaccard_at_k(X_v1, df_v1, v1_id_to_pos, i) for i in test_ids_v1])
tov_v1 = np.mean([title_word_overlap_rate(X_v1, df_v1, v1_id_to_pos, i) for i in test_ids_v1])
jac_v2 = np.mean([genre_jaccard_at_k(X_v2, df_v2, v2_id_to_pos, i) for i in test_ids_v2])
tov_v2 = np.mean([title_word_overlap_rate(X_v2, df_v2, v2_id_to_pos, i) for i in test_ids_v2])
print(f"V1: genre_jaccard@10={jac_v1:.3f}   title_word_overlap@10={tov_v1:.3f}")
print(f"V2: genre_jaccard@10={jac_v2:.3f}   title_word_overlap@10={tov_v2:.3f}")

# =========================================
# LEVEL 3 — HYBRID RERANK CHECK
# =========================================
print(f"\n{'='*50}\nHYBRID RERANK (V2 + quality score)\n{'='*50}")
for t in ["Interstellar", "The Dark Knight", "Toy Story"]:
    print(f"\n{t}:")
    print(hybrid_recommend(X_v2, df_v2, v2_id_to_pos, t, 10).to_string())

# =========================================
# DEBUG — explain a specific match (edit args as needed)
# =========================================
def explain_match(matrix, df, id_to_pos, label_a, label_b, top_terms=15):
    id_a, id_b = label_to_id(df, label_a), label_to_id(df, label_b)
    idx_a, idx_b = id_to_pos[id_a], id_to_pos[id_b]
    vec_a = matrix[idx_a].toarray().ravel()
    vec_b = matrix[idx_b].toarray().ravel()
    contribution = vec_a * vec_b
    all_names = np.concatenate([
        [f"ov_{t}" for t in overview_vec.get_feature_names_out()],
        [f"gn_{t}" for t in genres_vec.get_feature_names_out()],
        [f"kw_{t}" for t in keywords_vec.get_feature_names_out()],
        [f"co_{t}" for t in companies_vec.get_feature_names_out()],
        [f"ti_{t}" for t in title_vec.get_feature_names_out()],
        [f"tg_{t}" for t in tagline_vec.get_feature_names_out()],
    ])
    top_idx = contribution.argsort()[::-1][:top_terms]
    print(f"\n=== Why '{label_a}' matched '{label_b}' ===")
    for i in top_idx:
        if contribution[i] > 0:
            print(f"{all_names[i]:30s}  contrib={contribution[i]:.5f}")


explain_match(X_v2, df_v2, v2_id_to_pos, "Interstellar (2014)", "Daddy Cool (2008)")

print("\nevaluate.py completed.")