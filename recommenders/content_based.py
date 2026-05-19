import numpy as np
import pandas as pd
import math
from collections import Counter


# ═══════════════════════════════════════════════════════════════
# Text preprocessing helpers
# ═══════════════════════════════════════════════════════════════

# Common English stop-words to ignore during TF-IDF
_STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","this",
    "that","these","those","it","its","from","by","as","not","no","so",
    "if","then","than","also","into","up","out","about","after","before",
    "between","through","during","each","all","both","few","more","most",
}


def _tokenize(text: str) -> list:
    """Lowercase, strip punctuation, split into tokens, remove stop-words."""
    text = text.lower()
    # Replace non-alphanumeric characters with spaces
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text)
    tokens  = [t for t in cleaned.split() if t and t not in _STOP_WORDS]
    return tokens


def _build_corpus(products: pd.DataFrame) -> list:
    """Concatenate name, category, brand, tags, description into one string per product."""
    def row_to_text(row):
        return " ".join([
            str(row.get("name",        "")),
            str(row.get("category",    "")),
            str(row.get("brand",       "")),
            str(row.get("tags",        "")).replace(",", " "),
            str(row.get("description", "")),
        ])
    return products.apply(row_to_text, axis=1).tolist()


# ═══════════════════════════════════════════════════════════════
# Manual TF-IDF implementation
# ═══════════════════════════════════════════════════════════════

def _compute_tfidf(corpus: list, max_features: int = 500):
    n_docs = len(corpus)

    # Step 1 — Tokenise every document
    tokenized = [_tokenize(doc) for doc in corpus]

    # Step 2 — Build vocabulary: collect all unique tokens, pick top max_features by df
    df_count: Counter = Counter()
    for tokens in tokenized:
        df_count.update(set(tokens))   # count each term once per document

    # Keep only the max_features most frequent terms
    vocabulary = [term for term, _ in df_count.most_common(max_features)]
    vocab_index = {term: idx for idx, term in enumerate(vocabulary)}
    V = len(vocabulary)

    # Step 3 — Compute TF matrix  (n_docs × V)
    tf_matrix = np.zeros((n_docs, V), dtype=float)
    for d_idx, tokens in enumerate(tokenized):
        if not tokens:
            continue
        term_counts = Counter(tokens)
        total       = len(tokens)
        for term, cnt in term_counts.items():
            if term in vocab_index:
                tf_matrix[d_idx, vocab_index[term]] = cnt / total

    # Step 4 — Compute IDF vector  (V,)
    idf = np.zeros(V, dtype=float)
    for t_idx, term in enumerate(vocabulary):
        df = df_count[term]
        idf[t_idx] = math.log((1 + n_docs) / (1 + df)) + 1.0

    # Step 5 — TF-IDF = TF * IDF  (broadcast multiply)
    tfidf_matrix = tf_matrix * idf[np.newaxis, :]               # (n_docs, V)

    # Step 6 — L2-normalise each row (standard TF-IDF normalisation)
    norms = np.sqrt(np.sum(tfidf_matrix ** 2, axis=1, keepdims=True))
    norms[norms == 0] = 1e-10
    tfidf_matrix = tfidf_matrix / norms

    return tfidf_matrix, vocabulary


# ═══════════════════════════════════════════════════════════════
# cosine similarity  
# ═══════════════════════════════════════════════════════════════

def _cosine_sim_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    dot   = A @ B.T                                             # (m, n)
    norms_A = np.sqrt(np.sum(A ** 2, axis=1, keepdims=True))   # (m, 1)
    norms_B = np.sqrt(np.sum(B ** 2, axis=1, keepdims=True))   # (n, 1)
    denom   = norms_A * norms_B.T                               # (m, n)
    denom[denom == 0] = 1e-10
    return dot / denom


# ═══════════════════════════════════════════════════════════════
# Method 1 – TF-IDF + Cosine Similarity 
# ═══════════════════════════════════════════════════════════════

class TFIDFRecommender:
    def __init__(self, max_features: int = 500):
        self.max_features  = max_features
        self.tfidf_matrix  = None   # np.ndarray (n_products, n_features)
        self.vocabulary    = None   # list of terms
        self.products      = None
        self.product_ids   = None

    def fit(self, products: pd.DataFrame) -> "TFIDFRecommender":
        self.products    = products.reset_index(drop=True)
        self.product_ids = self.products["product_id"].tolist()
        corpus = _build_corpus(self.products)
        self.tfidf_matrix, self.vocabulary = _compute_tfidf(
            corpus, max_features=self.max_features)
        return self

    def get_tfidf_matrix(self) -> np.ndarray:
        return self.tfidf_matrix

    def recommend(self, seed_product_ids: list, seen_ids: list,
                  top_n: int = 10) -> list:

        if not seed_product_ids:
            return []

        seed_indices = [self.product_ids.index(pid)
                        for pid in seed_product_ids
                        if pid in self.product_ids]
        if not seed_indices:
            return []

        seed_vecs = self.tfidf_matrix[seed_indices]            # (n_seeds, V)

        # Cosine similarity of seeds vs. all products — manual
        all_sims = _cosine_sim_cross(seed_vecs, self.tfidf_matrix)  # (n_seeds, n_products)
        avg_sims = all_sims.mean(axis=0)                             # (n_products,)

        # Extract top keywords from averaged seed vector
        avg_seed_vec  = seed_vecs.mean(axis=0)                       # (V,)
        top_kw_idx    = np.argsort(avg_seed_vec)[::-1][:5]
        top_keywords  = [self.vocabulary[i] for i in top_kw_idx
                         if avg_seed_vec[i] > 0]

        results = []
        for idx, sim in enumerate(avg_sims):
            pid = self.product_ids[idx]
            if pid in seen_ids:
                continue
            results.append((pid, float(sim), top_keywords))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]


# ═══════════════════════════════════════════════════════════════
# Method 2 – Feature Vector Similarity 
# ═══════════════════════════════════════════════════════════════

def _one_hot_encode(series: pd.Series, prefix: str) -> np.ndarray:

    categories = sorted(series.unique().tolist())
    cat_index  = {c: i for i, c in enumerate(categories)}
    result     = np.zeros((len(series), len(categories)), dtype=float)
    for row_idx, val in enumerate(series):
        if val in cat_index:
            result[row_idx, cat_index[val]] = 1.0
    return result, categories


def _min_max_scale(arr: np.ndarray) -> np.ndarray:
    col_min = arr.min(axis=0)
    col_max = arr.max(axis=0)
    denom   = col_max - col_min
    denom[denom == 0] = 1e-10          # avoid /0 for constant columns
    return (arr - col_min) / denom


class FeatureVectorRecommender:
    def __init__(self):
        self.feature_matrix = None   # np.ndarray (n_products, n_features)
        self.products        = None
        self.product_ids     = None

    def fit(self, products: pd.DataFrame) -> "FeatureVectorRecommender":
        self.products    = products.reset_index(drop=True)
        self.product_ids = self.products["product_id"].tolist()

        # One-hot encode category and brand — manual
        cat_enc,   _ = _one_hot_encode(self.products["category"], prefix="cat")
        brand_enc, _ = _one_hot_encode(self.products["brand"],    prefix="brand")

        # Min-max normalise price and rating — manual
        numeric = self.products[["price", "rating"]].values.astype(float)
        numeric_scaled = _min_max_scale(numeric)

        # Concatenate into one feature matrix per product
        self.feature_matrix = np.hstack([cat_enc, brand_enc, numeric_scaled])
        return self

    def recommend(self, seed_product_ids: list, seen_ids: list,
                  top_n: int = 10) -> list:
        if not seed_product_ids:
            return []

        seed_indices = [self.product_ids.index(pid)
                        for pid in seed_product_ids
                        if pid in self.product_ids]
        if not seed_indices:
            return []

        seed_vecs = self.feature_matrix[seed_indices]           # (n_seeds, n_feats)
        all_sims  = _cosine_sim_cross(seed_vecs, self.feature_matrix)  # manual cosine
        avg_sims  = all_sims.mean(axis=0)                       # (n_products,)

        results = []
        for idx, sim in enumerate(avg_sims):
            pid = self.product_ids[idx]
            if pid in seen_ids:
                continue
            results.append((pid, float(sim), "structured attributes"))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]


# ═══════════════════════════════════════════════════════════════
# Singleton cache & unified dispatcher
# ═══════════════════════════════════════════════════════════════

_tfidf_rec: TFIDFRecommender        = None
_feat_rec:  FeatureVectorRecommender = None


def get_fitted_recommenders(products: pd.DataFrame):
    global _tfidf_rec, _feat_rec
    if _tfidf_rec is None:
        _tfidf_rec = TFIDFRecommender().fit(products)
    if _feat_rec is None:
        _feat_rec  = FeatureVectorRecommender().fit(products)
    return _tfidf_rec, _feat_rec


def reset_cb_recommenders():
    global _tfidf_rec, _feat_rec
    _tfidf_rec = None
    _feat_rec  = None


def get_cb_recommendations(products: pd.DataFrame,
                            seed_product_ids: list,
                            seen_ids: list,
                            method: str = "tfidf",
                            top_n: int = 10) -> list:
    tfidf_rec, feat_rec = get_fitted_recommenders(products)
    if method == "tfidf":
        return tfidf_rec.recommend(seed_product_ids, seen_ids, top_n)
    elif method == "feature":
        return feat_rec.recommend(seed_product_ids, seen_ids, top_n)
    else:
        raise ValueError(f"Unknown CB method: {method}")
