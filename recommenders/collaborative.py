import math
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# Low-level helpers 
# ═══════════════════════════════════════════════════════════════

def _top_n(scores: dict, n: int) -> list:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


def _user_means(matrix: pd.DataFrame) -> dict:
    means = {}
    for uid in matrix.index:
        row = matrix.loc[uid].dropna()
        # Manual mean: sum / count
        total = 0.0
        count = 0
        for v in row:
            total += float(v)
            count += 1
        means[uid] = total / count if count > 0 else 0.0
    return means


# ═══════════════════════════════════════════════════════════════
# Method 1 – User-Based CF with Pearson Correlation  
# ═══════════════════════════════════════════════════════════════

def _pearson_sim(matrix: pd.DataFrame, uid_a: int, uid_b: int,
                 means: dict) -> float:
    ra_mean = means[uid_a]
    rb_mean = means[uid_b]

    # Co-rated items
    rated_a = set(matrix.loc[uid_a].dropna().index)
    rated_b = set(matrix.loc[uid_b].dropna().index)
    common  = rated_a & rated_b

    if not common:
        return 0.0

    numerator   = 0.0
    sq_sum_a    = 0.0
    sq_sum_b    = 0.0

    for p in common:
        dev_a    = float(matrix.loc[uid_a, p]) - ra_mean
        dev_b    = float(matrix.loc[uid_b, p]) - rb_mean
        numerator += dev_a * dev_b
        sq_sum_a  += dev_a * dev_a
        sq_sum_b  += dev_b * dev_b

    denom = (sq_sum_a ** 0.5) * (sq_sum_b ** 0.5)
    if denom < 1e-10:
        return 0.0
    return numerator / denom


def user_based_pearson(matrix: pd.DataFrame, user_id: int,
                       top_k: int = 10, top_n: int = 10) -> list:
    if user_id not in matrix.index:
        return []

    means = _user_means(matrix)
    ra    = means[user_id]

    # Step 2: compute Pearson similarity to every other user
    sim_scores = {}
    for uid in matrix.index:
        if uid == user_id:
            continue
        sim_scores[uid] = _pearson_sim(matrix, user_id, uid, means)

    # Step 3: top-K neighbours (positive similarity only)
    neighbours = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
    neighbours = [(uid, s) for uid, s in neighbours if s > 0][:top_k]

    seen     = set(matrix.loc[user_id].dropna().index)
    scores   = {}
    sim_info = {}

    for pid in matrix.columns:
        if pid in seen:
            continue

        weighted_sum = 0.0
        total_weight = 0.0
        contributors = {}

        for uid_b, sim in neighbours:
            rating = matrix.loc[uid_b, pid]
            if pd.isna(rating):
                continue
            rb = means[uid_b]
            # Step 4 numerator accumulation
            weighted_sum += sim * (float(rating) - rb)
            total_weight += abs(sim)
            contributors[uid_b] = round(sim, 4)

        if total_weight > 0:
            scores[pid]   = ra + weighted_sum / total_weight
            sim_info[pid] = contributors

    ranked = _top_n(scores, top_n)
    return [(pid, score, sim_info.get(pid, {})) for pid, score in ranked]


# ═══════════════════════════════════════════════════════════════
# Method 2 – Jaccard Similarity  (binary interaction)
# ═══════════════════════════════════════════════════════════════

def _jaccard_sim(set_a: set, set_b: set) -> float:
    intersection = len(set_a & set_b)           # |A ∩ B|
    union        = len(set_a | set_b)           # |A ∪ B|
    return intersection / union if union > 0 else 0.0


def user_based_jaccard(matrix: pd.DataFrame, user_id: int,
                       top_k: int = 10, top_n: int = 10) -> list:
    if user_id not in matrix.index:
        return []

    # Build binary item sets per user
    item_sets = {}
    for uid in matrix.index:
        item_sets[uid] = set(matrix.loc[uid].dropna().index)

    set_a = item_sets[user_id]

    # Step 2: compute Jaccard similarity
    sim_scores = {}
    for uid in matrix.index:
        if uid == user_id:
            continue
        sim_scores[uid] = _jaccard_sim(set_a, item_sets[uid])

    # Step 3: top-K neighbours
    neighbours = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
    neighbours = [(uid, s) for uid, s in neighbours if s > 0][:top_k]

    seen   = set_a
    scores = {}
    sim_info = {}

    for pid in matrix.columns:
        if pid in seen:
            continue

        weighted_sum = 0.0
        total_weight = 0.0
        contributors = {}

        for uid_b, sim in neighbours:
            # Binary: 1 if the neighbour interacted with this item, else skip
            if pid not in item_sets[uid_b]:
                continue
            weighted_sum += sim
            total_weight += sim
            contributors[uid_b] = round(sim, 4)

        if total_weight > 0:
            scores[pid]   = weighted_sum / total_weight
            sim_info[pid] = contributors

    ranked = _top_n(scores, top_n)
    return [(pid, score, sim_info.get(pid, {})) for pid, score in ranked]


# ═══════════════════════════════════════════════════════════════
# Method 3 – Item-Based CF with Adjusted Cosine Similarity 
# ═══════════════════════════════════════════════════════════════

def _adjusted_cosine_sim(matrix: pd.DataFrame, item_i, item_j,
                          user_means: dict) -> float:
    numerator  = 0.0
    sq_sum_i   = 0.0
    sq_sum_j   = 0.0

    for uid in matrix.index:
        ri = matrix.loc[uid, item_i]
        rj = matrix.loc[uid, item_j]
        if pd.isna(ri) or pd.isna(rj):
            continue
        mean_u   = user_means[uid]
        dev_i    = float(ri) - mean_u
        dev_j    = float(rj) - mean_u
        numerator += dev_i * dev_j
        sq_sum_i  += dev_i * dev_i
        sq_sum_j  += dev_j * dev_j

    denom = (sq_sum_i ** 0.5) * (sq_sum_j ** 0.5)
    if denom < 1e-10:
        return 0.0
    return numerator / denom


def item_based_cosine(matrix: pd.DataFrame, user_id: int,
                      top_n: int = 10) -> list:
    if user_id not in matrix.index:
        return []

    user_means = _user_means(matrix)
    rated_by_user = matrix.loc[user_id].dropna()

    if rated_by_user.empty:
        return []

    seen     = set(rated_by_user.index)
    scores   = {}
    sim_info = {}

    for pid in matrix.columns:
        if pid in seen:
            continue

        weighted_sum = 0.0
        total_weight = 0.0
        contributors = {}

        for seed_item in seen:
            sim = _adjusted_cosine_sim(matrix, seed_item, pid, user_means)
            if sim <= 0:
                continue
            r_u_i         = float(rated_by_user[seed_item])
            weighted_sum += sim * r_u_i
            total_weight += abs(sim)
            contributors[seed_item] = round(sim, 4)

        if total_weight > 0:
            scores[pid]   = weighted_sum / total_weight
            sim_info[pid] = contributors

    ranked = _top_n(scores, top_n)
    return [(pid, score, sim_info.get(pid, {})) for pid, score in ranked]


# ═══════════════════════════════════════════════════════════════
# Method 4 – Matrix Factorization via Gradient Descent  
# ═══════════════════════════════════════════════════════════════

def _sgd_matrix_factorization(R: np.ndarray,
                               n_factors: int = 10,
                               n_epochs:  int = 50,
                               alpha:     float = 0.005,
                               lmbda:     float = 0.02,
                               seed:      int = 42) -> tuple:

    n_users, n_items = R.shape

    # Step 1 — random initialisation
    rng = np.random.default_rng(seed)
    P   = rng.random((n_users, n_factors)) * 0.1   # (n_users, k)
    Q   = rng.random((n_items, n_factors)) * 0.1   # (n_items, k)

    # Collect indices of known (non-NaN) ratings once
    known = [(u, i) for u in range(n_users)
                     for i in range(n_items)
                     if not np.isnan(R[u, i])]

    # Step 2+3 — iterative gradient descent
    for _ in range(n_epochs):
        for u, i in known:
            # Predicted rating: dot product of latent vectors (manual sum)
            pred_ui = 0.0
            for f in range(n_factors):
                pred_ui += P[u, f] * Q[i, f]

            # Error between actual and predicted
            e_ui = R[u, i] - pred_ui

            # Update each latent factor for user u and item i
            # Formula: P[u,f] += α*(2*e*Q[i,f] - λ*P[u,f])
            #          Q[i,f] += α*(2*e*P[u,f] - λ*Q[i,f])
            for f in range(n_factors):
                p_uf_old = P[u, f]
                P[u, f] += alpha * (2 * e_ui * Q[i, f] - lmbda * P[u, f])
                Q[i, f] += alpha * (2 * e_ui * p_uf_old - lmbda * Q[i, f])

    return P, Q


def matrix_factorization_sgd(matrix: pd.DataFrame, user_id: int,
                              n_factors: int = 10,
                              n_epochs:  int = 50,
                              alpha:     float = 0.005,
                              lmbda:     float = 0.02,
                              top_n:     int = 10) -> list:
    if user_id not in matrix.index:
        return []

    R = matrix.values.astype(float)       # preserve NaN for unknown entries
    P, Q = _sgd_matrix_factorization(R, n_factors=n_factors,
                                     n_epochs=n_epochs,
                                     alpha=alpha, lmbda=lmbda)

    user_idx  = list(matrix.index).index(user_id)
    seen      = set(matrix.loc[user_id].dropna().index)

    scores = {}
    for j, pid in enumerate(matrix.columns):
        if pid in seen:
            continue
        # Manual dot product: P[u] · Q[i]
        pred = 0.0
        for f in range(P.shape[1]):
            pred += P[user_idx, f] * Q[j, f]
        scores[pid] = pred

    ranked = _top_n(scores, top_n)
    return [(pid, score, {}) for pid, score in ranked]


# ═══════════════════════════════════════════════════════════════
# Batch helpers  (used by evaluation to avoid per-user re-training)
# ═══════════════════════════════════════════════════════════════

def _fast_item_cosine_batch(matrix: pd.DataFrame, top_n: int = 50) -> dict:

    R          = matrix.values.astype(float)          # (n_users, n_items)
    item_ids   = list(matrix.columns)
    user_ids   = list(matrix.index)
    valid_mask = ~np.isnan(R)                          # (n_users, n_items) bool

    # User means (over rated items only)
    row_sums   = np.where(valid_mask, R, 0.0).sum(axis=1)
    row_counts = valid_mask.sum(axis=1).astype(float)
    row_means  = np.where(row_counts > 0,
                          row_sums / np.where(row_counts > 0, row_counts, 1),
                          0.0)

    # Mean-centred ratings; zero where not rated
    R_centered = R - row_means[:, np.newaxis]
    R_masked   = np.where(valid_mask, R_centered, 0.0)
    sq_masked  = R_masked ** 2

    # Adjusted cosine similarity matrix — fully vectorised
    #   num[i,j]     = Σ_u dev_u_i * dev_u_j   (co-rated only, zeros elsewhere)
    #   denom_i[i,j] = Σ_u dev_u_i²             (co-rated only)
    #   denom_j[i,j] = Σ_u dev_u_j²             (co-rated only)
    num_mat    = R_masked.T @ R_masked                      # (n_items, n_items)
    denom_i    = sq_masked.T @ valid_mask.astype(float)     # (n_items, n_items)
    denom_j    = valid_mask.T.astype(float) @ sq_masked     # (n_items, n_items)
    denom      = np.sqrt(denom_i) * np.sqrt(denom_j)
    with np.errstate(invalid='ignore', divide='ignore'):
        sim_matrix = np.where(denom > 1e-10, num_mat / denom, 0.0)
    np.fill_diagonal(sim_matrix, 0.0)

    # Per-user recommendations
    result = {}
    for u_idx, uid in enumerate(user_ids):
        rated      = valid_mask[u_idx]
        seen_idx   = np.where(rated)[0]
        unseen_idx = np.where(~rated)[0]

        if len(seen_idx) == 0 or len(unseen_idx) == 0:
            result[uid] = []
            continue

        sims         = np.maximum(sim_matrix[seen_idx][:, unseen_idx], 0.0)
        ratings_seen = R[u_idx, seen_idx]
        weighted     = ratings_seen @ sims
        weight_sum   = sims.sum(axis=0)

        pred = np.where(weight_sum > 0,
                        weighted / np.where(weight_sum > 0, weight_sum, 1.0),
                        0.0)

        unseen_pids = [item_ids[j] for j in unseen_idx]
        scored = sorted(zip(unseen_pids, pred.tolist()),
                        key=lambda x: x[1], reverse=True)
        result[uid] = [(pid, s, {}) for pid, s in scored if s > 0][:top_n]

    return result


def batch_cf_recommendations(matrix: pd.DataFrame, method: str,
                              top_n: int = 50) -> dict:
    if method == "item_cosine":
        return _fast_item_cosine_batch(matrix, top_n=top_n)

    if method == "mf_sgd":
        R = matrix.values.astype(float)
        P, Q = _sgd_matrix_factorization(R, n_factors=10, n_epochs=20,
                                          alpha=0.005, lmbda=0.02)
        pred_matrix = P @ Q.T          # (n_users, n_items) — one vectorised call
        item_ids    = list(matrix.columns)
        result = {}
        for u_idx, uid in enumerate(matrix.index):
            seen   = set(matrix.loc[uid].dropna().index)
            scores = {pid: float(pred_matrix[u_idx, j])
                      for j, pid in enumerate(item_ids) if pid not in seen}
            ranked = _top_n(scores, top_n)
            result[uid] = [(pid, score, {}) for pid, score in ranked]
        return result

    return {uid: get_cf_recommendations(matrix, uid, method=method, top_n=top_n)
            for uid in matrix.index}


# ═══════════════════════════════════════════════════════════════
# Unified dispatcher
# ═══════════════════════════════════════════════════════════════

def get_cf_recommendations(matrix: pd.DataFrame, user_id: int,
                            method: str = "user_pearson",
                            top_n: int = 10) -> list:

    if method == "user_pearson":
        return user_based_pearson(matrix, user_id, top_n=top_n)
    elif method == "user_jaccard":
        return user_based_jaccard(matrix, user_id, top_n=top_n)
    elif method == "item_cosine":
        return item_based_cosine(matrix, user_id, top_n=top_n)
    elif method == "mf_sgd":
        return matrix_factorization_sgd(matrix, user_id, top_n=top_n)
    else:
        raise ValueError(
            f"Unknown CF method '{method}'. "
            "Choose from: 'user_pearson', 'user_jaccard', 'item_cosine', 'mf_sgd'."
        )
