import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from recommenders.collaborative import batch_cf_recommendations


# ═══════════════════════════════════════════════════════════════
# 1. RMSE
# ═══════════════════════════════════════════════════════════════

def compute_rmse(test_ratings: pd.DataFrame,
                 matrix: pd.DataFrame,
                 get_recommendations_fn,
                 method: str) -> float:
    squared_errors = []
    for uid, grp in test_ratings.groupby("user_id"):
        if uid not in matrix.index:
            continue
        recs = get_recommendations_fn(matrix, uid, method=method, top_n=len(matrix.columns))
        pred_dict = {pid: score for pid, score, *_ in recs}
        for _, row in grp.iterrows():
            pid = row["product_id"]
            actual = row["rating"]
            if pid in pred_dict:
                pred = pred_dict[pid]
                # Clip prediction to [1,5]
                pred = max(1.0, min(5.0, pred))
                squared_errors.append((actual - pred) ** 2)
    if not squared_errors:
        return float("nan")
    return float(np.sqrt(np.mean(squared_errors)))


# ═══════════════════════════════════════════════════════════════
# 2. Precision@K and Recall@K
# ═══════════════════════════════════════════════════════════════

def compute_precision_recall_at_k(test_ratings: pd.DataFrame,
                                   matrix: pd.DataFrame,
                                   get_recommendations_fn,
                                   method: str,
                                   k: int = 5,
                                   relevant_threshold: float = 4.0) -> tuple:
    precisions, recalls = [], []
    for uid, grp in test_ratings.groupby("user_id"):
        if uid not in matrix.index:
            continue
        relevant = set(grp.loc[grp["rating"] >= relevant_threshold, "product_id"].tolist())
        if not relevant:
            continue
        recs = get_recommendations_fn(matrix, uid, method=method, top_n=k)
        rec_ids = [pid for pid, *_ in recs[:k]]
        hits = len([pid for pid in rec_ids if pid in relevant])
        precisions.append(hits / k if k > 0 else 0)
        recalls.append(hits / len(relevant) if relevant else 0)

    if not precisions:
        return float("nan"), float("nan")
    return float(np.mean(precisions)), float(np.mean(recalls))


# ═══════════════════════════════════════════════════════════════
# 3. Coverage
# ═══════════════════════════════════════════════════════════════

def compute_coverage(matrix: pd.DataFrame,
                     all_product_ids: list,
                     get_recommendations_fn,
                     method: str,
                     top_n: int = 10) -> float:

    recommended = set()
    for uid in matrix.index:
        recs = get_recommendations_fn(matrix, uid, method=method, top_n=top_n)
        recommended.update(pid for pid, *_ in recs)
    if not all_product_ids:
        return 0.0
    return len(recommended) / len(all_product_ids)


# ═══════════════════════════════════════════════════════════════
# 4. Intra-List Diversity
# ═══════════════════════════════════════════════════════════════

def compute_diversity(matrix: pd.DataFrame,
                      tfidf_matrix: np.ndarray,
                      product_ids_ordered: list,
                      get_recommendations_fn,
                      method: str,
                      top_n: int = 10) -> float:

    pid_to_idx = {pid: idx for idx, pid in enumerate(product_ids_ordered)}
    diversities = []

    for uid in matrix.index:
        recs = get_recommendations_fn(matrix, uid, method=method, top_n=top_n)
        rec_ids = [pid for pid, *_ in recs]
        idxs = [pid_to_idx[pid] for pid in rec_ids if pid in pid_to_idx]
        if len(idxs) < 2:
            continue
        vecs = tfidf_matrix[idxs]
        sims = cosine_similarity(vecs)
        # distance = 1 - similarity; take upper triangle
        n = len(idxs)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(1.0 - sims[i, j])
        if dists:
            diversities.append(np.mean(dists))

    if not diversities:
        return float("nan")
    return float(np.mean(diversities))


# ═══════════════════════════════════════════════════════════════
# Full comparative evaluation
# ═══════════════════════════════════════════════════════════════

def run_full_evaluation(ds,
                        cb_recs: dict,
                        kb_recs: dict,
                        tfidf_matrix: np.ndarray,
                        product_ids_ordered: list,
                        k: int = 5) -> pd.DataFrame:

    train_matrix = ds.train_matrix()
    test_ratings = ds.test_ratings
    all_pids     = ds.products["product_id"].tolist()

    rows = []

    # ── CF methods ────────────────────────────────────────────
    # Pre-compute all users' recs once per method (avoids re-training MF N×4 times)
    def _make_cache_fn(cache):
        def _fn(matrix, uid, method, top_n):
            return cache.get(uid, [])[:top_n]
        return _fn

    for method_label, method_key in [
        ("User-Based Pearson",   "user_pearson"),
        ("User-Based Jaccard",   "user_jaccard"),
        ("Item-Based Adj.Cosine","item_cosine"),
        ("MF Gradient Descent",  "mf_sgd"),
    ]:
        cf_cache   = batch_cf_recommendations(train_matrix, method_key,
                                              top_n=len(train_matrix.columns))
        cached_fn  = _make_cache_fn(cf_cache)

        rmse = compute_rmse(test_ratings, train_matrix, cached_fn, method_key)
        prec, rec = compute_precision_recall_at_k(
            test_ratings, train_matrix, cached_fn, method_key, k=k)
        cov  = compute_coverage(train_matrix, all_pids, cached_fn, method_key)
        div  = compute_diversity(train_matrix, tfidf_matrix,
                                  product_ids_ordered, cached_fn, method_key)
        rows.append({
            "Method":       method_label,
            "RMSE":         round(rmse, 4) if not np.isnan(rmse) else None,
            f"Precision@{k}": round(prec, 4) if not np.isnan(prec) else None,
            f"Recall@{k}":    round(rec,  4) if not np.isnan(rec)  else None,
            "Coverage":     round(cov,  4),
            "Diversity":    round(div,  4) if not np.isnan(div) else None,
        })

    # ── Content-Based (average of TFIDF + feature) ────────────
    def _cb_fn_wrapper(matrix, uid, method, top_n):
        return cb_recs.get(method, {}).get(uid, [])[:top_n]

    # Use pre-computed CB recs for precision/recall/coverage/diversity
    for cb_label, cb_key in [("Content-Based TF-IDF", "tfidf"),
                               ("Content-Based Feature", "feature")]:
        uid_recs = cb_recs.get(cb_key, {})
        prec_list, rec_list, cov_set, div_list = [], [], set(), []
        for uid, grp in test_ratings.groupby("user_id"):
            relevant = set(grp.loc[grp["rating"] >= 4.0, "product_id"].tolist())
            recs = uid_recs.get(uid, [])[:k]
            rec_ids = [pid for pid, *_ in recs]
            cov_set.update(rec_ids)
            hits = len([pid for pid in rec_ids if pid in relevant])
            if relevant:
                prec_list.append(hits / k)
                rec_list.append(hits / len(relevant))
            pid_to_idx = {pid: i for i, pid in enumerate(product_ids_ordered)}
            idxs = [pid_to_idx[p] for p in [pid for pid, *_ in uid_recs.get(uid, [])[:k]]
                    if p in pid_to_idx]
            if len(idxs) >= 2:
                vecs = tfidf_matrix[idxs]
                sims = cosine_similarity(vecs)
                n = len(idxs)
                d = [1.0 - sims[i][j] for i in range(n) for j in range(i+1,n)]
                div_list.append(np.mean(d))

        rows.append({
            "Method":         cb_label,
            "RMSE":           None,
            f"Precision@{k}": round(np.mean(prec_list), 4) if prec_list else None,
            f"Recall@{k}":    round(np.mean(rec_list),  4) if rec_list else None,
            "Coverage":       round(len(cov_set) / len(all_pids), 4),
            "Diversity":      round(np.mean(div_list), 4) if div_list else None,
        })

    # ── Knowledge-Based (constraint + cbr) ────────────────────
    for kb_label, kb_key in [("Knowledge-Based Constraint", "constraint"),
                               ("Knowledge-Based CBR",        "cbr")]:
        uid_recs = kb_recs.get(kb_key, {})
        prec_list, rec_list, cov_set, div_list = [], [], set(), []
        for uid, grp in test_ratings.groupby("user_id"):
            relevant = set(grp.loc[grp["rating"] >= 4.0, "product_id"].tolist())
            recs = uid_recs.get(uid, [])[:k]
            rec_ids = [pid for pid, *_ in recs]
            cov_set.update(rec_ids)
            hits = len([pid for pid in rec_ids if pid in relevant])
            if relevant:
                prec_list.append(hits / k)
                rec_list.append(hits / len(relevant))
            pid_to_idx = {pid: i for i, pid in enumerate(product_ids_ordered)}
            idxs = [pid_to_idx[p] for p in rec_ids if p in pid_to_idx]
            if len(idxs) >= 2:
                vecs = tfidf_matrix[idxs]
                sims = cosine_similarity(vecs)
                n = len(idxs)
                d = [1.0 - sims[i][j] for i in range(n) for j in range(i+1,n)]
                div_list.append(np.mean(d))

        rows.append({
            "Method":         kb_label,
            "RMSE":           None,
            f"Precision@{k}": round(np.mean(prec_list), 4) if prec_list else None,
            f"Recall@{k}":    round(np.mean(rec_list),  4) if rec_list else None,
            "Coverage":       round(len(cov_set) / len(all_pids), 4),
            "Diversity":      round(np.mean(div_list), 4) if div_list else None,
        })

    return pd.DataFrame(rows)


def generate_analysis_paragraph(results_df: pd.DataFrame, k: int = 5) -> str:
    prec_col = f"Precision@{k}"

    def best(col):
        sub = results_df.dropna(subset=[col])
        if sub.empty:
            return "N/A", "N/A"
        idx = sub[col].idxmax()
        return sub.loc[idx, "Method"], sub.loc[idx, col]

    best_rmse_method = results_df.dropna(subset=["RMSE"])
    if not best_rmse_method.empty:
        idx = best_rmse_method["RMSE"].idxmin()
        rmse_winner = best_rmse_method.loc[idx, "Method"]
        rmse_val    = best_rmse_method.loc[idx, "RMSE"]
    else:
        rmse_winner, rmse_val = "N/A", "N/A"

    prec_winner, prec_val = best(prec_col)
    cov_winner,  cov_val  = best("Coverage")
    div_winner,  div_val  = best("Diversity")

    para = (
        f"Comparative Analysis: "
        f"Across all evaluated methods, **{rmse_winner}** achieved the lowest RMSE "
        f"({rmse_val:.4f}), indicating it predicts user ratings most accurately. "
        f"In terms of relevance, **{prec_winner}** led on Precision@{k} ({prec_val:.4f}), "
        f"meaning it returned the highest proportion of truly relevant items in its top-{k} lists. "
        f"**{cov_winner}** showed the broadest catalog coverage ({cov_val:.1%}), "
        f"exposing users to the widest variety of items — an important anti-popularity-bias metric. "
        f"Finally, **{div_winner}** produced the most diverse recommendation lists "
        f"(avg intra-list cosine distance: {div_val:.4f}), which reduces redundancy in suggestions. "
        f"Knowledge-Based methods generally excel when explicit user constraints are provided, "
        f"while Collaborative Filtering methods benefit from dense rating data. "
        f"Content-Based methods strike a balance and are most effective for new users with limited rating history."
    )
    return para
