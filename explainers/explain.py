import pandas as pd


# ═══════════════════════════════════════════════════════════════
# CF Explanations
# ═══════════════════════════════════════════════════════════════

def explain_cf(product_id: int,
               score: float,
               info: dict,
               method: str,
               products: pd.DataFrame,
               ratings: pd.DataFrame) -> str:
    prod_name = _product_name(product_id, products)

    if method == "user_pearson":
        if info:
            avg_sim    = sum(info.values()) / len(info)
            avg_rating = _avg_rating_by_users(product_id, list(info.keys()), ratings)
            return (
                f"Recommended because users similar to you "
                f"(avg Pearson similarity: {avg_sim:.3f}) also rated "
                f"'{prod_name}' highly (avg rating: {avg_rating:.1f}/5)."
            )
        return "Recommended based on the preferences of users with similar rating patterns to yours."

    elif method == "user_jaccard":
        if info:
            avg_sim    = sum(info.values()) / len(info)
            avg_rating = _avg_rating_by_users(product_id, list(info.keys()), ratings)
            return (
                f"Recommended because users who interacted with similar items "
                f"(avg Jaccard similarity: {avg_sim:.3f}) also engaged with "
                f"'{prod_name}' (avg rating: {avg_rating:.1f}/5)."
            )
        return "Recommended based on users who share similar interaction histories with you."

    elif method == "item_cosine":
        if info and isinstance(info, dict):
            top_seed = max(info, key=info.get)
            seed_name = _product_name(top_seed, products)
            return (
                f"Recommended because '{prod_name}' is highly similar "
                f"(adjusted cosine: {info[top_seed]:.3f}) to '{seed_name}', "
                f"which you have rated."
            )
        return "Recommended because it is similar to products you have rated highly."

    elif method == "mf_sgd":
        return (
            f"Recommended by Matrix Factorisation (SGD): the latent factor "
            f"model predicts a strong interest in '{prod_name}' based on "
            f"patterns learned across all users and items."
        )

    return "Recommended based on collaborative filtering."


# ═══════════════════════════════════════════════════════════════
# Content-Based Explanations
# ═══════════════════════════════════════════════════════════════

def explain_cb(product_id: int,
               score: float,
               info,
               method: str,
               products: pd.DataFrame) -> str:
    prod_name = _product_name(product_id, products)

    if method == "tfidf":
        keywords = info if isinstance(info, list) else []
        kw_str   = ", ".join(keywords[:4]) if keywords else "shared keywords"
        return (
            f"Recommended because '{prod_name}' shares key terms "
            f"[{kw_str}] with products you have purchased or liked."
        )

    elif method == "feature":
        row = products[products["product_id"] == product_id]
        if not row.empty:
            r = row.iloc[0]
            return (
                f"Recommended because '{prod_name}' has a similar "
                f"category ({r['category']}), brand ({r['brand']}), "
                f"and price range (EGP {r['price']:.0f}) to items in your history."
            )
        return f"Recommended because it shares structured attributes with your liked items."

    return "Recommended based on content similarity."


# ═══════════════════════════════════════════════════════════════
# Knowledge-Based Explanations
# ═══════════════════════════════════════════════════════════════

def explain_kb(product_id: int,
               score: float,
               info: dict,
               method: str,
               products: pd.DataFrame) -> str:
    prod_name = _product_name(product_id, products)

    if method == "constraint":
        budget_str   = info.get("budget",   "within budget")
        cat_str      = info.get("category", "preferred category")
        rating_str   = info.get("rating",   "meets minimum rating")
        brand_str    = info.get("brand",    None)
        brand_clause = f" brand ({brand_str})," if brand_str else ""
        return (
            f"Recommended because '{prod_name}' fits your budget "
            f"({budget_str}),{brand_clause} matches your preferred "
            f"category ({cat_str}), and has a rating of {rating_str}."
        )

    elif method == "cbr":
        ref_name   = info.get("reference",        "your selected reference product")
        cat_match  = info.get("category_match",   False)
        brand_match= info.get("brand_match",      False)
        price_prox = info.get("price_proximity",  0.0)
        parts = []
        if cat_match:   parts.append("same category")
        if brand_match: parts.append("same brand")
        parts.append(f"price proximity {price_prox:.0%}")
        return (
            f"Recommended because '{prod_name}' closely matches "
            f"'{ref_name}' in {', '.join(parts)}."
        )

    return "Recommended based on your stated requirements."


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _product_name(pid: int, products: pd.DataFrame) -> str:
    row = products[products["product_id"] == pid]
    return row.iloc[0]["name"] if not row.empty else f"Product #{pid}"


def _avg_rating_by_users(pid: int, user_ids: list, ratings: pd.DataFrame) -> float:
    mask = (ratings["product_id"] == pid) & (ratings["user_id"].isin(user_ids))
    vals = ratings.loc[mask, "rating"]
    return float(vals.mean()) if not vals.empty else 3.0


# ═══════════════════════════════════════════════════════════════
# Unified explanation builder
# ═══════════════════════════════════════════════════════════════

def build_explanation(product_id: int,
                      score: float,
                      info,
                      approach: str,   # 'cf' | 'cb' | 'kb'
                      method: str,
                      products: pd.DataFrame,
                      ratings: pd.DataFrame) -> str:
    if approach == "cf":
        return explain_cf(product_id, score, info, method, products, ratings)
    elif approach == "cb":
        return explain_cb(product_id, score, info, method, products)
    elif approach == "kb":
        return explain_kb(product_id, score, info, method, products)
    return "No explanation available."
