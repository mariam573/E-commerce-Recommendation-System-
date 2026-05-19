import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# Method 1 – Constraint-Based Filtering
# ═══════════════════════════════════════════════════════════════

def constraint_based(products: pd.DataFrame,
                     budget_max: float,
                     preferred_categories: list,
                     preferred_brands: list = None,
                     min_rating: float = 0.0,
                     seen_ids: list = None,
                     top_n: int = 10) -> list:

    df = products.copy()
    if seen_ids:
        df = df[~df["product_id"].isin(seen_ids)]

    # Apply hard filters
    df = df[df["price"] <= budget_max]
    if preferred_categories:
        df = df[df["category"].isin(preferred_categories)]
    if preferred_brands:
        df = df[df["brand"].isin(preferred_brands)]
    df = df[df["rating"] >= min_rating]

    # Rank by rating descending
    df = df.sort_values("rating", ascending=False)

    results = []
    for _, row in df.head(top_n).iterrows():
        constraints = {
            "budget":   f"EGP {row['price']:.0f} ≤ EGP {budget_max:.0f}",
            "category": row["category"],
            "rating":   f"{row['rating']:.1f} ≥ {min_rating:.1f}",
        }
        if preferred_brands:
            constraints["brand"] = row["brand"]
        results.append((int(row["product_id"]), float(row["rating"]), constraints))

    return results


# ═══════════════════════════════════════════════════════════════
# Method 2 – Case-Based Reasoning (CBR)
# ═══════════════════════════════════════════════════════════════

def case_based_reasoning(products: pd.DataFrame,
                         reference_product_id: int,
                         seen_ids: list = None,
                         top_n: int = 10) -> list:
    ref_row = products[products["product_id"] == reference_product_id]
    if ref_row.empty:
        return []
    ref = ref_row.iloc[0]

    price_range  = products["price"].max() - products["price"].min()
    rating_range = products["rating"].max() - products["rating"].min()
    if price_range  == 0: price_range  = 1.0
    if rating_range == 0: rating_range = 1.0

    results = []
    for _, row in products.iterrows():
        if row["product_id"] == reference_product_id:
            continue
        if seen_ids and row["product_id"] in seen_ids:
            continue

        cat_match    = 1.0 if row["category"] == ref["category"] else 0.0
        brand_match  = 1.0 if row["brand"]    == ref["brand"]    else 0.0
        price_prox   = 1.0 - abs(row["price"]  - ref["price"])  / price_range
        rating_prox  = 1.0 - abs(row["rating"] - ref["rating"]) / rating_range

        similarity = (0.40 * cat_match +
                      0.20 * brand_match +
                      0.20 * price_prox +
                      0.20 * rating_prox)

        match_info = {
            "category_match":  bool(cat_match),
            "brand_match":     bool(brand_match),
            "price_proximity": round(price_prox, 3),
            "rating_proximity": round(rating_prox, 3),
            "reference":       ref["name"],
        }
        results.append((int(row["product_id"]), round(float(similarity), 4), match_info))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


# ═══════════════════════════════════════════════════════════════
# Unified dispatcher
# ═══════════════════════════════════════════════════════════════

def get_kb_recommendations(products: pd.DataFrame,
                            method: str = "constraint",
                            seen_ids: list = None,
                            top_n: int = 10,
                            **kwargs) -> list:
    if method == "constraint":
        return constraint_based(
            products,
            budget_max=kwargs.get("budget_max", float("inf")),
            preferred_categories=kwargs.get("preferred_categories", []),
            preferred_brands=kwargs.get("preferred_brands", None),
            min_rating=kwargs.get("min_rating", 0.0),
            seen_ids=seen_ids,
            top_n=top_n,
        )
    elif method == "cbr":
        return case_based_reasoning(
            products,
            reference_product_id=kwargs.get("reference_product_id"),
            seen_ids=seen_ids,
            top_n=top_n,
        )
    else:
        raise ValueError(f"Unknown KB method: {method}")
