import os, sys, math
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

# ── Make all sub-packages importable ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, make_response)

from utils.data_loader           import get_datastore, reset_datastore
from recommenders.collaborative  import get_cf_recommendations
from recommenders.content_based  import (get_cb_recommendations,
                                          get_fitted_recommenders,
                                          reset_cb_recommenders)
from recommenders.knowledge_based import get_kb_recommendations
from explainers.explain           import build_explanation
from evaluation.metrics           import (run_full_evaluation,
                                          generate_analysis_paragraph)

# ── App setup ─────────────────────────────────────────────────────────────────
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
STATIC_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = "aie425-recommender-secret"

# ── Load data once at startup ─────────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT, "data")

def _startup():
    reset_datastore()
    reset_cb_recommenders()
    ds = get_datastore(DATA_DIR)
    tfidf_rec, feat_rec = get_fitted_recommenders(ds.products)
    return ds, tfidf_rec, feat_rec

DS, TFIDF_REC, FEAT_REC = _startup()

CATEGORIES   = sorted(DS.products["category"].unique().tolist())
ALL_USER_IDS = sorted(DS.users["user_id"].tolist())

# ── Helpers ───────────────────────────────────────────────────────────────────

def _image_url(pid: int, category: str) -> str:
    specific = f"images/p{pid}.jpg"
    specific_path = os.path.join(STATIC_DIR, specific)
    if os.path.exists(specific_path):
        return f"/static/{specific}"
    cat_safe = category.replace(" ", "_")
    fallback = f"images/cat_{cat_safe}.jpg"
    fallback_path = os.path.join(STATIC_DIR, fallback)
    if os.path.exists(fallback_path):
        return f"/static/{fallback}"
    return "/static/images/placeholder.jpg"


def _star_html(rating: float) -> str:
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty


def _get_uid_from_cookie(req) -> int:
    try:
        return int(req.cookies.get("user_id", ALL_USER_IDS[0]))
    except Exception:
        return ALL_USER_IDS[0]


def _user_display(uid: int) -> str:
    row = DS.users[DS.users["user_id"] == uid]
    if row.empty:
        return f"User {uid}"
    return f"{row.iloc[0]['name']} (#{uid})"


def _enrich_products(df):
    rows = df.to_dict(orient="records")
    for r in rows:
        r["image_url"] = _image_url(r["product_id"], r["category"])
        r["stars"]     = _star_html(r["rating"])
        r["price_fmt"] = f"EGP {r['price']:,.0f}"
    return rows


def _build_user_history(uid: int) -> dict:
    uid = int(uid)  # guarantee plain Python int

    # Read directly from the raw DataFrames using explicit int cast on both sides
    purchases_for_user = DS.purchases[DS.purchases["user_id"].astype(int) == uid]
    ratings_for_user   = DS.ratings[DS.ratings["user_id"].astype(int) == uid].sort_values(
        "rating", ascending=False
    )

    purchased = []
    for _, row in purchases_for_user.iterrows():
        pid = int(row["product_id"])
        prod = DS.products[DS.products["product_id"].astype(int) == pid]
        if prod.empty:
            continue
        p = prod.iloc[0]
        date_str = ""
        try:
            d = row["date"]
            date_str = d.strftime("%Y-%m-%d") if pd.notna(d) else ""
        except Exception:
            date_str = str(row["date"])[:10]
        purchased.append({
            "product_id": pid,
            "name":       str(p["name"]),
            "category":   str(p["category"]),
            "brand":      str(p["brand"]),
            "price_fmt":  f"EGP {float(p['price']):,.0f}" if pd.notna(p["price"]) else "N/A",
            "image_url":  _image_url(pid, str(p["category"])),
            "url":        f"/product/{pid}",
            "date":       date_str,
        })

    rated = []
    for _, r_row in ratings_for_user.iterrows():
        pid = int(r_row["product_id"])
        prod = DS.products[DS.products["product_id"].astype(int) == pid]
        if prod.empty:
            continue
        p = prod.iloc[0]
        user_rating = float(r_row["rating"])
        rated.append({
            "product_id":  pid,
            "name":        str(p["name"]),
            "category":    str(p["category"]),
            "brand":       str(p["brand"]),
            "user_rating": user_rating,
            "stars":       _star_html(user_rating),
            "image_url":   _image_url(pid, str(p["category"])),
            "url":         f"/product/{pid}",
        })

    return {"purchased": purchased, "rated": rated}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/set_user", methods=["POST"])
def set_user():
    uid  = request.form.get("user_id", ALL_USER_IDS[0])
    next_url = request.form.get("next", "/")
    resp = make_response(redirect(next_url))
    resp.set_cookie("user_id", str(uid), max_age=60*60*24*30)
    return resp


@app.route("/")
def index():
    uid      = _get_uid_from_cookie(request)
    page     = int(request.args.get("page", 1))
    per_page = 24
    category = request.args.get("category", "")
    search   = request.args.get("q", "").strip().lower()
    sort_by  = request.args.get("sort", "")

    products = DS.products.copy()

    if category:
        products = products[products["category"] == category]
    if search:
        mask = (
            products["name"].str.lower().str.contains(search, na=False) |
            products["category"].str.lower().str.contains(search, na=False) |
            products["brand"].str.lower().str.contains(search, na=False) |
            products["tags"].str.lower().str.contains(search, na=False)
        )
        products = products[mask]

    if sort_by == "price_asc":
        products = products.sort_values("price")
    elif sort_by == "price_desc":
        products = products.sort_values("price", ascending=False)
    elif sort_by == "rating":
        products = products.sort_values("rating", ascending=False)

    total   = len(products)
    n_pages = max(1, math.ceil(total / per_page))
    page    = max(1, min(page, n_pages))
    start   = (page - 1) * per_page
    chunk   = products.iloc[start: start + per_page]

    enriched = _enrich_products(chunk)

    return render_template("index.html",
        products    = enriched,
        categories  = CATEGORIES,
        cur_cat     = category,
        search_q    = request.args.get("q", ""),
        sort_by     = sort_by,
        page        = page,
        n_pages     = n_pages,
        total       = total,
        uid         = uid,
        all_users   = [(u, _user_display(u)) for u in ALL_USER_IDS],
        user_display= _user_display(uid),
    )


@app.route("/product/<int:pid>")
def product_detail(pid):
    uid = _get_uid_from_cookie(request)
    row = DS.products[DS.products["product_id"] == pid]
    if row.empty:
        return redirect(url_for("index"))

    product = row.iloc[0].to_dict()
    product["image_url"] = _image_url(pid, product["category"])
    product["stars"]     = _star_html(product["rating"])
    product["price_fmt"] = f"EGP {product['price']:,.0f}"
    product["tags_list"] = [t.strip() for t in str(product.get("tags","")).split(",")]

    # Related products (same category, top 8 by rating)
    related_df = DS.products[
        (DS.products["category"] == product["category"]) &
        (DS.products["product_id"] != pid)
    ].sort_values("rating", ascending=False).head(8)
    related = _enrich_products(related_df)

    # Quick CF recommendation (user_cosine, top 6)
    try:
        seen = (DS.get_user_purchased_products(uid) +
                DS.get_user_rated_products(uid))
        train_m = DS.train_matrix()
        recs_raw = get_cf_recommendations(train_m, uid, method="user_pearson", top_n=6)
        rec_pids = [r[0] for r in recs_raw]
        rec_df   = DS.products[DS.products["product_id"].isin(rec_pids)].copy()
        rec_df   = rec_df.set_index("product_id").loc[rec_pids].reset_index()
        recs_enriched = _enrich_products(rec_df)
    except Exception:
        recs_enriched = []

    return render_template("product.html",
        product   = product,
        related   = related,
        recs      = recs_enriched,
        uid       = uid,
        all_users = [(u, _user_display(u)) for u in ALL_USER_IDS],
        user_display = _user_display(uid),
        categories = CATEGORIES,
    )


@app.route("/recommendations")
def recommendations_page():
    uid     = _get_uid_from_cookie(request)
    history = _build_user_history(uid)
    return render_template("recommendations.html",
        uid              = uid,
        all_users        = [(u, _user_display(u)) for u in ALL_USER_IDS],
        user_display     = _user_display(uid),
        categories       = CATEGORIES,
        cats_all         = sorted(DS.products["category"].unique().tolist()),
        brands_all       = sorted(DS.products["brand"].unique().tolist()),
        profile          = DS.get_user_profile(uid),
        purchased_items  = history["purchased"],
        rated_items      = history["rated"],
    )


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data     = request.json or {}
    uid      = int(data.get("user_id", ALL_USER_IDS[0]))
    approach = data.get("approach", "cf")          # cf | cb | kb
    method   = data.get("method",   "user_cosine")
    top_n    = int(data.get("top_n", 10))

    seen = (DS.get_user_purchased_products(uid) +
            DS.get_user_rated_products(uid))

    results      = []
    explain_list = []

    try:
        if approach == "cf":
            train_m  = DS.train_matrix()
            recs_raw = get_cf_recommendations(train_m, uid, method=method, top_n=top_n)
            for pid, score, info in recs_raw:
                expl = build_explanation(pid, score, info, "cf", method,
                                         DS.products, DS.ratings)
                results.append(pid)
                explain_list.append(expl)

        elif approach == "cb":
            seeds = DS.get_user_rated_products(uid, min_rating=4.0)
            if not seeds:
                seeds = DS.get_user_rated_products(uid)[:5]
            recs_raw = get_cb_recommendations(DS.products, seeds, seen,
                                               method=method, top_n=top_n)
            for pid, score, info in recs_raw:
                expl = build_explanation(pid, score, info, "cb", method,
                                         DS.products, DS.ratings)
                results.append(pid)
                explain_list.append(expl)

        elif approach == "kb":
            profile    = DS.get_user_profile(uid)
            pref_cats  = data.get("preferred_categories",
                                  str(profile.get("preferred_categories","")).split(","))
            budget_max = float(data.get("budget_max",
                                        profile.get("budget_max", 9999)))
            min_rating = float(data.get("min_rating", 3.0))

            if method == "constraint":
                recs_raw = get_kb_recommendations(
                    DS.products, method="constraint", seen_ids=seen,
                    top_n=top_n, budget_max=budget_max,
                    preferred_categories=pref_cats, min_rating=min_rating)
            else:  # cbr
                rated_high = DS.get_user_rated_products(uid, min_rating=4.0)
                ref_pid    = rated_high[0] if rated_high else int(DS.products.iloc[0]["product_id"])
                recs_raw   = get_kb_recommendations(
                    DS.products, method="cbr", seen_ids=seen,
                    top_n=top_n, reference_product_id=ref_pid)

            for pid, score, info in recs_raw:
                expl = build_explanation(pid, score, info, "kb", method,
                                         DS.products, DS.ratings)
                results.append(pid)
                explain_list.append(expl)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Build response rows
    if not results:
        return jsonify({"items": []})

    prod_rows = DS.products[DS.products["product_id"].isin(results)].copy()
    prod_rows = prod_rows.set_index("product_id").loc[results].reset_index()
    items = []
    for i, (_, row) in enumerate(prod_rows.iterrows()):
        pid = int(row["product_id"])
        items.append({
            "product_id": pid,
            "name":        row["name"],
            "category":    row["category"],
            "brand":       row["brand"],
            "price":       float(row["price"]),
            "price_fmt":   f"EGP {row['price']:,.0f}",
            "rating":      float(row["rating"]),
            "stars":       _star_html(float(row["rating"])),
            "image_url":   _image_url(pid, row["category"]),
            "explanation": explain_list[i] if i < len(explain_list) else "",
            "url":         f"/product/{pid}",
        })

    return jsonify({"items": items})


@app.route("/analysis")
def analysis():
    uid = _get_uid_from_cookie(request)
    return render_template("analysis.html",
        uid          = uid,
        all_users    = [(u, _user_display(u)) for u in ALL_USER_IDS],
        user_display = _user_display(uid),
        categories   = CATEGORIES,
    )


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Run full evaluation – may take ~30 s on first call."""
    try:
        products  = DS.products
        train_m   = DS.train_matrix()

        cb_recs = {"tfidf": {}, "feature": {}}
        kb_recs = {"constraint": {}, "cbr": {}}

        # Build a lookup of test-set product IDs per user so we can exclude them
        # from `seen` and `seeds` during evaluation.  If test items are in `seen`
        # they get filtered out of CB/KB recommendations, making precision/recall
        # always 0 — the algorithm can never recommend items it's told to skip.
        test_pids_by_user = (
            DS.test_ratings.groupby("user_id")["product_id"]
            .apply(set).to_dict()
        )

        for uid in DS.users["user_id"]:
            test_items  = test_pids_by_user.get(uid, set())

            # Use only TRAIN interactions so test items remain recommendable
            all_rated        = DS.get_user_rated_products(uid)
            train_rated      = [p for p in all_rated      if p not in test_items]
            train_rated_high = [p for p in DS.get_user_rated_products(uid, min_rating=4.0)
                                if p not in test_items]

            seen  = list(set(DS.get_user_purchased_products(uid) + train_rated))
            seeds = train_rated_high if train_rated_high else train_rated[:5]

            for cb_m in ["tfidf", "feature"]:
                cb_recs[cb_m][uid] = get_cb_recommendations(
                    products, seeds, seen, method=cb_m, top_n=20)

            profile    = DS.get_user_profile(uid)
            pref_cats  = str(profile.get("preferred_categories","")).split(",")
            budget_max = float(profile.get("budget_max", 9999))

            kb_recs["constraint"][uid] = get_kb_recommendations(
                products, method="constraint", seen_ids=seen, top_n=20,
                budget_max=budget_max, preferred_categories=pref_cats)

            ref_pid = seeds[0] if seeds else int(products.iloc[0]["product_id"])
            kb_recs["cbr"][uid] = get_kb_recommendations(
                products, method="cbr", seen_ids=seen, top_n=20,
                reference_product_id=ref_pid)

        tfidf_arr = TFIDF_REC.get_tfidf_matrix()
        pid_order = TFIDF_REC.product_ids

        results_df = run_full_evaluation(
            DS, cb_recs, kb_recs,
            tfidf_arr, pid_order, k=5)

        analysis_text = generate_analysis_paragraph(results_df, k=5)

        return jsonify({
            "columns": list(results_df.columns),
            "rows":    results_df.fillna("N/A").values.tolist(),
            "analysis": analysis_text,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug/history")
def debug_history():
    uid = _get_uid_from_cookie(request)
    purch_rows = DS.purchases[DS.purchases["user_id"].astype(int) == uid]
    rated_rows = DS.ratings[DS.ratings["user_id"].astype(int) == uid]
    return jsonify({
        "uid": uid,
        "uid_type": str(type(uid)),
        "purchases_col_dtype": str(DS.purchases["user_id"].dtype),
        "ratings_col_dtype":   str(DS.ratings["user_id"].dtype),
        "n_purchases": int(len(purch_rows)),
        "n_ratings":   int(len(rated_rows)),
        "purchase_pids": purch_rows["product_id"].tolist(),
        "rating_pids":   rated_rows["product_id"].tolist(),
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting AIE425 Recommender Web App on http://localhost:5000")
    app.run(debug=True, port=5000, use_reloader=False)
