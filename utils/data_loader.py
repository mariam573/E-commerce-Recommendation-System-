import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


class DataStore:

    def __init__(self):
        self.products: pd.DataFrame = None
        self.users: pd.DataFrame = None
        self.ratings: pd.DataFrame = None
        self.purchases: pd.DataFrame = None
        # user-item rating matrix  (users × products)
        self.user_item_matrix: pd.DataFrame = None
        # train / test splits
        self.train_ratings: pd.DataFrame = None
        self.test_ratings: pd.DataFrame = None

    # ─── loaders ─────────────────────────────────────────────────────────────

    def load(self, data_dir: str = DATA_DIR) -> "DataStore":
        self.products  = pd.read_csv(os.path.join(data_dir, "products.csv"))
        self.users     = pd.read_csv(os.path.join(data_dir, "users.csv"))
        self.ratings   = pd.read_csv(os.path.join(data_dir, "ratings.csv"),
                                     parse_dates=["timestamp"])
        self.purchases = pd.read_csv(os.path.join(data_dir, "purchases.csv"),
                                     parse_dates=["date"])
        self._build_matrix()
        self._train_test_split()
        return self

    def _build_matrix(self):
        self.user_item_matrix = self.ratings.pivot_table(
            index="user_id", columns="product_id", values="rating"
        )

    def _train_test_split(self, test_frac: float = 0.20):
        train_parts, test_parts = [], []
        for uid, grp in self.ratings.groupby("user_id"):
            grp_sorted = grp.sort_values("timestamp")
            n_test = max(1, int(len(grp_sorted) * test_frac))
            test_parts.append(grp_sorted.tail(n_test))
            train_parts.append(grp_sorted.iloc[: len(grp_sorted) - n_test])
        self.train_ratings = pd.concat(train_parts).reset_index(drop=True)
        self.test_ratings  = pd.concat(test_parts).reset_index(drop=True)

    # ─── helpers ─────────────────────────────────────────────────────────────

    def get_user_profile(self, user_id: int) -> dict:
        row = self.users[self.users["user_id"] == user_id]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def get_user_rated_products(self, user_id: int, min_rating: float = 1.0) -> list:
        mask = (self.ratings["user_id"] == user_id) & (self.ratings["rating"] >= min_rating)
        return self.ratings.loc[mask, "product_id"].tolist()

    def get_user_purchased_products(self, user_id: int) -> list:
        return self.purchases.loc[
            self.purchases["user_id"] == user_id, "product_id"
        ].tolist()

    def get_product_info(self, product_id: int) -> dict:
        row = self.products[self.products["product_id"] == product_id]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def train_matrix(self) -> pd.DataFrame:
        return self.train_ratings.pivot_table(
            index="user_id", columns="product_id", values="rating"
        )


_store: DataStore = None


def get_datastore(data_dir: str = DATA_DIR) -> DataStore:
    global _store
    if _store is None:
        _store = DataStore().load(data_dir)
    return _store


def reset_datastore():
    global _store
    _store = None
