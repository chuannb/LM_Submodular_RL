"""
Amazon Product Review Dataset Loader.

File review fields:
  reviewerID, asin, reviewerName, vote, style, reviewText, overall,
  summary, unixReviewTime, reviewTime, image

File metadata fields:
  asin, title, feature, description, price, imageURL, related,
  salesRank, brand, categories, tech1, tech2

NOTE: All_Amazon_Meta.json.gz is double-gzipped. Use _open_jsonl() to read it.
"""

import contextlib
import gzip
import json
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Low-level I/O — handles single-gzip AND double-gzip files
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _open_jsonl(path: str):
    """
    Context manager that yields a text file handle for a JSONL path.
    Transparently handles:
      - plain text (.json)
      - single-gzip (.json.gz or .json with gzip magic bytes)
      - double-gzip (e.g. All_Amazon_Meta.json.gz which contains another .gz)
    """
    # Peek at magic bytes to detect gzip regardless of extension
    with open(path, "rb") as probe:
        magic = probe.read(2)
    is_gzip = (magic == b"\x1f\x8b")

    if not is_gzip:
        with open(path, "rt", encoding="utf-8") as f:
            yield f
        return

    # Peek at the first 2 bytes of the decompressed stream
    with gzip.open(path, "rb") as probe:
        inner_magic = probe.read(2)

    if inner_magic == b"\x1f\x8b":  # inner content is also gzip
        outer = gzip.open(path, "rb")
        try:
            with gzip.open(outer, "rt", encoding="utf-8") as inner:
                yield inner
        finally:
            outer.close()
    else:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield f


def _iter_jsonl(path: str, max_records: Optional[int] = None):
    """Iterate over parsed JSON records, optionally stopping early."""
    count = 0
    with _open_jsonl(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                count += 1
                if max_records and count >= max_records:
                    return
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Review loader
# ---------------------------------------------------------------------------

def load_amazon_reviews(
    review_path: str,
    max_users: Optional[int] = None,
    min_interactions: int = 5,
) -> pd.DataFrame:
    """
    Load the Amazon review file with streaming to avoid reading the full 24GB file.

    When max_users is set, reads sequentially and stops once max_users distinct
    users have been accumulated (each with >= min_interactions reviews).

    Parameters
    ----------
    review_path       : path to .json or .json.gz review file
    max_users         : stop after collecting this many users
    min_interactions  : skip users with fewer reviews

    Returns a DataFrame with columns:
      reviewerID, asin, overall, unixReviewTime, reviewText, summary
    """
    keep_fields = {"reviewerID", "asin", "overall", "unixReviewTime", "reviewText", "summary"}

    if max_users is None:
        # Full load
        records = [
            {k: v for k, v in rec.items() if k in keep_fields}
            for rec in _iter_jsonl(review_path)
        ]
    else:
        # Streaming: accept only first max_users users that reach min_interactions.
        # Once we have max_users qualifying users, stop entirely.
        user_records: Dict[str, List[dict]] = {}
        completed_users: set = set()

        for rec in _iter_jsonl(review_path):
            uid = rec.get("reviewerID")
            if uid is None:
                continue

            # If this user is neither completed nor being tracked, only start
            # tracking them if we still need more completed users.
            if uid not in user_records:
                if len(completed_users) >= max_users:
                    continue          # already have enough — ignore new users
                user_records[uid] = []

            user_records[uid].append(
                {k: v for k, v in rec.items() if k in keep_fields}
            )

            if len(user_records[uid]) == min_interactions:
                completed_users.add(uid)
                if len(completed_users) >= max_users:
                    break             # done — stop reading the file

        # Keep only completed users (those with >= min_interactions)
        records = [
            r for uid, recs in user_records.items()
            if uid in completed_users
            for r in recs
        ]

    df = pd.DataFrame(records)
    if df.empty:
        return df

    keep = ["reviewerID", "asin", "overall", "unixReviewTime", "reviewText", "summary"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()
    df["overall"] = pd.to_numeric(df.get("overall", 3.0), errors="coerce").fillna(3.0)
    df["unixReviewTime"] = (
        pd.to_numeric(df.get("unixReviewTime", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    df = df.dropna(subset=["reviewerID", "asin"])
    df = df.sort_values(["reviewerID", "unixReviewTime"]).reset_index(drop=True)
    return df


def load_amazon_reviews_for_items(
    review_path: str,
    keep_asins: Set[str],
    max_users: Optional[int] = None,
    min_interactions: int = 5,
) -> pd.DataFrame:
    """
    Stream reviews, keeping only records for ASINs in keep_asins.
    Stop after collecting max_users qualifying users.

    Use this when you start from a product catalog and want matching reviews.
    Much faster than load_amazon_reviews() when keep_asins is small.
    """
    keep_fields = {"reviewerID", "asin", "overall", "unixReviewTime", "reviewText", "summary"}
    user_records: Dict[str, List[dict]] = {}
    completed_users: set = set()

    for rec in _iter_jsonl(review_path):
        asin = rec.get("asin")
        if asin not in keep_asins:
            continue                 # not a product we care about
        uid = rec.get("reviewerID")
        if uid is None:
            continue

        if uid not in user_records:
            if max_users and len(completed_users) >= max_users:
                continue
            user_records[uid] = []

        user_records[uid].append(
            {k: v for k, v in rec.items() if k in keep_fields}
        )

        if len(user_records[uid]) == min_interactions:
            completed_users.add(uid)
            if max_users and len(completed_users) >= max_users:
                break

    records = [
        r for uid, recs in user_records.items()
        if uid in completed_users
        for r in recs
    ]

    df = pd.DataFrame(records)
    if df.empty:
        return df

    keep = ["reviewerID", "asin", "overall", "unixReviewTime", "reviewText", "summary"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()
    df["overall"] = pd.to_numeric(df.get("overall", 3.0), errors="coerce").fillna(3.0)
    df["unixReviewTime"] = (
        pd.to_numeric(df.get("unixReviewTime", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    df = df.dropna(subset=["reviewerID", "asin"])
    df = df.sort_values(["reviewerID", "unixReviewTime"]).reset_index(drop=True)
    return df


def load_amazon_metadata(
    meta_path: str,
    keep_asins: Optional[Set[str]] = None,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the Amazon metadata file.

    Parameters
    ----------
    meta_path   : path to .json.gz meta file (may be double-gzipped)
    keep_asins  : if provided, only load products whose asin is in this set
    max_records : stop after loading this many records (for fast sampling)

    Returns a DataFrame with columns:
      asin, title, description, price, brand, categories, feature
    """
    records = []
    for rec in _iter_jsonl(meta_path, max_records=max_records):
        if keep_asins is not None and rec.get("asin") not in keep_asins:
            continue
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty or "asin" not in df.columns:
        return pd.DataFrame(columns=["asin", "title", "description", "price",
                                     "brand", "categories", "feature"])
    keep = ["asin", "title", "description", "price", "brand", "categories", "feature",
            "category", "main_cat"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    # Unify "category" / "categories" field
    if "category" in df.columns and "categories" not in df.columns:
        df["categories"] = df["category"]
    if "categories" not in df.columns:
        df["categories"] = ""

    # Flatten list-type fields to strings
    for col in ["description", "feature", "categories", "category"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: " ".join(x) if isinstance(x, list) else (str(x) if x else "")
            )

    # Normalise price to float
    if "price" in df.columns:
        df["price"] = (
            df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    else:
        df["price"] = 0.0

    if "title" not in df.columns:
        df["title"] = ""
    df["title"] = df["title"].fillna("").astype(str)

    df = df.dropna(subset=["asin"]).drop_duplicates("asin")
    return df


# ---------------------------------------------------------------------------
# Fallback: build minimal metadata from review data itself
# ---------------------------------------------------------------------------

def build_meta_from_reviews(review_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a minimal product metadata DataFrame from review data alone.
    Used when a matching metadata file is unavailable.

    For each unique ASIN, aggregates:
      - title       : most-frequent review summary
      - description : concatenation of up to 3 review texts
      - price       : mean star rating (proxy; budget computation still works)
      - brand       : ""
      - categories  : ""
      - feature     : ""
    """
    records = []
    for asin, grp in review_df.groupby("asin"):
        # most-common summary as title
        summaries = grp["summary"].dropna().astype(str)
        title = summaries.mode().iloc[0] if not summaries.empty else asin
        # concat up to 3 review texts as description
        texts = grp["reviewText"].dropna().astype(str).tolist()[:3]
        description = " | ".join(t[:200] for t in texts)
        price = float(grp["overall"].mean()) if "overall" in grp.columns else 3.0
        records.append({
            "asin": asin,
            "title": title,
            "description": description,
            "price": price,
            "brand": "",
            "categories": "",
            "feature": "",
        })
    df = pd.DataFrame(records)
    df = df.drop_duplicates("asin").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_item_index(meta_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map asin <-> integer item id."""
    asins = meta_df["asin"].tolist()
    item2id = {a: i for i, a in enumerate(asins)}
    id2item = {i: a for a, i in item2id.items()}
    return item2id, id2item


def build_user_index(review_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map reviewerID <-> integer user id."""
    users = review_df["reviewerID"].unique().tolist()
    user2id = {u: i for i, u in enumerate(users)}
    id2user = {i: u for u, i in user2id.items()}
    return user2id, id2user


# ---------------------------------------------------------------------------
# High-level dataset class
# ---------------------------------------------------------------------------

class AmazonDataset(Dataset):
    """
    PyTorch Dataset wrapping Amazon interactions.

    Each sample is a dict:
      user_id        : int
      item_id        : int  (next item = label)
      history_ids    : List[int]   (last L items)
      history_stars  : List[float] (stars for history)
      budget         : float       (inferred from history prices)
      price          : float       (price of item_id)
      split          : str         ("train" | "val" | "test")

    Parameters
    ----------
    review_path        : path to review .json.gz
    meta_path          : path to meta .json.gz  (may be double-gzipped)
    history_length     : L — number of past interactions to keep
    min_interactions   : drop users with fewer than this many reviews
    split              : "train" | "val" | "test"
    train_ratio        : fraction of each user's timeline for train
    val_ratio          : fraction of each user's timeline for val
    max_users          : if set, sample this many users (for memory efficiency)
    preloaded_reviews  : pass a pre-loaded DataFrame to skip disk IO
    preloaded_meta     : pass a pre-loaded DataFrame to skip disk IO
    """

    def __init__(
        self,
        review_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        history_length: int = 10,
        min_interactions: int = 5,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        max_users: Optional[int] = None,
        preloaded_reviews: Optional[pd.DataFrame] = None,
        preloaded_meta: Optional[pd.DataFrame] = None,
    ):
        self.history_length = history_length
        self.split = split

        # ---- Load or use preloaded data ----
        if preloaded_reviews is not None:
            reviews = preloaded_reviews.copy()
        else:
            if review_path is None:
                raise ValueError("review_path is required when preloaded_reviews is None")
            reviews = load_amazon_reviews(review_path, max_users=max_users)

        if preloaded_meta is not None:
            meta = preloaded_meta.copy()
        else:
            if meta_path is None:
                raise ValueError("meta_path is required when preloaded_meta is None")
            keep_asins = set(reviews["asin"].unique())
            meta = load_amazon_metadata(meta_path, keep_asins=keep_asins)

        # ---- Build indices ----
        self.item2id, self.id2item = build_item_index(meta)
        self.user2id, self.id2user = build_user_index(reviews)
        self.meta = meta.set_index("asin")

        self.price_map: Dict[int, float] = {
            self.item2id[row.Index]: float(row.price)
            for row in meta.itertuples()
            if row.Index in self.item2id
        }

        reviews["item_id"] = reviews["asin"].map(self.item2id)
        reviews["user_id"] = reviews["reviewerID"].map(self.user2id)
        reviews = reviews.dropna(subset=["item_id", "user_id"])
        reviews["item_id"] = reviews["item_id"].astype(int)
        reviews["user_id"] = reviews["user_id"].astype(int)

        self.samples: List[Dict] = []
        self._build_samples(reviews, min_interactions, train_ratio, val_ratio)

    # ------------------------------------------------------------------
    def _build_samples(
        self,
        reviews: pd.DataFrame,
        min_interactions: int,
        train_ratio: float,
        val_ratio: float,
    ) -> None:
        L = self.history_length
        for user_id, group in reviews.groupby("user_id"):
            group = group.sort_values("unixReviewTime")
            items = group["item_id"].tolist()
            stars = group["overall"].tolist()
            if len(items) < min_interactions:
                continue

            n = len(items)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            for t in range(1, n):
                if t < train_end:
                    s = "train"
                elif t < val_end:
                    s = "val"
                else:
                    s = "test"
                if s != self.split:
                    continue

                history = items[max(0, t - L): t]
                history_stars = stars[max(0, t - L): t]
                next_item = items[t]

                past_prices = [self.price_map.get(i, 0.0) for i in history]
                budget = float(np.mean(past_prices)) if past_prices else 0.0

                self.samples.append({
                    "user_id": user_id,
                    "item_id": next_item,
                    "history_ids": history,
                    "history_stars": history_stars,
                    "budget": max(budget, 1.0),
                    "price": self.price_map.get(next_item, 0.0),
                    "split": s,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    @property
    def num_items(self) -> int:
        return len(self.item2id)

    @property
    def num_users(self) -> int:
        return len(self.user2id)

    @property
    def title_map(self) -> Dict[int, str]:
        """Returns {item_idx: title} for all indexed items."""
        result = {}
        for asin, idx in self.item2id.items():
            if asin in self.meta.index:
                result[idx] = str(self.meta.loc[asin, "title"])
        return result

    def to_products_list(self) -> List[Dict]:
        """
        Export all indexed products as a list of dicts suitable for
        BM25Retriever.build() and EmbeddingRetriever.build_index().
        """
        products = []
        for asin, idx in self.item2id.items():
            if asin not in self.meta.index:
                continue
            row = self.meta.loc[asin]
            products.append({
                "item_id": asin,
                "title": str(row.get("title", "")),
                "description": str(row.get("description", "")),
                "brand": str(row.get("brand", "")),
                "categories": str(row.get("categories", "")),
                "feature": str(row.get("feature", "")),
                "price": float(row.get("price", 0.0)),
            })
        return products
