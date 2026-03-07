"""
Amazon Product Review Dataset Loader.

File review fields:
  reviewerID, asin, reviewerName, vote, style, reviewText, overall,
  summary, unixReviewTime, reviewTime, image

File metadata fields:
  asin, title, feature, description, price, imageURL, related,
  salesRank, brand, categories, tech1, tech2
"""

import json
import gzip
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Raw file parsing helpers
# ---------------------------------------------------------------------------

def _parse_jsonl(path: str) -> List[dict]:
    """Read a .json or .json.gz file where each line is a JSON object."""
    records = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_amazon_reviews(review_path: str) -> pd.DataFrame:
    """
    Load the Amazon review file.

    Returns a DataFrame with columns:
      reviewerID, asin, overall, unixReviewTime, reviewText, summary
    """
    records = _parse_jsonl(review_path)
    df = pd.DataFrame(records)
    keep = ["reviewerID", "asin", "overall", "unixReviewTime", "reviewText", "summary"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()
    df["overall"] = pd.to_numeric(df.get("overall", 3.0), errors="coerce").fillna(3.0)
    df["unixReviewTime"] = pd.to_numeric(df.get("unixReviewTime", 0), errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["reviewerID", "asin"])
    df = df.sort_values(["reviewerID", "unixReviewTime"]).reset_index(drop=True)
    return df


def load_amazon_metadata(meta_path: str) -> pd.DataFrame:
    """
    Load the Amazon metadata file.

    Returns a DataFrame with columns:
      asin, title, description, price, brand, categories
    """
    records = _parse_jsonl(meta_path)
    df = pd.DataFrame(records)
    keep = ["asin", "title", "description", "price", "brand", "categories", "feature"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    # Flatten list-type fields to strings
    for col in ["description", "feature", "categories"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: " ".join(x) if isinstance(x, list) else (str(x) if x else "")
            )

    # Normalise price to float
    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    else:
        df["price"] = 0.0

    df = df.dropna(subset=["asin"]).drop_duplicates("asin")
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
    """

    def __init__(
        self,
        review_path: str,
        meta_path: str,
        history_length: int = 10,
        min_interactions: int = 5,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        self.history_length = history_length
        self.split = split

        reviews = load_amazon_reviews(review_path)
        meta = load_amazon_metadata(meta_path)

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

                # Budget = mean price of past purchases
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
