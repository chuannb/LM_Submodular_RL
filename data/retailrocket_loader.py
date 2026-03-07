"""
RetailRocket E-commerce Dataset Loader.

Files:
  events.csv           : timestamp, visitorid, event, itemid, transactionid
  category_tree.csv    : categoryid, parentid
  item_properties_part1.csv / part2.csv : timestamp, itemid, property, value
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Event reward weights (view < addtocart < transaction)
# ---------------------------------------------------------------------------
EVENT_REWARD: Dict[str, float] = {
    "view": 0.1,
    "addtocart": 0.5,
    "transaction": 1.0,
}


# ---------------------------------------------------------------------------
# Raw file loaders
# ---------------------------------------------------------------------------

def load_events(path: str) -> pd.DataFrame:
    """Load events.csv and return cleaned DataFrame."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df = df.dropna(subset=["timestamp", "user_id", "item_id", "event"])
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["item_id"] = df["item_id"].astype(int)
    df["user_id"] = df["user_id"].astype(int)
    df["event"] = df["event"].str.strip().str.lower()
    df["reward"] = df["event"].map(EVENT_REWARD).fillna(0.0)
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def load_category_tree(path: str) -> pd.DataFrame:
    """Load category_tree.csv and return DataFrame with categoryid, parentid."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_item_properties(part1_path: str, part2_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and merge item_properties files.
    Returns DataFrame: timestamp, itemid, property, value
    """
    dfs = [pd.read_csv(part1_path)]
    if part2_path and part2_path != part1_path:
        dfs.append(pd.read_csv(part2_path))
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"itemid": "item_id"})
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").dropna()
    # Keep only latest snapshot per (item_id, property)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["item_id", "property"], keep="last")
    return df


def build_item_category_map(item_props: pd.DataFrame) -> Dict[int, int]:
    """Map item_id -> categoryid (from item properties)."""
    cat_rows = item_props[item_props["property"] == "categoryid"].copy()
    cat_rows["item_id"] = cat_rows["item_id"].astype(int)
    cat_rows["value"] = pd.to_numeric(cat_rows["value"], errors="coerce")
    return dict(zip(cat_rows["item_id"], cat_rows["value"].fillna(-1).astype(int)))


# ---------------------------------------------------------------------------
# Session builder
# ---------------------------------------------------------------------------

def build_sessions(
    events: pd.DataFrame,
    session_gap_ms: int = 1_800_000,
) -> pd.DataFrame:
    """
    Split user event sequences into sessions based on timestamp gap.

    Returns events DataFrame with an added 'session_id' column.
    The session_gap_ms default = 30 minutes in milliseconds.
    """
    events = events.sort_values(["user_id", "timestamp"]).copy()
    session_ids = []
    current_session = 0
    prev_uid = None
    prev_ts = None

    for _, row in events.iterrows():
        uid = row["user_id"]
        ts = row["timestamp"]
        if prev_uid is None or uid != prev_uid or (ts - prev_ts) > session_gap_ms:
            current_session += 1
        session_ids.append(current_session)
        prev_uid = uid
        prev_ts = ts

    events["session_id"] = session_ids
    return events


# ---------------------------------------------------------------------------
# High-level dataset class
# ---------------------------------------------------------------------------

class RetailRocketDataset(Dataset):
    """
    PyTorch Dataset wrapping RetailRocket interactions.

    Each sample is a dict:
      user_id        : int
      item_id        : int   (next item = label)
      event          : str   (next event type)
      reward         : float (reward of next event)
      history_ids    : List[int]
      history_events : List[str]
      budget         : int   (= slate_size k, uniform cost)
      session_id     : int
      split          : str
    """

    def __init__(
        self,
        events_path: str,
        category_tree_path: Optional[str] = None,
        item_props_path: Optional[str] = None,
        item_props_path2: Optional[str] = None,
        history_length: int = 10,
        session_gap_seconds: int = 1800,
        min_session_length: int = 3,
        slate_size: int = 10,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        self.history_length = history_length
        self.slate_size = slate_size
        self.split = split

        events = load_events(events_path)
        events = build_sessions(events, session_gap_ms=session_gap_seconds * 1000)

        # Build item / user index from events
        all_items = events["item_id"].unique().tolist()
        all_users = events["user_id"].unique().tolist()
        self.item2id = {it: i for i, it in enumerate(sorted(all_items))}
        self.user2id = {u: i for i, u in enumerate(sorted(all_users))}
        self.id2item = {v: k for k, v in self.item2id.items()}
        self.id2user = {v: k for k, v in self.user2id.items()}

        # Optional: category map
        self.item_category: Dict[int, int] = {}
        if item_props_path:
            ip = load_item_properties(item_props_path, item_props_path2)
            self.item_category = build_item_category_map(ip)

        events["item_idx"] = events["item_id"].map(self.item2id)
        events["user_idx"] = events["user_id"].map(self.user2id)

        self.samples: List[Dict] = []
        self._build_samples(events, min_session_length, train_ratio, val_ratio)

    # ------------------------------------------------------------------
    def _build_samples(
        self,
        events: pd.DataFrame,
        min_session_length: int,
        train_ratio: float,
        val_ratio: float,
    ) -> None:
        L = self.history_length
        for session_id, group in events.groupby("session_id"):
            group = group.sort_values("timestamp")
            items = group["item_idx"].tolist()
            evts = group["event"].tolist()
            rewards = group["reward"].tolist()

            if len(items) < min_session_length:
                continue

            n = len(items)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            user_id = group["user_idx"].iloc[0]

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
                history_evts = evts[max(0, t - L): t]

                self.samples.append({
                    "user_id": user_id,
                    "item_id": items[t],
                    "event": evts[t],
                    "reward": rewards[t],
                    "history_ids": history,
                    "history_events": history_evts,
                    "budget": self.slate_size,   # uniform cost c(i)=1
                    "session_id": session_id,
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
