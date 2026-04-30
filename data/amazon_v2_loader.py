"""
Loader for pre-split Amazon V2 dataset.

Expected layout:
  dataset_dir/
    train.jsonl   — {"user_id", "history":[{"asin","stars","ts"}...],
                      "target_asin", "target_stars", "r_hit", "user_mean_stars", "ts"}
    val.jsonl
    test.jsonl
  meta_path       — sampled_meta.jsonl (one JSON object per line)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_v2_dataset(
    dataset_dir: str,
    meta_path: str,
    max_train_samples: Optional[int] = None,
) -> dict:
    """
    Load pre-split V2 dataset and build all lookup maps.

    Returns
    -------
    dict with keys:
      train, val, test  : List[dict]  — raw sample records
      item2id           : asin  -> int
      id2item           : int   -> asin
      user2id           : uid   -> int
      products          : List[dict]  — for BM25 index
      price_map         : int   -> float
      title_map         : int   -> str
      copurchase_map    : int   -> List[int]  (also_buy neighbors)
      num_items, num_users
    """
    dataset_dir = Path(dataset_dir)
    meta_path   = Path(meta_path)

    # ── 1. Load meta → build item2id + product catalog ────────────────────
    print("  [v2] Loading meta...", flush=True)
    meta_lookup: Dict[str, dict] = {}

    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("asin", "")
            if asin:
                meta_lookup[asin] = rec

    print(f"  [v2] Meta records: {len(meta_lookup):,}", flush=True)

    # ── 2. Collect all ASINs from splits to ensure full coverage ──────────
    print("  [v2] Scanning splits for ASINs...", flush=True)
    all_asins: set = set(meta_lookup.keys())

    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = rec.get("target_asin", "")
                if asin:
                    all_asins.add(asin)
                for h in rec.get("history", []):
                    ha = h.get("asin", "")
                    if ha:
                        all_asins.add(ha)

    # Deterministic ordering
    item2id: Dict[str, int] = {asin: idx for idx, asin in enumerate(sorted(all_asins))}
    id2item: Dict[int, str] = {v: k for k, v in item2id.items()}
    num_items = len(item2id)
    print(f"  [v2] Unique items (vocab): {num_items:,}", flush=True)

    # ── 3. Build products list (for BM25) ──────────────────────────────────
    products: List[dict] = []
    price_map:  Dict[int, float] = {}
    title_map:  Dict[int, str]   = {}
    copurchase_map: Dict[int, List[int]] = {}

    for asin, idx in item2id.items():
        m = meta_lookup.get(asin, {})

        # Parse price
        price_raw = str(m.get("price", "") or "").replace("$", "").strip()
        price = 0.0
        if price_raw and price_raw.lower() not in ("", "none"):
            try:
                price = float(price_raw.split("-")[0].replace(",", "").strip())
                if price < 0 or price > 100_000:
                    price = 0.0
            except ValueError:
                pass

        # Flatten description
        desc = m.get("description") or []
        if isinstance(desc, list):
            desc = " ".join(str(d) for d in desc)
        desc = str(desc).strip()[:500]

        # Category string
        cats = m.get("category") or []
        cat_str = cats[0] if cats else ""

        # Feature string
        feats = m.get("feature") or []
        feat_str = " ".join(str(x) for x in feats)[:300]

        title = str(m.get("title", "") or asin).strip()

        products.append({
            "item_id":    idx,
            "asin":       asin,
            "title":      title,
            "brand":      str(m.get("brand", "") or "").strip(),
            "description": desc,
            "categories": cat_str,
            "feature":    feat_str,
            "price":      price,
        })

        title_map[idx] = title
        if price > 0:
            price_map[idx] = price

        # Co-purchase neighbors (also_buy)
        ab = m.get("also_buy") or []
        neighbors = [item2id[a] for a in ab if a in item2id]
        if neighbors:
            copurchase_map[idx] = neighbors

    print(f"  [v2] Products built: {len(products):,}", flush=True)

    # ── 4. Load splits ─────────────────────────────────────────────────────
    splits_data: Dict[str, List[dict]] = {}
    user2id: Dict[str, int] = {}

    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"  [v2] WARNING: {path} not found", flush=True)
            splits_data[split] = []
            continue

        data: List[dict] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if max_train_samples and split == "train" and len(data) >= max_train_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = rec.get("user_id", "")
                if uid and uid not in user2id:
                    user2id[uid] = len(user2id)
                data.append(rec)

        splits_data[split] = data
        print(f"  [v2] {split:5}: {len(data):>10,} samples", flush=True)

    return {
        "train":         splits_data["train"],
        "val":           splits_data["val"],
        "test":          splits_data["test"],
        "item2id":       item2id,
        "id2item":       id2item,
        "user2id":       user2id,
        "products":      products,
        "price_map":     price_map,
        "title_map":     title_map,
        "copurchase_map": copurchase_map,
        "num_items":     num_items,
        "num_users":     len(user2id),
    }
