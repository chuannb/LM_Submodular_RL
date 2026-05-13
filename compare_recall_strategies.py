"""
compare_recall_strategies.py
─────────────────────────────
So sánh 4 chiến lược recall cho Amazon next-item prediction.

Mỗi chiến lược chỉ dùng thông tin HISTORY của user (không dùng target):

  S0  BM25 last-3       : query = 3 title cuối trong history  (baseline hiện tại)
  S1  BM25 full-hist    : query = toàn bộ title history (tối đa L items)
  S2  Category-Pop      : top-K items phổ biến nhất trong các category user đã mua
  S3  Also-Buy graph    : co-purchase neighbors của các items trong history
  S4  Hybrid RRF        : RRF fusion của S1 + S2 + S3

Metric: Recall@K — % target item xuất hiện trong top-K candidates

Usage:
    python compare_recall_strategies.py
    python compare_recall_strategies.py --sample 5000 --topk 50 100 200
    python compare_recall_strategies.py --pop_sample 500000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from retrieval.bm25_retriever import BM25Retriever

# ── Paths ─────────────────────────────────────────────────────────────────────
PRODUCTS_PATH = "/workspace/amazon/output_amazon/products.jsonl"
BM25_INDEX    = "/workspace/amazon/output_amazon/bm25_index.pkl"
META_PATH     = "/workspace/amazon/sampled_meta.jsonl"
TRAIN_PATH    = "/workspace/amazon/dataset/train.jsonl"
TEST_PATH     = "/workspace/amazon/dataset/test.jsonl"


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_products(path: str):
    """asin → {item_id, title, top_cat}"""
    asin2id  = {}
    asin2title = {}
    print("  Loading products.jsonl...", flush=True)
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            asin = p["asin"]
            asin2id[asin]    = p["item_id"]
            asin2title[asin] = p.get("title", "")
    print(f"    {len(asin2id):,} products", flush=True)
    return asin2id, asin2title


def load_meta(path: str, known_asins: Set[str]):
    """
    sampled_meta.jsonl → asin → {top_cat: str, also_buy: List[str]}
    Chỉ load các asin có trong catalog (known_asins).
    """
    asin2cat     = {}   # asin → top-level category string
    asin2alsobuy = {}   # asin → list[asin] (also_buy)
    print(f"  Loading meta ({path})...", flush=True)
    with open(path) as f:
        for line in f:
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = m.get("asin", "")
            if asin not in known_asins:
                continue
            cats = m.get("category") or []
            asin2cat[asin] = cats[0] if cats else ""
            ab = [a for a in (m.get("also_buy") or []) if a in known_asins]
            if ab:
                asin2alsobuy[asin] = ab
    print(f"    {len(asin2cat):,} meta records loaded  "
          f"({len(asin2alsobuy):,} with also_buy)", flush=True)
    return asin2cat, asin2alsobuy


def build_popularity(train_path: str, asin2id: Dict[str, int],
                     max_records: Optional[int] = None) -> Dict[int, int]:
    """
    Đếm số lần mỗi item xuất hiện là target_asin trong train.
    → proxy popularity không bị bias bởi test set.
    """
    print(f"  Building popularity from train.jsonl "
          f"({'all' if not max_records else f'{max_records:,}'} records)...",
          flush=True)
    pop: Counter = Counter()
    with open(train_path) as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("target_asin", "")
            if asin in asin2id:
                pop[asin2id[asin]] += 1
    print(f"    {len(pop):,} distinct popular items counted", flush=True)
    return dict(pop)


def build_cat_index(asin2id: Dict[str, int],
                    asin2cat: Dict[str, str],
                    popularity: Dict[int, int]) -> Dict[str, List[int]]:
    """
    {top_cat → [item_id sorted by popularity desc]}
    """
    cat_items: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for asin, idx in asin2id.items():
        cat = asin2cat.get(asin, "")
        if not cat:
            continue
        score = popularity.get(idx, 0)
        cat_items[cat].append((score, idx))

    # Sort each category by popularity descending
    cat2pop = {
        cat: [idx for _, idx in sorted(items, reverse=True)]
        for cat, items in cat_items.items()
    }
    print(f"  Category index: {len(cat2pop):,} categories", flush=True)
    return cat2pop


def reservoir_sample(path: str, n: int, seed: int = 42) -> List[dict]:
    random.seed(seed)
    reservoir: List[dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            if len(reservoir) < n:
                reservoir.append(rec)
            else:
                j = random.randint(0, i)
                if j < n:
                    reservoir[j] = rec
    return reservoir


# ═════════════════════════════════════════════════════════════════════════════
# Recall strategies
# ═════════════════════════════════════════════════════════════════════════════

def s0_bm25_last3(history: list, asin2title: dict,
                  bm25: BM25Retriever, k: int) -> List[int]:
    """BM25 với query = 3 title cuối."""
    titles = [asin2title.get(h["asin"], "").strip()
              for h in history[-3:] if h.get("asin") in asin2title]
    query  = " ".join(t for t in titles if t)
    if not query:
        return []
    return [int(r.item_id) for r in bm25.search(query, top_k=k)]


def s1_bm25_fullhist(history: list, asin2title: dict,
                     bm25: BM25Retriever, k: int) -> List[int]:
    """BM25 với query = tất cả title trong history."""
    titles = [asin2title.get(h["asin"], "").strip()
              for h in history if h.get("asin") in asin2title]
    query  = " ".join(t for t in titles if t)
    if not query:
        return []
    return [int(r.item_id) for r in bm25.search(query, top_k=k)]


def s2_category_pop(history: list, asin2id: dict, asin2cat: dict,
                    cat2pop: Dict[str, List[int]], k: int) -> List[int]:
    """
    Top-K items phổ biến nhất trong các top-level category user đã mua.
    Không bias: popularity tính từ train, không dùng target.
    """
    # Tìm các categories từ history
    user_cats: Dict[str, int] = Counter()
    for h in history:
        cat = asin2cat.get(h.get("asin", ""), "")
        if cat:
            user_cats[cat] += 1

    if not user_cats:
        return []

    # Gom items từ tất cả categories, xếp hạng theo popularity
    # Ưu tiên category xuất hiện nhiều hơn trong history
    seen:   Set[int] = set()
    result: List[int] = []

    # Interleave từ các categories theo tần suất
    sorted_cats = sorted(user_cats, key=user_cats.__getitem__, reverse=True)
    pointers = {cat: 0 for cat in sorted_cats}

    while len(result) < k:
        added_any = False
        for cat in sorted_cats:
            items = cat2pop.get(cat, [])
            ptr   = pointers[cat]
            while ptr < len(items):
                idx = items[ptr]
                ptr += 1
                if idx not in seen:
                    seen.add(idx)
                    result.append(idx)
                    added_any = True
                    break
            pointers[cat] = ptr
        if not added_any:
            break

    return result[:k]


def s3_also_buy(history: list, asin2id: dict,
                asin2alsobuy: Dict[str, List[str]], k: int) -> List[int]:
    """
    Co-purchase neighbors của items trong history.
    Rank: items được nhiều history items recommend cao hơn.
    Không bias: dựa trên product graph Amazon, không dùng target.
    """
    freq: Counter = Counter()
    for h in history:
        asin = h.get("asin", "")
        for neighbor_asin in asin2alsobuy.get(asin, []):
            nb_idx = asin2id.get(neighbor_asin, -1)
            if nb_idx >= 0:
                freq[nb_idx] += 1

    if not freq:
        return []

    return [idx for idx, _ in freq.most_common(k)]


def rrf_fusion(*ranked_lists: List[int], k_rrf: int = 60,
               top_k: int = 200) -> List[int]:
    """
    Reciprocal Rank Fusion — cách combine nhiều ranked lists.
    score(i) = Σ_list  1 / (k_rrf + rank(i, list))
    """
    scores: Dict[int, float] = {}
    for lst in ranked_lists:
        for rank, idx in enumerate(lst, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)

    return [idx for idx, _ in sorted(scores.items(),
                                     key=lambda x: x[1], reverse=True)][:top_k]


def s4_hybrid(history: list,
              asin2title: dict, asin2id: dict, asin2cat: dict,
              cat2pop: dict, asin2alsobuy: dict,
              bm25: BM25Retriever, k: int) -> List[int]:
    """RRF fusion của S1 + S2 + S3."""
    s1 = s1_bm25_fullhist(history, asin2title, bm25, k)
    s2 = s2_category_pop(history, asin2id, asin2cat, cat2pop, k)
    s3 = s3_also_buy(history, asin2id, asin2alsobuy, k)
    return rrf_fusion(s1, s2, s3, top_k=k)


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def check_recall(candidates: List[int], target_idx: int,
                 topk_list: List[int]) -> Dict[int, bool]:
    id_set = {idx: i for i, idx in enumerate(candidates)}
    result = {}
    for k in topk_list:
        result[k] = (target_idx in id_set and id_set[target_idx] < k)
    return result


def run(args):
    print("═" * 60)
    print("  Loading data...")
    print("═" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    asin2id, asin2title = load_products(PRODUCTS_PATH)
    known = set(asin2id.keys())

    asin2cat, asin2alsobuy = load_meta(META_PATH, known)

    popularity = build_popularity(TRAIN_PATH, asin2id,
                                  max_records=args.pop_sample)

    cat2pop = build_cat_index(asin2id, asin2cat, popularity)

    print("  Loading BM25 index...", flush=True)
    t0   = time.perf_counter()
    bm25 = BM25Retriever.load(BM25_INDEX)
    print(f"    Loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"\n  Sampling {args.sample:,} records from test.jsonl...", flush=True)
    samples = reservoir_sample(TEST_PATH, args.sample, seed=args.seed)
    print(f"    {len(samples):,} samples ready", flush=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    top_k_list = sorted(args.topk)
    max_k      = max(top_k_list)

    strategy_names = ["S0 BM25-last3 ", "S1 BM25-full  ",
                      "S2 Cat-Pop    ", "S3 Also-Buy   ", "S4 Hybrid-RRF "]
    hits   = {name: {k: 0 for k in top_k_list} for name in strategy_names}
    totals = {name: 0 for name in strategy_names}

    # Track unique items recommended (diversity check)
    unique_recs = {name: set() for name in strategy_names}

    print(f"\n═" * 60)
    print(f"  Evaluating {len(samples):,} samples  (top_k={max_k})...")
    print(f"═" * 60, flush=True)

    t0 = time.perf_counter()

    for n, rec in enumerate(samples):
        target_asin = rec.get("target_asin", "")
        if target_asin not in asin2id:
            continue

        target_idx = asin2id[target_asin]
        history    = rec.get("history", [])

        cands = {
            "S0 BM25-last3 ": s0_bm25_last3(
                history, asin2title, bm25, max_k),
            "S1 BM25-full  ": s1_bm25_fullhist(
                history, asin2title, bm25, max_k),
            "S2 Cat-Pop    ": s2_category_pop(
                history, asin2id, asin2cat, cat2pop, max_k),
            "S3 Also-Buy   ": s3_also_buy(
                history, asin2id, asin2alsobuy, max_k),
            "S4 Hybrid-RRF ": s4_hybrid(
                history, asin2title, asin2id, asin2cat,
                cat2pop, asin2alsobuy, bm25, max_k),
        }

        for name, cand_list in cands.items():
            if not cand_list:
                continue
            totals[name] += 1
            unique_recs[name].update(cand_list)
            for k, hit in check_recall(cand_list, target_idx, top_k_list).items():
                if hit:
                    hits[name][k] += 1

        if (n + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate    = (n + 1) / elapsed
            eta     = (args.sample - n - 1) / rate
            top_k_show = top_k_list[-1]
            row = "  ".join(
                f"{name.strip()}:{hits[name][top_k_show]/max(totals[name],1):.0%}"
                for name in strategy_names
            )
            print(f"  [{n+1:>5}/{args.sample}]  {row}  ETA={eta:.0f}s", flush=True)

    # ── Print results ─────────────────────────────────────────────────────────
    valid = totals["S0 BM25-last3 "]   # baseline denominator
    elapsed = time.perf_counter() - t0

    print(f"\n{'═'*68}")
    print(f"  {'Strategy':<18}", end="")
    for k in top_k_list:
        print(f"  Recall@{k:<4}", end="")
    print(f"  Unique/{max_k}")
    print(f"{'─'*68}")

    for name in strategy_names:
        n = totals[name]
        print(f"  {name}", end="")
        for k in top_k_list:
            pct = hits[name][k] / n * 100 if n else 0
            print(f"  {pct:7.1f}%  ", end="")
        uniq = len(unique_recs[name])
        print(f"  {uniq:>9,}")

    print(f"{'─'*68}")
    print(f"  Samples evaluated: {valid:,}  |  Total time: {elapsed:.1f}s  "
          f"({valid/elapsed:.0f} samples/s)")
    print(f"{'═'*68}")

    # ── Bias notes ────────────────────────────────────────────────────────────
    print("""
  Ghi chú bias của mỗi chiến lược:
  ┌──────────────────┬────────────────────────────────────────────────┐
  │ S0 BM25-last3    │ Bias: chỉ capture short-term intent (3 items) │
  │ S1 BM25-full     │ Bias: long query → term dilution (BM25 yếu)  │
  │ S2 Cat-Pop       │ Bias: popularity bias (nặng → mọi user giống) │
  │ S3 Also-Buy      │ Bias: Amazon product graph → cold-start tốt   │
  │ S4 Hybrid RRF    │ Bias: kết hợp → giảm bias đơn lẻ             │
  └──────────────────┴────────────────────────────────────────────────┘
  "Unique/{k}" = số items khác nhau được recommend qua {k} test samples
  → Thấp = popularity bias nặng (cùng vài items cho mọi user)
  → Cao  = diverse, ít bias hơn
""".format(k=args.sample))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample",     type=int, default=2_000,
                   help="Số test samples (default: 2000)")
    p.add_argument("--topk",       type=int, nargs="+", default=[50, 100, 200],
                   help="Recall@K values to measure")
    p.add_argument("--pop_sample", type=int, default=None,
                   help="Giới hạn records đọc từ train để tính popularity "
                        "(None = đọc tất cả 11.7M records, ~60s)")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
