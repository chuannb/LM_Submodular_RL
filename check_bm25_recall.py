"""
Check BM25 recall rate on the Amazon test split.

For each sampled test record, build a query from the last 3 history titles
(same logic as make_query_fn in run_amazon.py), search BM25, and check
whether the target item appears in the top-k results.

Usage:
    python check_bm25_recall.py
    python check_bm25_recall.py --sample 5000 --topk 10 50 100 200
    python check_bm25_recall.py --split train --sample 20000
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieval.bm25_retriever import BM25Retriever


PRODUCTS_PATH = "/workspace/amazon/output_amazon/products.jsonl"
BM25_INDEX    = "/workspace/amazon/output_amazon/bm25_index.pkl"
DATASET_DIR   = "/workspace/amazon/dataset"


def load_title_maps(products_path: str):
    asin2title = {}
    asin2idx   = {}
    print("Loading product catalog...", flush=True)
    with open(products_path) as f:
        for line in f:
            p = json.loads(line)
            asin2title[p["asin"]] = p["title"]
            asin2idx[p["asin"]]   = p["item_id"]
    print(f"  {len(asin2title):,} products loaded", flush=True)
    return asin2title, asin2idx


def reservoir_sample(path: str, n: int, seed: int = 42) -> list:
    random.seed(seed)
    reservoir = []
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


def run(args):
    # ── Load data ────────────────────────────────────────────────────────────
    asin2title, asin2idx = load_title_maps(PRODUCTS_PATH)

    print(f"Loading BM25 index...", flush=True)
    t0   = time.perf_counter()
    bm25 = BM25Retriever.load(BM25_INDEX)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    split_path = Path(DATASET_DIR) / f"{args.split}.jsonl"
    print(f"\nSampling {args.sample:,} records from {split_path.name}...", flush=True)
    samples = reservoir_sample(str(split_path), args.sample, seed=args.seed)
    print(f"  {len(samples):,} samples ready", flush=True)

    # ── Recall check ─────────────────────────────────────────────────────────
    top_k_list  = sorted(args.topk)
    max_k       = max(top_k_list)
    hits        = {k: 0 for k in top_k_list}
    skip_no_idx = 0
    skip_empty  = 0

    print(f"\nChecking BM25 recall (top_k={max_k})...", flush=True)
    t0 = time.perf_counter()

    for n, rec in enumerate(samples):
        target_asin = rec.get("target_asin", "")
        if target_asin not in asin2idx:
            skip_no_idx += 1
            continue

        target_idx = asin2idx[target_asin]

        # Query: last 3 history item titles  (same as make_query_fn)
        history     = rec.get("history", [])
        hist_titles = [
            asin2title.get(h["asin"], "").strip()
            for h in history[-3:]
            if h.get("asin") in asin2title
        ]
        query = " ".join(t for t in hist_titles if t)
        if not query:
            skip_empty += 1
            continue

        results    = bm25.search(query, top_k=max_k)
        result_ids = [int(r.item_id) for r in results]

        for k in top_k_list:
            if target_idx in result_ids[:k]:
                hits[k] += 1

        if (n + 1) % 1000 == 0:
            done = n + 1 - skip_no_idx - skip_empty
            rate = (n + 1) / (time.perf_counter() - t0)
            eta  = (args.sample - n - 1) / rate
            print(f"  {n+1:>6,}/{args.sample:,}  "
                  f"hit@{top_k_list[-1]}={hits[top_k_list[-1]]/max(done,1):.1%}  "
                  f"{rate:.0f}/s  ETA={eta:.0f}s", flush=True)

    # ── Print results ─────────────────────────────────────────────────────────
    valid = len(samples) - skip_no_idx - skip_empty
    total_time = time.perf_counter() - t0

    print(f"\n{'═'*55}")
    print(f"  Split              : {args.split}")
    print(f"  Samples checked    : {valid:,} / {len(samples):,}")
    print(f"  Skipped (no idx)   : {skip_no_idx:,}")
    print(f"  Skipped (no query) : {skip_empty:,}")
    print(f"{'─'*55}")
    for k in top_k_list:
        pct = hits[k] / valid * 100 if valid else 0
        bar = "█" * int(pct / 2)
        print(f"  BM25 Recall@{k:<4}  = {hits[k]:>6,}/{valid:,} = {pct:5.1f}%  {bar}")
    print(f"{'─'*55}")
    miss = valid - hits[top_k_list[-1]]
    print(f"  NOT found @{top_k_list[-1]:<4}  = {miss:>6,}/{valid:,} = {miss/valid*100:5.1f}%")
    print(f"{'═'*55}")
    print(f"  Total time: {total_time:.1f}s  ({valid/total_time:.0f} queries/s)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",  default="test",  choices=["train", "val", "test"])
    p.add_argument("--sample", type=int, default=10_000)
    p.add_argument("--topk",   type=int, nargs="+", default=[10, 50, 100, 200])
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
