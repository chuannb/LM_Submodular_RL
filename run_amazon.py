"""
Full Amazon Pipeline Runner
===========================

Steps:
  1. Load + sample Amazon data (reviews + metadata)
  2. Export products.jsonl
  3. Build BM25 index (rank-bm25, in-memory)
  4. (Optional) Build dense index with Qwen3-Embedding-0.6B
  5. Load Qwen3-Reranker-0.6B
  6. Initialise UnifiedPipeline (Embedding → Reranker → Submodular → RL)
  7. Split data  Train : Val : Test = 8 : 1 : 1
  8. Train (RL actor-critic + submodular diversity params)
  9. Evaluate on test set
 10. Print metrics: hit@k, ndcg@k, mrr@k, coverage, ILD

Default data files:
  review_path : /workspace/All_Amazon_Review_5_10M_filtered.json
  meta_path   : /workspace/All_Amazon_Meta_in_Review.json

Usage (uses defaults):
  cd /workspace/LM_Submodular_RL
  python run_amazon.py

Full options:
  python run_amazon.py \\
      --review_path /workspace/All_Amazon_Review_5_10M_filtered.json \\
      --meta_path   /workspace/All_Amazon_Meta_in_Review.json \\
      --max_users   2000 \\
      --epochs      3 \\
      --steps_per_epoch 200 \\
      --slate_size  10 \\
      --build_dense          # add for Qwen3-Embedding dense retrieval

Quick smoke-test (tiny subset, CPU only):
  python run_amazon.py \\
      --max_users   200 \\
      --epochs      1 \\
      --steps_per_epoch 50 \\
      --eval_steps  50 \\
      --slate_size  10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
RECSYS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(RECSYS_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_query_fn(title_map: Dict[int, str]):
    """Returns a function that builds a BM25/dense query from a TrajectoryStep."""
    def make_query(step) -> str:
        # Use titles of the last 3 history items as a proxy query
        titles = [title_map.get(i, "").strip() for i in step.history_ids[-3:]]
        titles = [t for t in titles if t]
        return " ".join(titles) if titles else "product recommendation"
    return make_query


def export_products_jsonl(products: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in products:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Exported {len(products)} products -> {out_path}")


def compute_ild(slates: List[List[int]], embeddings: torch.Tensor) -> float:
    """Compute mean Intra-List Diversity across all slates."""
    from utils.metrics import diversity_score
    scores = [diversity_score(s, embeddings) for s in slates if len(s) >= 2]
    return float(np.mean(scores)) if scores else 0.0


def save_split_inspection(
    train_steps, val_steps, test_steps,
    title_map: Dict[int, str],
    output_dir: str,
    n_samples: int = 20,
) -> None:
    """
    Save train/val/test split statistics and sample records for inspection.

    Outputs (in output_dir/split_inspection/):
      stats.json          — count, unique users/items, history length stats, budget stats
      train_samples.json  — first n_samples TrajectorySteps (human-readable)
      val_samples.json
      test_samples.json
    """
    import collections

    inspect_dir = os.path.join(output_dir, "split_inspection")
    os.makedirs(inspect_dir, exist_ok=True)

    def _step_stats(steps) -> dict:
        if not steps:
            return {}
        hist_lens = [len(s.history_ids) for s in steps]
        budgets = [s.budget for s in steps]
        unique_users = len({s.user_id for s in steps})
        unique_items = len({s.item_id for s in steps})
        # items per user distribution
        user_counts = collections.Counter(s.user_id for s in steps)
        counts = list(user_counts.values())
        return {
            "n_steps": len(steps),
            "unique_users": unique_users,
            "unique_target_items": unique_items,
            "steps_per_user": {
                "min": int(min(counts)),
                "max": int(max(counts)),
                "mean": round(float(np.mean(counts)), 2),
                "median": round(float(np.median(counts)), 2),
            },
            "history_length": {
                "min": int(min(hist_lens)),
                "max": int(max(hist_lens)),
                "mean": round(float(np.mean(hist_lens)), 2),
            },
            "budget": {
                "min": round(float(min(budgets)), 4),
                "max": round(float(max(budgets)), 4),
                "mean": round(float(np.mean(budgets)), 4),
            },
        }

    def _step_to_dict(s) -> dict:
        history_titles = [title_map.get(i, f"item_{i}") for i in s.history_ids]
        target_title   = title_map.get(s.item_id, f"item_{s.item_id}")
        return {
            "user_id":       s.user_id,
            "target_item_id":   s.item_id,
            "target_title":  target_title,
            "history_ids":   s.history_ids,
            "history_titles": history_titles,
            "history_stars": s.history_extras,
            "budget":        round(s.budget, 4),
            "split":         s.split,
        }

    splits = {"train": train_steps, "val": val_steps, "test": test_steps}

    # ---- Stats ----
    stats = {split: _step_stats(steps) for split, steps in splits.items()}
    stats_path = os.path.join(inspect_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Split stats saved -> {stats_path}")

    # ---- Sample records ----
    for split, steps in splits.items():
        sample_records = [_step_to_dict(s) for s in steps[:n_samples]]
        out_path = os.path.join(inspect_dir, f"{split}_samples.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sample_records, f, indent=2, ensure_ascii=False)
        print(f"  {split:5s} samples ({len(sample_records)}) -> {out_path}")

    # ---- Print summary to stdout ----
    print(f"\n  {'Split':<8} {'Steps':>8} {'Users':>8} {'Items':>8} "
          f"{'Steps/User(mean)':>18} {'HistLen(mean)':>14} {'Budget(mean)':>14}")
    print(f"  {'-'*80}")
    for split, st in stats.items():
        print(f"  {split:<8} {st['n_steps']:>8,} {st['unique_users']:>8,} "
              f"{st['unique_target_items']:>8,} "
              f"{st['steps_per_user']['mean']:>18.2f} "
              f"{st['history_length']['mean']:>14.2f} "
              f"{st['budget']['mean']:>14.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load data (shared across splits to avoid triple disk read)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step 1: Loading Amazon data")
    print(f"{'='*60}")

    from data.amazon_loader import (
        load_amazon_metadata,
        load_amazon_reviews,
        load_amazon_reviews_for_items,
        build_meta_from_reviews,
        AmazonDataset,
    )

    # Reviews-first approach:
    #   1. Stream reviews to collect max_users qualifying users + their ASINs
    #   2. Try to load matching metadata (fast targeted scan of meta file)
    #   3. Fall back to building minimal product catalog from review data
    print(f"  Streaming reviews (max_users={args.max_users}, min_interactions=5)...")
    reviews_df = load_amazon_reviews(
        args.review_path,
        max_users=args.max_users,
        min_interactions=5,
    )
    if reviews_df.empty:
        print("ERROR: No reviews found. Check the review file path.")
        return
    print(f"  Reviews loaded: {len(reviews_df):,} rows, "
          f"{reviews_df['reviewerID'].nunique():,} users, "
          f"{reviews_df['asin'].nunique():,} unique ASINs")

    keep_asins = set(reviews_df["asin"].unique())
    meta_df = None
    if args.meta_path:
        print(f"  Loading metadata for {len(keep_asins):,} review ASINs "
              f"(scanning up to {args.max_items:,} meta records)...")
        meta_df = load_amazon_metadata(
            args.meta_path,
            keep_asins=keep_asins,
            max_records=args.max_items,
        )

    if meta_df is None or meta_df.empty:
        print("  No matching metadata found — building product catalog from review summaries.")
        meta_df = build_meta_from_reviews(reviews_df)
    print(f"  Metadata loaded: {len(meta_df):,} products")

    # Shared-data datasets (no duplicate disk reads)
    print("  Building train / val / test splits (8:1:1)...")
    train_ds = AmazonDataset(
        history_length=args.history_length,
        split="train",
        preloaded_reviews=reviews_df,
        preloaded_meta=meta_df,
    )
    val_ds = AmazonDataset(
        history_length=args.history_length,
        split="val",
        preloaded_reviews=reviews_df,
        preloaded_meta=meta_df,
    )
    test_ds = AmazonDataset(
        history_length=args.history_length,
        split="test",
        preloaded_reviews=reviews_df,
        preloaded_meta=meta_df,
    )

    num_items = train_ds.num_items
    num_users = train_ds.num_users
    print(f"  Users: {num_users:,}  Items: {num_items:,}")
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    # -----------------------------------------------------------------------
    # 2. Build product catalog
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step 2: Building product catalog")
    print(f"{'='*60}")

    products = train_ds.to_products_list()
    products_path = os.path.join(args.output_dir, "products.jsonl")
    export_products_jsonl(products, products_path)

    # id_map: asin (str) -> int index (same as train_ds.item2id)
    id_map: Dict[str, int] = dict(train_ds.item2id)
    id_map_path = os.path.join(args.output_dir, "id_map.json")
    with open(id_map_path, "w") as f:
        json.dump(id_map, f)
    print(f"  id_map saved ({len(id_map)} items) -> {id_map_path}")

    # -----------------------------------------------------------------------
    # 3. Build BM25 index
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step 3: Building BM25 index (rank-bm25)")
    print(f"{'='*60}")

    from retrieval.bm25_retriever import BM25Retriever
    bm25 = BM25Retriever.build(products, backend="rank_bm25")
    bm25_path = os.path.join(args.output_dir, "bm25_index.pkl")
    bm25.save(bm25_path)
    print(f"  BM25 index saved -> {bm25_path}")

    # -----------------------------------------------------------------------
    # 4. (Optional) Build dense index with Qwen3-Embedding-0.6B
    # -----------------------------------------------------------------------
    dense = None
    if args.build_dense:
        print(f"\n{'='*60}")
        print("Step 4: Building dense index (Qwen3-Embedding-0.6B)")
        print(f"{'='*60}")
        from retrieval.embedding_retriever import EmbeddingRetriever
        dense = EmbeddingRetriever(
            model_id="Qwen/Qwen3-Embedding-0.6B",
            device=args.device,
            batch_size=args.embed_batch_size,
        )
        dense.build_index(products)
        dense_path = os.path.join(args.output_dir, "dense_index.pkl")
        dense.save_index(dense_path)
        print(f"  Dense index saved -> {dense_path}")
    else:
        print("\nStep 4: Skipping dense index (use --build_dense to enable)")

    # -----------------------------------------------------------------------
    # 5. Load Qwen3-Reranker-0.6B
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step 5: Loading Qwen3-Reranker-0.6B")
    print(f"{'='*60}")

    from retrieval.reranker import Qwen3Reranker
    reranker = Qwen3Reranker(
        model_id="Qwen/Qwen3-Reranker-0.6B",
        device=args.device,
        batch_size=args.reranker_batch_size,
    )

    # -----------------------------------------------------------------------
    # 6. Initialise models and pipeline
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step 6: Initialising models")
    print(f"{'='*60}")

    from models.submodular import RerankerBackedSubmodular
    from retrieval.unified_pipeline import UnifiedPipeline, UnifiedRLPolicy
    from utils.encoders import StateEncoder

    embed_dim = 128
    submodular = RerankerBackedSubmodular(
        num_items=num_items, embed_dim=64,
        alpha_init=args.alpha_init,
    ).to(device)
    state_encoder = StateEncoder(
        num_items=num_items, embed_dim=embed_dim,
    ).to(device)
    rl_policy = UnifiedRLPolicy(
        state_dim=embed_dim, hidden_dim=256, lr=args.lr_rl, gamma=args.gamma,
    ).to(device)

    costs_map: Dict[int, float] = {
        idx: float(train_ds.price_map.get(idx, 1.0))
        for idx in range(num_items)
    }

    pipeline = UnifiedPipeline(
        bm25=bm25,
        reranker=reranker,
        submodular=submodular,
        rl_policy=rl_policy,
        state_encoder=state_encoder,
        id_map=id_map,
        dense=dense,
        device=device,
        n_bm25=args.n_bm25,
        n_dense=args.n_dense,
        n_fuse=args.n_fuse,
        slate_size=args.slate_size,
        history_length=args.history_length,
        costs_map=costs_map,
    )

    total_params = sum(p.numel() for p in list(submodular.parameters())
                       + list(state_encoder.parameters())
                       + list(rl_policy.parameters()))
    print(f"  Trainable parameters: {total_params:,}")
    print(f"  Slate size k={args.slate_size}  History L={args.history_length}")
    print(f"  BM25 recall={args.n_bm25}  Dense recall={args.n_dense}  Fuse={args.n_fuse}")

    # -----------------------------------------------------------------------
    # 7. Build trajectories + query function
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step 7: Building trajectories")
    print(f"{'='*60}")

    from algorithms.trajectory_builder import build_trajectories

    train_steps = build_trajectories(train_ds, "train")
    val_steps   = build_trajectories(val_ds,   "val")
    test_steps  = build_trajectories(test_ds,  "test")
    print(f"  Train: {len(train_steps):,}  Val: {len(val_steps):,}  Test: {len(test_steps):,}")

    title_map = train_ds.title_map
    make_query = make_query_fn(title_map)

    # -----------------------------------------------------------------------
    # 7b. Save split inspection files
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step 7b: Saving split inspection")
    print(f"{'='*60}")
    save_split_inspection(
        train_steps, val_steps, test_steps,
        title_map=title_map,
        output_dir=args.output_dir,
        n_samples=50,
    )

    # -----------------------------------------------------------------------
    # 7b-2. Log co-purchase graph stats
    # -----------------------------------------------------------------------
    from data.amazon_loader import build_copurchase_graph
    graph = build_copurchase_graph(train_ds)
    gs = graph["stats"]
    print(f"\n  Co-purchase graph (also_buy):")
    print(f"    Items with neighbors : {gs['items_with_copurchase']:,} / {gs['n_items']:,}")
    print(f"    Total edges          : {gs['copurchase_edges']:,}")
    print(f"  Co-view graph (also_view):")
    print(f"    Items with neighbors : {gs['items_with_coview']:,} / {gs['n_items']:,}")
    print(f"    Total edges          : {gs['coview_edges']:,}")

    # -----------------------------------------------------------------------
    # 7c. Build and save DPO pairs (for reranker / DPO fine-tuning)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step 7c: Building DPO preference pairs")
    print(f"{'='*60}")

    dpo_dir = os.path.join(args.output_dir, "dpo_pairs")
    dpo_files = {s: os.path.join(dpo_dir, f"{s}.jsonl") for s in ("train", "val", "test")}
    dpo_exists = all(os.path.exists(p) for p in dpo_files.values())

    if dpo_exists:
        print(f"  DPO pairs already exist — skipping generation.")
        for split_name, path in dpo_files.items():
            n = sum(1 for _ in open(path, encoding="utf-8"))
            print(f"  {split_name:5s}: {n:,} pairs (cached) <- {path}")
    else:
        from data.amazon_loader import build_dpo_pairs
        os.makedirs(dpo_dir, exist_ok=True)
        for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            pairs = build_dpo_pairs(ds, split=split_name, n_negatives=4, seed=args.seed)
            out_path = dpo_files[split_name]
            with open(out_path, "w", encoding="utf-8") as f:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            print(f"  {split_name:5s}: {len(pairs):,} pairs -> {out_path}")

    # -----------------------------------------------------------------------
    # 8. Train (RL + submodular)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 8: Training — {args.epochs} epochs")
    print(f"{'='*60}")

    from algorithms.unified_trainer import UnifiedJointTrainer

    trainer = UnifiedJointTrainer(
        pipeline=pipeline,
        rl_policy=rl_policy,
        submodular=submodular,
        state_encoder=state_encoder,
        lambda_sub=args.lambda_sub,
        lambda_rank=args.lambda_rank,
        lr_sub=args.lr_sub,
        lr_encoder=args.lr_encoder,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer=args.min_buffer,
        gamma=args.gamma,
        history_length=args.history_length,
        device=device,
    )

    best_hit = 0.0
    ckpt_path = os.path.join(args.output_dir, "best_unified.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        losses = trainer.train_epoch(
            trajectory_steps=train_steps,
            pipeline_query_fn=make_query,
            steps_per_epoch=args.steps_per_epoch,
            log_every=args.log_every,
            dataset_type="amazon",
            fast_mode=True,   # BM25 scores as relevance proxy (100x faster than reranker)
        )

        eval_steps = val_steps[:args.eval_steps] if args.eval_steps else val_steps
        val_metrics = trainer.evaluate(
            eval_steps=eval_steps,
            pipeline_query_fn=make_query,
            dataset_type="amazon",
        )

        loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items()) or "(warmup)"
        print(f"  Losses: {loss_str}")
        print(f"  Val  hit@{args.slate_size}={val_metrics['hit@k']:.4f}  "
              f"ndcg@{args.slate_size}={val_metrics['ndcg@k']:.4f}  "
              f"mrr@{args.slate_size}={val_metrics.get('mrr@k', 0.0):.4f}  "
              f"coverage={val_metrics['coverage']:.4f}  "
              f"n={val_metrics['n_samples']}")

        if val_metrics["hit@k"] > best_hit:
            best_hit = val_metrics["hit@k"]
            torch.save({
                "submodular":    submodular.state_dict(),
                "state_encoder": state_encoder.state_dict(),
                "rl_actor":      rl_policy.actor.state_dict(),
                "rl_critic":     rl_policy.critic.state_dict(),
                "epoch":         epoch,
                "best_hit":      best_hit,
            }, ckpt_path)
            print(f"  *** New best hit@{args.slate_size}={best_hit:.4f}  checkpoint saved ***")

    # -----------------------------------------------------------------------
    # 9. Final evaluation on TEST set
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step 9: Final evaluation on TEST set")
    print(f"{'='*60}")

    # Load best checkpoint
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        submodular.load_state_dict(ckpt["submodular"])
        state_encoder.load_state_dict(ckpt["state_encoder"])
        rl_policy.actor.load_state_dict(ckpt["rl_actor"])
        rl_policy.critic.load_state_dict(ckpt["rl_critic"])
        print(f"  Loaded best checkpoint (epoch {ckpt['epoch']}, "
              f"val hit@k={ckpt['best_hit']:.4f})")

    eval_test_steps = test_steps[:args.eval_steps] if args.eval_steps else test_steps
    test_metrics = trainer.evaluate(
        eval_steps=eval_test_steps,
        pipeline_query_fn=make_query,
        dataset_type="amazon",
    )

    # ILD requires embeddings from submodular model
    all_test_slates = trainer.metrics.all_slates
    emb_weight = submodular.item_emb.weight.detach().cpu()
    ild = compute_ild(all_test_slates, emb_weight)

    # -----------------------------------------------------------------------
    # 10. Print results
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS  (k={args.slate_size})")
    print(f"{'='*60}")
    print(f"  Hit@{args.slate_size}      = {test_metrics['hit@k']:.4f}")
    print(f"  NDCG@{args.slate_size}     = {test_metrics['ndcg@k']:.4f}")
    print(f"  MRR@{args.slate_size}      = {test_metrics.get('mrr@k', 0.0):.4f}")
    print(f"  Coverage    = {test_metrics['coverage']:.4f}")
    print(f"  ILD         = {ild:.4f}")
    print(f"  N samples   = {test_metrics['n_samples']}")
    print(f"{'='*60}")
    print(f"  Best val hit@k = {best_hit:.4f}  (checkpoint: {ckpt_path})")

    # Save results to JSON
    results = {
        "test": {**test_metrics, "ild": ild},
        "best_val_hit": best_hit,
        "args": vars(args),
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved -> {results_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Amazon Recommendation Pipeline: Embedding + Reranker + Submodular + RL"
    )

    # Data
    p.add_argument("--review_path",
                   default="/workspace/All_Amazon_Review_5_10M_filtered.json",
                   help="Path to review JSONL file")
    p.add_argument("--meta_path",
                   default="/workspace/All_Amazon_Meta_in_Review.json",
                   help="Path to metadata JSONL file")
    p.add_argument("--max_items",    type=int,   default=50_000,
                   help="Load first N products from meta file (avoids scanning full 12GB)")
    p.add_argument("--max_users",    type=int,   default=2000,
                   help="Sample N users with >= 5 reviews for catalog products")
    p.add_argument("--history_length", type=int, default=10,
                   help="Number of past interactions to encode as state")

    # Retrieval
    p.add_argument("--build_dense",  action="store_true",
                   help="Build Qwen3-Embedding-0.6B dense index for retrieval")
    p.add_argument("--n_bm25",       type=int, default=100,
                   help="BM25 recall size (candidates from BM25)")
    p.add_argument("--n_dense",      type=int, default=50,
                   help="Dense recall size (candidates from dense index)")
    p.add_argument("--n_fuse",       type=int, default=30,
                   help="Candidates after RRF fusion, passed to reranker")
    p.add_argument("--embed_batch_size",    type=int, default=16)
    p.add_argument("--reranker_batch_size", type=int, default=8)

    # Recommendation
    p.add_argument("--slate_size",   type=int,   default=10, help="Final slate size k")

    # Training
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--eval_steps",   type=int,   default=None,
                   help="Max steps for val/test evaluation (None = all)")
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--buffer_size",  type=int,   default=10_000)
    p.add_argument("--min_buffer",   type=int,   default=64)
    p.add_argument("--log_every",    type=int,   default=50)

    # Optimiser
    p.add_argument("--lr_rl",        type=float, default=3e-4,  help="RL policy lr")
    p.add_argument("--lr_sub",       type=float, default=1e-3,  help="Submodular params lr")
    p.add_argument("--lr_encoder",   type=float, default=1e-3,  help="State encoder lr")
    p.add_argument("--gamma",        type=float, default=0.99,  help="RL discount factor")
    p.add_argument("--lambda_sub",   type=float, default=0.5,   help="Submodular loss weight")
    p.add_argument("--lambda_rank",  type=float, default=0.1,   help="Diversity ranking loss weight")
    p.add_argument("--alpha_init",   type=float, default=0.7,   help="Initial relevance weight α")

    # Misc
    p.add_argument("--device",       default="cpu", help="'cpu' or 'cuda'")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output_dir",   default="output_amazon", help="Directory for checkpoints/indexes")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
