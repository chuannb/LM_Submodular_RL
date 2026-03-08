"""
Offline jobs:
  1. build_index    — build BM25 + dense index + id_map.json từ product DB
  2. export_prefs   — export interaction logs -> preference JSONL
  3. train_ranker   — fine-tune reranker (pairwise loss) từ preference pairs
  4. train_dpo      — DPO/ORPO fine-tuning của reranker
  5. train_unified  — online RL + submodular training trên trajectory data

Chạy:
  python offline.py --action build_index    --products_path products.jsonl --build_dense
  python offline.py --action export_prefs   --db interactions.db --output prefs.jsonl
  python offline.py --action train_ranker   --train prefs_train.jsonl --val prefs_val.jsonl
  python offline.py --action train_dpo      --train dpo_train.jsonl --method orpo
  python offline.py --action train_unified  --dataset retailrocket --events_path events.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List


# ---------------------------------------------------------------------------
# Load products
# ---------------------------------------------------------------------------

def load_products(path: str) -> List[Dict]:
    """Load product list from JSONL file."""
    products = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))
    print(f"Loaded {len(products)} products from {path}")
    return products


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def action_build_index(args: argparse.Namespace) -> None:
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.embedding_retriever import EmbeddingRetriever

    products = load_products(args.products_path)

    # id_map: {item_id_str: int_idx}  — REQUIRED by UnifiedPipeline and serve.py
    id_map = {str(p["item_id"]): i for i, p in enumerate(products)}
    id_map_path = os.path.join(os.path.dirname(args.bm25_index_path), "id_map.json")
    with open(id_map_path, "w") as f:
        json.dump(id_map, f)
    print(f"id_map saved -> {id_map_path}  ({len(id_map)} items)")

    # BM25 index
    print("Building BM25 index...")
    bm25 = BM25Retriever.build(products, backend="rank_bm25")
    bm25.save(args.bm25_index_path)
    print(f"BM25 index saved -> {args.bm25_index_path}")

    # Dense index
    if args.build_dense:
        device = args.device
        print(f"Building dense index on {device}...")
        dense = EmbeddingRetriever(device=device)
        dense.build_index(products)
        dense.save_index(args.dense_index_path)
        print(f"Dense index saved -> {args.dense_index_path}")


def action_export_prefs(args: argparse.Namespace) -> None:
    from interaction.logger import InteractionLogger
    from interaction.preference_converter import PreferenceConverter

    print(f"Loading interaction DB: {args.db}")
    logger = InteractionLogger(args.db)

    # Build item text map from products
    products = load_products(args.products_path)
    item_texts = {
        str(p["item_id"]): " ".join([
            str(p.get("title", "")),
            str(p.get("description", "")),
        ])
        for p in products
    }
    item_titles = {str(p["item_id"]): str(p.get("title", "")) for p in products}

    converter = PreferenceConverter(logger, item_texts, item_titles)

    if args.output_type == "ranker":
        pairs = converter.extract_ranker_pairs(
            max_pairs_per_impression=args.max_pairs
        )
        print(f"Extracted {len(pairs)} ranker pairs")
    else:
        pairs = converter.extract_dpo_samples(
            max_pairs_per_impression=args.max_pairs
        )
        print(f"Extracted {len(pairs)} DPO samples")

    # Train / val split
    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    train, val = pairs[:split], pairs[split:]

    train_path = args.output.replace(".jsonl", "_train.jsonl")
    val_path   = args.output.replace(".jsonl", "_val.jsonl")

    converter.to_jsonl(train, train_path)
    converter.to_jsonl(val, val_path)
    print(f"Train: {len(train)}  Val: {len(val)}")


def action_train_ranker(args: argparse.Namespace) -> None:
    from training.reranker_trainer import RerankerTrainer

    trainer = RerankerTrainer(
        model_id=args.model_id,
        device=args.device,
        use_lora=not args.no_lora,
        lr=args.lr,
    )
    trainer.train(
        train_path=args.train,
        val_path=args.val,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.output_dir,
    )


def action_train_dpo(args: argparse.Namespace) -> None:
    from training.dpo_trainer import DPOFinetuner

    finetuner = DPOFinetuner(
        model_id=args.model_id,
        device=args.device,
        beta=args.beta,
        use_lora=not args.no_lora,
    )

    if args.method == "orpo":
        finetuner.train_orpo(
            train_path=args.train,
            val_path=args.val,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        finetuner.train_dpo(
            train_path=args.train,
            val_path=args.val,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )


def action_train_unified(args: argparse.Namespace) -> None:
    """
    Online RL + submodular training trên trajectory data từ dataset.
    Reranker được load nhưng KHÔNG update ở đây (update offline qua train_ranker/dpo).
    """
    import json as _json
    import torch

    from algorithms.trajectory_builder import build_trajectories
    from algorithms.unified_trainer import UnifiedJointTrainer
    from models.submodular import RerankerBackedSubmodular
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.reranker import Qwen3Reranker
    from retrieval.unified_pipeline import UnifiedPipeline, UnifiedRLPolicy
    from utils.encoders import StateEncoder

    device = torch.device(args.device)

    # Load id_map
    id_map_path = os.path.join(os.path.dirname(args.bm25_index_path), "id_map.json")
    with open(id_map_path) as f:
        id_map: dict = _json.load(f)
    num_items = len(id_map)

    # Load / build retrieval components
    bm25 = BM25Retriever.load(args.bm25_index_path)
    reranker = Qwen3Reranker(
        model_id=args.model_id, device=args.device, batch_size=16
    )

    dense = None
    if args.build_dense and os.path.exists(args.dense_index_path):
        from retrieval.embedding_retriever import EmbeddingRetriever
        dense = EmbeddingRetriever(device=args.device)
        dense.load_index(args.dense_index_path)

    # Trainable online components
    embed_dim = 128
    submodular = RerankerBackedSubmodular(num_items=num_items, embed_dim=64).to(device)
    state_encoder = StateEncoder(num_items=num_items, embed_dim=embed_dim).to(device)
    rl_policy = UnifiedRLPolicy(state_dim=embed_dim, lr=args.lr).to(device)

    # Build unified pipeline
    products = load_products(args.products_path)
    costs_map = {id_map[str(p["item_id"])]: float(p.get("price", 1.0)) for p in products
                 if str(p["item_id"]) in id_map}

    pipeline = UnifiedPipeline(
        bm25=bm25,
        reranker=reranker,
        submodular=submodular,
        rl_policy=rl_policy,
        state_encoder=state_encoder,
        id_map=id_map,
        dense=dense,
        device=device,
        slate_size=args.slate_size,
        history_length=args.history_length,
        costs_map=costs_map,
    )

    # Build trajectory steps
    if args.dataset == "amazon":
        from data.amazon_loader import AmazonDataset
        ds_train = AmazonDataset(
            review_path=args.review_path,
            meta_path=args.meta_path,
            history_length=args.history_length,
            split="train",
        )
        ds_val = AmazonDataset(
            review_path=args.review_path,
            meta_path=args.meta_path,
            history_length=args.history_length,
            split="val",
        )
        dataset_type = "amazon"
        # Simple query from item history: use last item title as proxy query
        def make_query(step):
            # In real usage, this comes from actual search queries logged
            return f"product recommendation"
    else:
        from data.retailrocket_loader import RetailRocketDataset
        ds_train = RetailRocketDataset(
            events_path=args.events_path,
            category_tree_path=args.category_tree,
            item_props_path=args.item_props_path,
            history_length=args.history_length,
            split="train",
        )
        ds_val = RetailRocketDataset(
            events_path=args.events_path,
            category_tree_path=args.category_tree,
            item_props_path=args.item_props_path,
            history_length=args.history_length,
            split="val",
        )
        dataset_type = "retailrocket"
        def make_query(step):
            return "product recommendation"

    from algorithms.trajectory_builder import build_trajectories
    train_steps = build_trajectories(ds_train, "train")
    val_steps   = build_trajectories(ds_val,   "val")
    print(f"Train steps: {len(train_steps)}  Val steps: {len(val_steps)}")

    trainer = UnifiedJointTrainer(
        pipeline=pipeline,
        rl_policy=rl_policy,
        submodular=submodular,
        state_encoder=state_encoder,
        device=device,
        batch_size=args.batch_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_hit = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        losses = trainer.train_epoch(
            train_steps, make_query,
            steps_per_epoch=args.steps_per_epoch,
            dataset_type=dataset_type,
        )
        metrics = trainer.evaluate(val_steps, make_query, dataset_type=dataset_type)

        loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
        print(f"  Losses | {loss_str if loss_str else '(warmup)'}")
        print(f"  Val  hit@k={metrics['hit@k']:.4f}  "
              f"ndcg@k={metrics['ndcg@k']:.4f}  coverage={metrics['coverage']:.4f}")

        if metrics["hit@k"] > best_hit:
            best_hit = metrics["hit@k"]
            ckpt_path = os.path.join(args.output_dir, "best_unified.pt")
            torch.save({
                "submodular": submodular.state_dict(),
                "state_encoder": state_encoder.state_dict(),
                "rl_actor": rl_policy.actor.state_dict(),
                "rl_critic": rl_policy.critic.state_dict(),
                "epoch": epoch,
                "best_hit": best_hit,
            }, ckpt_path)
            print(f"  *** best hit@k={best_hit:.4f} saved -> {ckpt_path} ***")

    print(f"\nDone. Best hit@k = {best_hit:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline jobs for unified product search pipeline")
    p.add_argument("--action", required=True,
                   choices=["build_index", "export_prefs", "train_ranker",
                            "train_dpo", "train_unified"])
    p.add_argument("--device",          default="cpu")
    p.add_argument("--model_id",        default="Qwen/Qwen3-Reranker-0.6B")
    p.add_argument("--dataset",         default="retailrocket",
                   choices=["amazon", "retailrocket"])

    # Index / DB paths
    p.add_argument("--products_path",   default="products.jsonl")
    p.add_argument("--bm25_index_path", default="bm25_index.pkl")
    p.add_argument("--dense_index_path",default="dense_index.pkl")
    p.add_argument("--db",              default="interactions.db")

    # Dataset paths (Amazon)
    p.add_argument("--review_path",     default=None)
    p.add_argument("--meta_path",       default=None)

    # Dataset paths (RetailRocket)
    p.add_argument("--events_path",     default=None)
    p.add_argument("--category_tree",   default=None)
    p.add_argument("--item_props_path", default=None)

    # Output paths
    p.add_argument("--output",          default="preferences.jsonl")
    p.add_argument("--output_dir",      default="checkpoints")
    p.add_argument("--train",           default=None)
    p.add_argument("--val",             default=None)

    # Flags
    p.add_argument("--build_dense",     action="store_true")
    p.add_argument("--no_lora",         action="store_true")
    p.add_argument("--output_type",     default="ranker", choices=["ranker", "dpo"])
    p.add_argument("--method",          default="orpo",   choices=["dpo", "orpo"])

    # Hyper-params (shared)
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=2e-5)
    p.add_argument("--beta",            type=float, default=0.1)
    p.add_argument("--max_pairs",       type=int,   default=5)
    p.add_argument("--history_length",  type=int,   default=10)
    p.add_argument("--slate_size",      type=int,   default=10)
    p.add_argument("--steps_per_epoch", type=int,   default=300)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dispatch = {
        "build_index":   action_build_index,
        "export_prefs":  action_export_prefs,
        "train_ranker":  action_train_ranker,
        "train_dpo":     action_train_dpo,
        "train_unified": action_train_unified,
    }
    dispatch[args.action](args)
