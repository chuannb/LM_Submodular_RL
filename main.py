"""
Main entry point.

Usage:
  # Amazon
  python main.py --dataset amazon \
      --review_path data/Electronics_5.json.gz \
      --meta_path   data/meta_Electronics.json.gz

  # RetailRocket
  python main.py --dataset retailrocket \
      --events_path       data/events.csv \
      --category_tree     data/category_tree.csv \
      --item_props_path   data/item_properties_part1.csv \
      --item_props_path2  data/item_properties_part2.csv
"""

import argparse
import random
import sys

import numpy as np
import torch

from config import Config, DataConfig, ModelConfig
from algorithms.joint_trainer import JointTrainer
from algorithms.trajectory_builder import build_trajectories
from data.amazon_loader import AmazonDataset
from data.retailrocket_loader import RetailRocketDataset
from models.generator import GeneratorModel
from models.rl_policy import RLPolicy, Z_DIM, ALPHA_DIM, KAPPA_DIM
from models.submodular import SubmodularUtility
from utils.encoders import StateEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(
    num_items: int,
    cfg: Config,
    device: torch.device,
) -> tuple:
    mc = cfg.model
    encoder = StateEncoder(
        num_items=num_items,
        embed_dim=mc.embed_dim,
        hidden_dim=mc.hidden_dim,
        extra_dim=1,
        dropout=mc.generator_dropout,
    )
    generator = GeneratorModel(
        num_items=num_items,
        embed_dim=mc.embed_dim,
        hidden_dim=mc.hidden_dim,
        latent_dim=Z_DIM,
        num_layers=mc.generator_layers,
        dropout=mc.generator_dropout,
    )
    submodular = SubmodularUtility(
        num_items=num_items,
        embed_dim=mc.embed_dim,
        hidden_dim=mc.hidden_dim,
        alpha_init=mc.relevance_weight,
        kernel=mc.submodular_kernel,
    )
    policy = RLPolicy(
        state_dim=mc.embed_dim,
        hidden_dim=mc.rl_hidden_dim,
        num_layers=mc.rl_layers,
        gamma=mc.gamma,
        lr=mc.lr_policy,
    )
    return encoder, generator, submodular, policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    cfg = Config(
        data=DataConfig(
            dataset=args.dataset,
            history_length=args.history_length,
            amazon_review_path=args.review_path,
            amazon_meta_path=args.meta_path,
            rr_events_path=args.events_path,
            rr_category_tree_path=args.category_tree,
            rr_item_props_path=args.item_props_path,
            rr_item_props_path2=getattr(args, "item_props_path2", None),
        ),
        model=ModelConfig(
            slate_size=args.slate_size,
            candidate_size=args.candidate_size,
            num_epochs=args.epochs,
        ),
        seed=args.seed,
        device=args.device,
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ---- Load dataset ----
    print(f"Loading {cfg.data.dataset} dataset...")
    if cfg.data.dataset == "amazon":
        if not cfg.data.amazon_review_path or not cfg.data.amazon_meta_path:
            print("ERROR: --review_path and --meta_path are required for Amazon dataset.")
            sys.exit(1)
        train_ds = AmazonDataset(
            review_path=cfg.data.amazon_review_path,
            meta_path=cfg.data.amazon_meta_path,
            history_length=cfg.data.history_length,
            split="train",
        )
        val_ds = AmazonDataset(
            review_path=cfg.data.amazon_review_path,
            meta_path=cfg.data.amazon_meta_path,
            history_length=cfg.data.history_length,
            split="val",
        )
        num_items = train_ds.num_items
        costs_map = dict(train_ds.price_map)
        dataset_type = "amazon"
        print(f"  Users: {train_ds.num_users}  Items: {num_items}")
        print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    else:  # retailrocket
        if not cfg.data.rr_events_path:
            print("ERROR: --events_path is required for RetailRocket dataset.")
            sys.exit(1)
        train_ds = RetailRocketDataset(
            events_path=cfg.data.rr_events_path,
            category_tree_path=cfg.data.rr_category_tree_path,
            item_props_path=cfg.data.rr_item_props_path,
            item_props_path2=cfg.data.rr_item_props_path2,
            history_length=cfg.data.history_length,
            slate_size=cfg.model.slate_size,
            split="train",
        )
        val_ds = RetailRocketDataset(
            events_path=cfg.data.rr_events_path,
            category_tree_path=cfg.data.rr_category_tree_path,
            item_props_path=cfg.data.rr_item_props_path,
            item_props_path2=cfg.data.rr_item_props_path2,
            history_length=cfg.data.history_length,
            slate_size=cfg.model.slate_size,
            split="val",
        )
        num_items = train_ds.num_items
        costs_map = None   # uniform cost
        dataset_type = "retailrocket"
        print(f"  Users: {train_ds.num_users}  Items: {num_items}")
        print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ---- Build trajectory steps ----
    print("Building trajectories...")
    train_steps = build_trajectories(train_ds, split="train")
    val_steps = build_trajectories(val_ds, split="val")
    print(f"  Train steps: {len(train_steps)}  Val steps: {len(val_steps)}")

    # ---- Build models ----
    print("Initialising models...")
    encoder, generator, submodular, policy = build_models(num_items, cfg, device)

    total_params = sum(
        p.numel() for m in [encoder, generator, submodular, policy]
        for p in m.parameters()
    )
    print(f"  Total parameters: {total_params:,}")

    # ---- Create trainer ----
    trainer = JointTrainer(
        encoder=encoder,
        generator=generator,
        submodular=submodular,
        policy=policy,
        slate_size=cfg.model.slate_size,
        history_length=cfg.data.history_length,
        candidate_size=cfg.model.candidate_size,
        gamma=cfg.model.gamma,
        batch_size=cfg.model.batch_size,
        buffer_size=cfg.model.buffer_size,
        min_buffer_size=cfg.model.min_buffer_size,
        lr_gen=cfg.model.lr_generator,
        lr_sub=cfg.model.lr_submodular,
        device=device,
        dataset_type=dataset_type,
        costs_map=costs_map,
    )

    # ---- Training loop ----
    print(f"\nStarting training for {cfg.model.num_epochs} epochs...")
    best_hit = 0.0

    for epoch in range(1, cfg.model.num_epochs + 1):
        print(f"\n[Epoch {epoch}/{cfg.model.num_epochs}]")
        train_losses = trainer.train_epoch(
            train_steps=train_steps,
            steps_per_epoch=cfg.model.steps_per_epoch,
            log_every=cfg.log_every,
        )
        val_metrics = trainer.evaluate(val_steps)

        loss_str = "  ".join(f"{k}={v:.4f}" for k, v in train_losses.items())
        print(f"  Train losses | {loss_str if loss_str else '(warmup)'}")
        print(f"  Val   hit@{cfg.model.slate_size}={val_metrics['hit@k']:.4f}  "
              f"ndcg@{cfg.model.slate_size}={val_metrics['ndcg@k']:.4f}  "
              f"coverage={val_metrics['coverage']:.4f}")

        if val_metrics["hit@k"] > best_hit:
            best_hit = val_metrics["hit@k"]
            torch.save({
                "encoder": encoder.state_dict(),
                "generator": generator.state_dict(),
                "submodular": submodular.state_dict(),
                "policy_actor": policy.actor.state_dict(),
                "policy_critic": policy.critic.state_dict(),
                "epoch": epoch,
                "best_hit": best_hit,
            }, "best_model.pt")
            print(f"  *** New best hit@k={best_hit:.4f} — model saved ***")

    print(f"\nTraining complete. Best hit@{cfg.model.slate_size} = {best_hit:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recommendation System: Generative + Submodular + RL"
    )
    p.add_argument("--dataset", choices=["amazon", "retailrocket"], default="retailrocket")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--history_length", type=int, default=10)
    p.add_argument("--slate_size", type=int, default=10)
    p.add_argument("--candidate_size", type=int, default=50)

    # Amazon
    p.add_argument("--review_path", default=None)
    p.add_argument("--meta_path", default=None)

    # RetailRocket
    p.add_argument("--events_path", default=None)
    p.add_argument("--category_tree", default=None)
    p.add_argument("--item_props_path", default=None)
    p.add_argument("--item_props_path2", default=None)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
