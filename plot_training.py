"""
Plot training curves from logged loss values.
Saves charts to /workspace/LM_Submodular_RL/output_amazon_full/plots/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Raw data from training logs
# ---------------------------------------------------------------------------

# Format: (global_step, metric_dict)
# Epoch 1: steps 300-900 (log_every=100, steps_per_epoch=1000)
# Epoch 2: steps 1000-1900
# Epoch 3: steps 2000-2900

LOG_DATA = [
    # --- Epoch 1 ---
    (300,  {"rl/critic_loss": 0.0157, "rl/actor_loss": 0.0570, "rl/pg_loss":  0.0012, "sub/reinforce_loss":  0.0420, "sub/div_rank_loss": 0.4987, "sub/total": 0.0919, "sub/alpha_mean": 0.5057}),
    (400,  {"rl/critic_loss": 0.0294, "rl/actor_loss": 0.0675, "rl/pg_loss":  0.0168, "sub/reinforce_loss":  0.0698, "sub/div_rank_loss": 0.4902, "sub/total": 0.1188, "sub/alpha_mean": 0.5023}),
    (500,  {"rl/critic_loss": 0.0145, "rl/actor_loss": 0.0475, "rl/pg_loss": -0.0027, "sub/reinforce_loss":  0.0349, "sub/div_rank_loss": 0.4343, "sub/total": 0.0783, "sub/alpha_mean": 0.4935}),
    (600,  {"rl/critic_loss": 0.0089, "rl/actor_loss": 0.0436, "rl/pg_loss": -0.0067, "sub/reinforce_loss":  0.0280, "sub/div_rank_loss": 0.3046, "sub/total": 0.0585, "sub/alpha_mean": 0.5063}),
    (700,  {"rl/critic_loss": 0.0268, "rl/actor_loss": 0.0675, "rl/pg_loss":  0.0219, "sub/reinforce_loss":  0.0698, "sub/div_rank_loss": 0.2132, "sub/total": 0.0911, "sub/alpha_mean": 0.4929}),
    (800,  {"rl/critic_loss": 0.0151, "rl/actor_loss": 0.0210, "rl/pg_loss": -0.0226, "sub/reinforce_loss":  0.0354, "sub/div_rank_loss": 0.1716, "sub/total": 0.0526, "sub/alpha_mean": 0.5085}),
    (900,  {"rl/critic_loss": 0.0007, "rl/actor_loss": 0.0200, "rl/pg_loss": -0.0201, "sub/reinforce_loss":  0.0000, "sub/div_rank_loss": 0.1328, "sub/total": 0.0133, "sub/alpha_mean": 0.4784}),
    # --- Epoch 2 ---
    (1000, {"rl/critic_loss": 0.0252, "rl/actor_loss": 0.0406, "rl/pg_loss":  0.0025, "sub/reinforce_loss":  0.0637, "sub/div_rank_loss": 0.1068, "sub/total": 0.0744, "sub/alpha_mean": 0.4932}),
    (1100, {"rl/critic_loss": 0.0080, "rl/actor_loss": 0.0431, "rl/pg_loss":  0.0126, "sub/reinforce_loss":  0.0280, "sub/div_rank_loss": 0.0627, "sub/total": 0.0342, "sub/alpha_mean": 0.4735}),
    (1200, {"rl/critic_loss": 0.0023, "rl/actor_loss":-0.0185, "rl/pg_loss": -0.0533, "sub/reinforce_loss":  0.0000, "sub/div_rank_loss": 0.0472, "sub/total": 0.0047, "sub/alpha_mean": 0.4858}),
    (1300, {"rl/critic_loss": 0.0426, "rl/actor_loss": 0.1089, "rl/pg_loss":  0.0731, "sub/reinforce_loss":  0.1054, "sub/div_rank_loss": 0.0701, "sub/total": 0.1124, "sub/alpha_mean": 0.5008}),
    (1400, {"rl/critic_loss": 0.0150, "rl/actor_loss": 0.0120, "rl/pg_loss": -0.0146, "sub/reinforce_loss":  0.0353, "sub/div_rank_loss": 0.1097, "sub/total": 0.0463, "sub/alpha_mean": 0.4999}),
    (1500, {"rl/critic_loss": 0.0280, "rl/actor_loss": 0.0236, "rl/pg_loss": -0.0118, "sub/reinforce_loss":  0.0695, "sub/div_rank_loss": 0.1018, "sub/total": 0.0797, "sub/alpha_mean": 0.4901}),
    (1600, {"rl/critic_loss": 0.0007, "rl/actor_loss": 0.0178, "rl/pg_loss": -0.0151, "sub/reinforce_loss":  0.0000, "sub/div_rank_loss": 0.0462, "sub/total": 0.0046, "sub/alpha_mean": 0.4956}),
    (1700, {"rl/critic_loss": 0.0095, "rl/actor_loss": 0.0359, "rl/pg_loss":  0.0026, "sub/reinforce_loss":  0.0283, "sub/div_rank_loss": 0.0851, "sub/total": 0.0368, "sub/alpha_mean": 0.5073}),
    (1800, {"rl/critic_loss": 0.0145, "rl/actor_loss": 0.0293, "rl/pg_loss":  0.0021, "sub/reinforce_loss":  0.0421, "sub/div_rank_loss": 0.1061, "sub/total": 0.0527, "sub/alpha_mean": 0.4975}),
    (1900, {"rl/critic_loss": 0.0013, "rl/actor_loss": 0.0216, "rl/pg_loss": -0.0115, "sub/reinforce_loss":  0.0000, "sub/div_rank_loss": 0.1217, "sub/total": 0.0122, "sub/alpha_mean": 0.5172}),
    # --- Epoch 3 ---
    (2000, {"rl/critic_loss": 0.0007, "rl/actor_loss": 0.0207, "rl/pg_loss": -0.0081, "sub/reinforce_loss":  0.0000, "sub/div_rank_loss": 0.0872, "sub/total": 0.0087, "sub/alpha_mean": 0.5187}),
    (2100, {"rl/critic_loss": 0.0287, "rl/actor_loss": 0.0309, "rl/pg_loss":  0.0050, "sub/reinforce_loss":  0.0707, "sub/div_rank_loss": 0.0729, "sub/total": 0.0780, "sub/alpha_mean": 0.5213}),
    (2200, {"rl/critic_loss": 0.0222, "rl/actor_loss": 0.0341, "rl/pg_loss":  0.0012, "sub/reinforce_loss":  0.0634, "sub/div_rank_loss": 0.0747, "sub/total": 0.0708, "sub/alpha_mean": 0.5052}),
    (2300, {"rl/critic_loss": 0.0018, "rl/actor_loss": 0.0016, "rl/pg_loss": -0.0217, "sub/reinforce_loss":  0.0000, "sub/div_rank_loss": 0.1134, "sub/total": 0.0113, "sub/alpha_mean": 0.5187}),
    (2400, {"rl/critic_loss": 0.0206, "rl/actor_loss": 0.0498, "rl/pg_loss":  0.0247, "sub/reinforce_loss":  0.0701, "sub/div_rank_loss": 0.0933, "sub/total": 0.0794, "sub/alpha_mean": 0.5077}),
    (2500, {"rl/critic_loss": 0.0434, "rl/actor_loss": 0.0460, "rl/pg_loss":  0.0190, "sub/reinforce_loss":  0.1049, "sub/div_rank_loss": 0.0930, "sub/total": 0.1142, "sub/alpha_mean": 0.5119}),
    (2600, {"rl/critic_loss": 0.0276, "rl/actor_loss": 0.0245, "rl/pg_loss": -0.0028, "sub/reinforce_loss":  0.0704, "sub/div_rank_loss": 0.0593, "sub/total": 0.0763, "sub/alpha_mean": 0.5116}),
    (2700, {"rl/critic_loss": 0.0115, "rl/actor_loss": 0.0121, "rl/pg_loss": -0.0129, "sub/reinforce_loss":  0.0349, "sub/div_rank_loss": 0.0949, "sub/total": 0.0444, "sub/alpha_mean": 0.4960}),
    (2800, {"rl/critic_loss": 0.0090, "rl/actor_loss": 0.0213, "rl/pg_loss":  0.0004, "sub/reinforce_loss":  0.0280, "sub/div_rank_loss": 0.1138, "sub/total": 0.0394, "sub/alpha_mean": 0.5209}),
    (2900, {"rl/critic_loss": 0.0092, "rl/actor_loss": 0.0015, "rl/pg_loss": -0.0205, "sub/reinforce_loss":  0.0355, "sub/div_rank_loss": 0.0618, "sub/total": 0.0416, "sub/alpha_mean": 0.5044}),
]

# Epoch-level summaries (end of each epoch)
EPOCH_DATA = [
    {"epoch": 1, "rl/critic_loss": 0.0163, "rl/actor_loss": 0.0437, "rl/pg_loss":  0.0007, "sub/reinforce_loss": 0.0462, "sub/div_rank_loss": 0.2884, "sub/total": 0.0750, "sub/alpha_mean": 0.5073, "val/hit@10": 0.0100, "val/ndcg@10": 0.0100, "val/mrr@10": 0.0100, "val/coverage": 0.0128},
    {"epoch": 2, "rl/critic_loss": 0.0188, "rl/actor_loss": 0.0344, "rl/pg_loss":  0.0013, "sub/reinforce_loss": 0.0509, "sub/div_rank_loss": 0.0818, "sub/total": 0.0591, "sub/alpha_mean": 0.5052, "val/hit@10": 0.0080, "val/ndcg@10": 0.0080, "val/mrr@10": 0.0080, "val/coverage": 0.0129},
    {"epoch": 3, "rl/critic_loss": 0.0173, "rl/actor_loss": 0.0266, "rl/pg_loss": -0.0001, "sub/reinforce_loss": 0.0464, "sub/div_rank_loss": 0.0846, "sub/total": 0.0548, "sub/alpha_mean": 0.5038, "val/hit@10": 0.0060, "val/ndcg@10": 0.0060, "val/mrr@10": 0.0060, "val/coverage": 0.0128},
]

# ---------------------------------------------------------------------------
# Extract series
# ---------------------------------------------------------------------------

steps  = [d[0] for d in LOG_DATA]
metrics = {k: [d[1][k] for d in LOG_DATA] for k in LOG_DATA[0][1]}

epochs     = [d["epoch"] for d in EPOCH_DATA]
val_hit    = [d["val/hit@10"] for d in EPOCH_DATA]
val_cov    = [d["val/coverage"] for d in EPOCH_DATA]
ep_div     = [d["sub/div_rank_loss"] for d in EPOCH_DATA]
ep_critic  = [d["rl/critic_loss"] for d in EPOCH_DATA]
ep_actor   = [d["rl/actor_loss"] for d in EPOCH_DATA]
ep_alpha   = [d["sub/alpha_mean"] for d in EPOCH_DATA]

EPOCH_BOUNDARIES = [1000, 2000]  # global step where each epoch starts

OUT_DIR = "/workspace/LM_Submodular_RL/output_amazon_full/plots"
os.makedirs(OUT_DIR, exist_ok=True)

STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
}

COLORS = {
    "critic":   "#58a6ff",
    "actor":    "#f78166",
    "pg":       "#ffa657",
    "reinforce":"#79c0ff",
    "div_rank": "#7ee787",
    "sub_total":"#d2a8ff",
    "alpha":    "#e3b341",
    "hit":      "#56d364",
    "coverage": "#79c0ff",
}


def add_epoch_lines(ax, boundaries=EPOCH_BOUNDARIES):
    for b in boundaries:
        ax.axvline(b, color="#484f58", linestyle=":", linewidth=1.2, alpha=0.8)


def smooth(values, w=3):
    """Simple moving average."""
    if len(values) < w:
        return values
    return np.convolve(values, np.ones(w) / w, mode="same").tolist()


# ===========================================================================
# Figure 1 — RL Losses (step-level)
# ===========================================================================
with plt.style.context(STYLE):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("RL Policy Losses (step-level)", fontsize=14, fontweight="bold", y=0.98)

    # Critic loss
    ax = axes[0]
    ax.plot(steps, metrics["rl/critic_loss"], color=COLORS["critic"], alpha=0.35, linewidth=1)
    ax.plot(steps, smooth(metrics["rl/critic_loss"], 5), color=COLORS["critic"], linewidth=2, label="critic_loss")
    add_epoch_lines(ax)
    ax.set_ylabel("Critic Loss (MSE)")
    ax.legend(loc="upper right")
    ax.grid(True)

    # Actor loss
    ax = axes[1]
    ax.plot(steps, metrics["rl/actor_loss"], color=COLORS["actor"], alpha=0.35, linewidth=1)
    ax.plot(steps, smooth(metrics["rl/actor_loss"], 5), color=COLORS["actor"], linewidth=2, label="actor_loss")
    add_epoch_lines(ax)
    ax.set_ylabel("Actor Loss")
    ax.legend(loc="upper right")
    ax.grid(True)

    # PG loss
    ax = axes[2]
    ax.axhline(0, color="#484f58", linewidth=1)
    ax.plot(steps, metrics["rl/pg_loss"], color=COLORS["pg"], alpha=0.35, linewidth=1)
    ax.plot(steps, smooth(metrics["rl/pg_loss"], 5), color=COLORS["pg"], linewidth=2, label="pg_loss")
    add_epoch_lines(ax)
    ax.set_ylabel("Policy Gradient Loss")
    ax.set_xlabel("Global Step")
    ax.legend(loc="upper right")
    ax.grid(True)

    for ax in axes:
        for b in EPOCH_BOUNDARIES:
            ax.text(b + 20, ax.get_ylim()[1] * 0.92, f"Epoch↑", color="#484f58", fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "01_rl_losses.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ===========================================================================
# Figure 2 — Submodular Losses (step-level)
# ===========================================================================
with plt.style.context(STYLE):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Submodular Losses (step-level)", fontsize=14, fontweight="bold", y=0.98)

    ax = axes[0]
    ax.plot(steps, metrics["sub/div_rank_loss"], color=COLORS["div_rank"], alpha=0.35, linewidth=1)
    ax.plot(steps, smooth(metrics["sub/div_rank_loss"], 5), color=COLORS["div_rank"], linewidth=2, label="div_rank_loss")
    add_epoch_lines(ax)
    ax.set_ylabel("Diversity Ranking Loss")
    ax.legend(loc="upper right")
    ax.grid(True)

    ax = axes[1]
    ax.axhline(0, color="#484f58", linewidth=1)
    ax.plot(steps, metrics["sub/reinforce_loss"], color=COLORS["reinforce"], alpha=0.35, linewidth=1)
    ax.plot(steps, smooth(metrics["sub/reinforce_loss"], 5), color=COLORS["reinforce"], linewidth=2, label="reinforce_loss")
    add_epoch_lines(ax)
    ax.set_ylabel("REINFORCE Loss")
    ax.legend(loc="upper right")
    ax.grid(True)

    ax = axes[2]
    ax.plot(steps, metrics["sub/total"], color=COLORS["sub_total"], alpha=0.35, linewidth=1)
    ax.plot(steps, smooth(metrics["sub/total"], 5), color=COLORS["sub_total"], linewidth=2, label="sub/total")
    add_epoch_lines(ax)
    ax.set_ylabel("Total Submodular Loss")
    ax.set_xlabel("Global Step")
    ax.legend(loc="upper right")
    ax.grid(True)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "02_submodular_losses.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ===========================================================================
# Figure 3 — Alpha (α) over steps
# ===========================================================================
with plt.style.context(STYLE):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("α_t — Relevance/Diversity Trade-off (Actor output)", fontsize=13, fontweight="bold")

    ax.axhline(0.5, color="#484f58", linewidth=1, linestyle="--", label="α = 0.5 (balanced)")
    ax.fill_between(steps, 0.45, 0.55, alpha=0.1, color=COLORS["alpha"])
    ax.plot(steps, metrics["sub/alpha_mean"], color=COLORS["alpha"], alpha=0.4, linewidth=1)
    ax.plot(steps, smooth(metrics["sub/alpha_mean"], 7), color=COLORS["alpha"], linewidth=2.5, label="alpha_mean")
    add_epoch_lines(ax)
    ax.set_ylim(0.40, 0.60)
    ax.set_ylabel("α (sigmoid of actor output)")
    ax.set_xlabel("Global Step")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "03_alpha_tradeoff.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ===========================================================================
# Figure 4 — Val metrics per epoch
# ===========================================================================
with plt.style.context(STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Validation Metrics per Epoch", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, val_hit, "o-", color=COLORS["hit"], linewidth=2.5, markersize=9, label="hit@10 = ndcg@10 = mrr@10")
    ax.axhline(0.0003, color="#484f58", linewidth=1, linestyle="--", label="Random baseline (~0.0003)")
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Hit@10 / NDCG@10 / MRR@10")
    ax.legend(fontsize=9)
    ax.grid(True)
    # Annotate best
    best_ep = epochs[val_hit.index(max(val_hit))]
    ax.annotate(f"Best: {max(val_hit):.4f}\n(epoch {best_ep})",
                xy=(best_ep, max(val_hit)),
                xytext=(best_ep + 0.15, max(val_hit) + 0.0005),
                color=COLORS["hit"], fontsize=9,
                arrowprops=dict(arrowstyle="->", color=COLORS["hit"]))

    ax = axes[1]
    ax.plot(epochs, val_cov, "s-", color=COLORS["coverage"], linewidth=2.5, markersize=9, label="coverage")
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of catalog")
    ax.set_title("Catalog Coverage")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "04_val_metrics.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ===========================================================================
# Figure 5 — Combined overview (all losses per epoch average)
# ===========================================================================
with plt.style.context(STYLE):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Training Overview — LM Submodular RL (3 Epochs, 50k Users, RTX 3090)",
                 fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # (0,0) div_rank over steps — most interesting
    ax = fig.add_subplot(gs[0, :])
    ax.plot(steps, metrics["sub/div_rank_loss"], color=COLORS["div_rank"], alpha=0.25, linewidth=1)
    ax.plot(steps, smooth(metrics["sub/div_rank_loss"], 5), color=COLORS["div_rank"], linewidth=2.5, label="div_rank_loss (diversity embedding)")
    ax.plot(steps, metrics["sub/reinforce_loss"], color=COLORS["reinforce"], alpha=0.25, linewidth=1)
    ax.plot(steps, smooth(metrics["sub/reinforce_loss"], 5), color=COLORS["reinforce"], linewidth=2, linestyle="--", label="reinforce_loss")
    add_epoch_lines(ax)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Global Step")
    ax.set_title("Submodular Losses (step-level)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)
    for b in EPOCH_BOUNDARIES:
        ax.text(b + 15, ax.get_ylim()[1] * 0.88, "E↑", color="#8b949e", fontsize=9)

    # (1,0) critic + actor per epoch
    ax = fig.add_subplot(gs[1, 0])
    x = np.array(epochs)
    ax.bar(x - 0.2, ep_critic, 0.35, color=COLORS["critic"], label="critic_loss", alpha=0.85)
    ax.bar(x + 0.2, ep_actor,  0.35, color=COLORS["actor"],  label="actor_loss",  alpha=0.85)
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("RL Losses (epoch avg)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")

    # (1,1) val hit@10
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(epochs, val_hit, "o-", color=COLORS["hit"], linewidth=2.5, markersize=10)
    for e, h in zip(epochs, val_hit):
        ax.annotate(f"{h:.4f}", (e, h), textcoords="offset points", xytext=(5, 6),
                    color=COLORS["hit"], fontsize=10)
    ax.axhline(0.0003, color="#484f58", linewidth=1, linestyle="--")
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_title("Val Hit@10")
    ax.grid(True)

    # (2,0) alpha over steps
    ax = fig.add_subplot(gs[2, 0])
    ax.axhline(0.5, color="#484f58", linewidth=1, linestyle="--")
    ax.fill_between(steps, 0.45, 0.55, alpha=0.08, color=COLORS["alpha"])
    ax.plot(steps, smooth(metrics["sub/alpha_mean"], 7), color=COLORS["alpha"], linewidth=2)
    add_epoch_lines(ax)
    ax.set_ylim(0.40, 0.60)
    ax.set_xlabel("Global Step")
    ax.set_title("α Trade-off (Actor)")
    ax.grid(True)

    # (2,1) coverage per epoch
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(epochs, val_cov, "s-", color=COLORS["coverage"], linewidth=2.5, markersize=10)
    for e, c in zip(epochs, val_cov):
        ax.annotate(f"{c:.4f}", (e, c), textcoords="offset points", xytext=(5, 4),
                    color=COLORS["coverage"], fontsize=10)
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_title("Val Coverage")
    ax.grid(True)

    out = os.path.join(OUT_DIR, "05_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


print(f"\nAll plots saved to: {OUT_DIR}")
