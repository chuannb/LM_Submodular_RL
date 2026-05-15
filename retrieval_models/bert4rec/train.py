"""
BERT4Rec Retrieval Training — thay thế BM25 Stage-1

Mục tiêu: train BERT4Rec để dùng thay BM25 trong pipeline LM_Submodular_RL.
  BM25  recall@200 ≈ 5%   (baseline)
  BERT4Rec recall@200 → mục tiêu 35-55%

Cloze task:
  - Mỗi sequence trong history, mask ngẫu nhiên ~20% item
  - 80% thay bằng [MASK] token, 10% thay bằng random item, 10% giữ nguyên
  - Loss BCE tại các masked positions
  - Reference: BertTrainDataset từ jaywonchung/BERT4Rec-VAE-Pytorch

Inference:
  - Append [MASK] vào cuối history → encode → lấy embedding tại vị trí MASK
  - L2-normalize → FAISS.search(top_k)

Usage:
  # Quick test (500k records, 2 epochs)
  python -m retrieval_models.bert4rec.train --max_train 500000 --epochs 2

  # Full run (11.79M records, 5 epochs)
  python -m retrieval_models.bert4rec.train --epochs 5

  # Resume từ checkpoint
  python -m retrieval_models.bert4rec.train --resume --skip_preprocess --epochs 10

Reference:
  Sun et al., "BERT4Rec", CIKM 2019. https://arxiv.org/abs/1904.06690
  PyTorch port: https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from retrieval_models.bert4rec.model import BERT4Rec


# ── Paths ──────────────────────────────────────────────────────────────────
PRODUCTS_PATH = "/workspace/amazon/output_amazon/products.jsonl"
TRAIN_PATH    = "/workspace/amazon/dataset/train.jsonl"
TEST_PATH     = "/workspace/amazon/dataset/test.jsonl"

BM25_BASELINE = {10: 0.015, 50: 0.030, 100: 0.042, 200: 0.052}


# ===========================================================================
# 1. Build asin → item_id map  (reuse same logic as SASRec)
# ===========================================================================

def build_asin2iid(products_path: str) -> dict:
    asin2iid = {}
    with open(products_path) as f:
        for line in tqdm(f, desc="products.jsonl", unit="rec", dynamic_ncols=True):
            p = json.loads(line)
            asin2iid[p["asin"]] = p["item_id"] + 1   # 1-indexed; 0 = padding
    print(f"  {len(asin2iid):,} products  (ids 1..{max(asin2iid.values()):,})", flush=True)
    return asin2iid


# ===========================================================================
# 2. Preprocess jsonl → numpy arrays (identical to SASRec, shared cache)
# ===========================================================================

def preprocess_jsonl(jsonl_path, asin2iid, maxlen, cache_dir, split_name):
    seq_f = cache_dir / f"{split_name}_seqs_L{maxlen}.npy"
    tgt_f = cache_dir / f"{split_name}_targets.npy"

    if seq_f.exists() and tgt_f.exists():
        print(f"  Cache hit: {split_name} ({seq_f.stat().st_size/1e6:.0f} MB)", flush=True)
        return np.load(seq_f), np.load(tgt_f)

    print(f"\nPreprocessing {Path(jsonl_path).name} → numpy...", flush=True)
    seqs, targets = [], []

    with open(jsonl_path) as f:
        for line in tqdm(f, desc=split_name, unit="rec", dynamic_ncols=True):
            rec    = json.loads(line)
            target = asin2iid.get(rec.get("target_asin", ""))
            if target is None:
                continue
            hist = [
                asin2iid[h["asin"]]
                for h in rec.get("history", [])
                if h.get("asin") in asin2iid
            ]
            if not hist:
                continue
            hist = hist[-maxlen:]
            seq  = np.zeros(maxlen, dtype=np.int32)
            seq[-len(hist):] = hist
            seqs.append(seq)
            targets.append(target)

    seqs_arr = np.stack(seqs).astype(np.int32)
    tgts_arr = np.array(targets, dtype=np.int32)
    np.save(seq_f, seqs_arr)
    np.save(tgt_f, tgts_arr)
    print(f"  {len(tgts_arr):,} records → {seq_f}", flush=True)
    return seqs_arr, tgts_arr


# ===========================================================================
# 3. Dataset — Cloze task (ported from BertTrainDataset in reference repo)
# ===========================================================================

class BERT4RecDataset(Dataset):
    """
    Cloze masking dataset.

    For each sequence:
      - Include target as last item (shift sequence one step)
      - For each non-padding position, with probability mask_prob:
          80% → replace with [MASK] token
          10% → replace with random item
          10% → keep original
      - Guarantee at least 1 position is masked per sample
      - pos_ids: original item at masked positions, 0 elsewhere
      - neg_ids: random negative at masked positions, 0 elsewhere

    Reference: BertTrainDataset.__getitem__ in jaywonchung/BERT4Rec-VAE-Pytorch
    """

    def __init__(
        self,
        seqs:       np.ndarray,   # (N, L) pre-padded, left-aligned
        targets:    np.ndarray,   # (N,)
        item_num:   int,
        mask_prob:  float = 0.2,
        mask_token: int   = None,
    ):
        self.seqs       = seqs
        self.targets    = targets
        self.item_num   = item_num
        self.mask_prob  = mask_prob
        self.mask_token = mask_token if mask_token is not None else item_num + 1
        self.maxlen     = seqs.shape[1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        seq    = self.seqs[idx].astype(np.int64)   # (L,) left-padded
        target = int(self.targets[idx])

        # Append target as last item, shift left by 1 to create a richer sequence
        # [pad, i0, i1, ..., i_{L-2}] → [i0, i1, ..., i_{L-2}, target]
        seq_with_target       = np.empty(self.maxlen, dtype=np.int64)
        seq_with_target[:-1]  = seq[1:]    # drop first (leftmost) slot
        seq_with_target[-1]   = target

        # Cloze masking (80/10/10 scheme from BERT paper via reference repo)
        masked_seq = seq_with_target.copy()
        pos_ids = np.zeros(self.maxlen, dtype=np.int64)   # original item at masked positions

        non_pad_positions = np.where(seq_with_target != 0)[0]
        if len(non_pad_positions) == 0:
            return (
                torch.zeros(self.maxlen, dtype=torch.long),
                torch.zeros(self.maxlen, dtype=torch.long),
            )

        # Decide which positions to mask
        will_mask = [p for p in non_pad_positions if random.random() < self.mask_prob]
        if not will_mask:
            will_mask = [non_pad_positions[-1]]   # at least 1 per sample

        for pos in will_mask:
            orig = seq_with_target[pos]
            r    = random.random()
            if r < 0.8:
                masked_seq[pos] = self.mask_token          # 80% → [MASK]
            elif r < 0.9:
                masked_seq[pos] = random.randint(1, self.item_num)  # 10% → random
            # else: keep original (10%)
            pos_ids[pos] = orig

        return (
            torch.from_numpy(masked_seq),
            torch.from_numpy(pos_ids),
        )


# ===========================================================================
# 4. Build item embeddings for FAISS
# ===========================================================================

def build_item_embs(model, item_num, device, batch_size=8192):
    model.eval()
    parts    = []
    ids_full = torch.arange(1, item_num + 1, device=device)
    for start in tqdm(range(0, item_num, batch_size),
                      desc="building item embs", unit="batch", dynamic_ncols=True):
        chunk = ids_full[start: start + batch_size]
        with torch.no_grad():
            emb = nn.functional.normalize(model.embedding.token(chunk), dim=-1)
        parts.append(emb)
    item_embs = torch.cat(parts, dim=0)
    print(f"  item embs: {item_embs.shape}, "
          f"{item_embs.element_size()*item_embs.numel()/1e6:.0f} MB on {device}", flush=True)
    return item_embs


def build_faiss_index_cpu(item_embs_gpu):
    import faiss
    arr   = item_embs_gpu.cpu().numpy().astype(np.float32)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    return index


# ===========================================================================
# 5. Recall@K evaluation (GPU matmul, identical structure to SASRec)
# ===========================================================================

def evaluate_recall(model, item_embs, jsonl_path, asin2iid, maxlen,
                    topk_list, device, sample=5000, batch_size=256, seed=42):
    random.seed(seed)
    reservoir = []
    with open(jsonl_path) as f:
        for i, line in enumerate(tqdm(f, desc="sampling test", unit="rec", dynamic_ncols=True)):
            rec = json.loads(line)
            if len(reservoir) < sample:
                reservoir.append(rec)
            else:
                j = random.randint(0, i)
                if j < sample:
                    reservoir[j] = rec

    max_k = max(topk_list)
    hits  = {k: 0 for k in topk_list}
    valid = 0
    model.eval()
    seqs_buf, tgts_buf = [], []

    def flush():
        nonlocal valid
        if not seqs_buf:
            return
        seqs_t = torch.from_numpy(np.stack(seqs_buf)).to(device)
        with torch.no_grad():
            uembs   = model.user_embedding(seqs_t)           # (B, d)
            scores  = torch.matmul(uembs, item_embs.T)       # (B, item_num)
            _, topk = scores.topk(max_k, dim=1)              # (B, max_k)
        topk_np = (topk + 1).cpu().numpy()                   # → 1-indexed item_id
        for row, target in zip(topk_np, tgts_buf):
            for k in topk_list:
                if target in row[:k]:
                    hits[k] += 1
            valid += 1
        seqs_buf.clear()
        tgts_buf.clear()

    for rec in tqdm(reservoir, desc="recall@K", unit="q", dynamic_ncols=True):
        target = asin2iid.get(rec.get("target_asin", ""))
        if target is None:
            continue
        hist = [asin2iid[h["asin"]] for h in rec.get("history", [])
                if h.get("asin") in asin2iid]
        if not hist:
            continue
        hist = hist[-maxlen:]
        seq  = np.zeros(maxlen, dtype=np.int64)
        seq[-len(hist):] = hist
        seqs_buf.append(seq)
        tgts_buf.append(target)
        if len(seqs_buf) >= batch_size:
            flush()
    flush()

    recall = {k: hits[k] / max(valid, 1) for k in topk_list}
    print(f"\n{'─'*60}")
    print(f"  Recall ({valid:,} test samples)")
    print(f"{'─'*60}")
    for k in sorted(topk_list):
        bm  = BM25_BASELINE.get(k, 0.0)
        b4r = recall[k]
        tag = f"  (+{(b4r-bm)/max(bm,1e-6)*100:.0f}% vs BM25)" if b4r > bm else \
              f"  ({(b4r-bm)/max(bm,1e-6)*100:.0f}% vs BM25)"
        bar = "█" * min(int(b4r * 200), 50)
        print(f"  @{k:<4}: BM25={bm:.4f}  BERT4Rec={b4r:.4f}{tag}  {bar}")
    print(f"{'─'*60}")
    return recall


# ===========================================================================
# 6. HuggingFace upload
# ===========================================================================

def upload_hf(out_dir, repo_id):
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["preprocessed/"],
        )
        print(f"  → https://huggingface.co/{repo_id}", flush=True)
    except Exception as e:
        print(f"  HF upload failed: {e}", flush=True)


# ===========================================================================
# Args
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--products_path", default=PRODUCTS_PATH)
    p.add_argument("--train_path",    default=TRAIN_PATH)
    p.add_argument("--test_path",     default=TEST_PATH)
    p.add_argument("--output_dir",    default="retrieval_models/bert4rec/output")
    # Model
    p.add_argument("--maxlen",     type=int,   default=50,
                   help="Max sequence length. Larger than SASRec(20) since most users hit the limit.")
    p.add_argument("--hidden_dim", type=int,   default=128,
                   help="Hidden dim. Larger than SASRec(64) for 2.6M item catalog.")
    p.add_argument("--num_heads",  type=int,   default=2)
    p.add_argument("--num_blocks", type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--mask_prob",  type=float, default=0.2,
                   help="Cloze mask probability (original BERT4Rec uses 0.2).")
    # Training
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch_size", type=int,   default=1024)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07,
                   help="InfoNCE temperature. Lower → harder negatives. Default 0.07.")
    p.add_argument("--max_train",  type=int,   default=None,
                   help="Cap training records. E.g. 500000 for quick test.")
    # Eval
    p.add_argument("--topk",        type=int, nargs="+", default=[10, 50, 100, 200])
    p.add_argument("--eval_sample", type=int, default=5000)
    p.add_argument("--eval_batch",  type=int, default=256)
    # Misc
    p.add_argument("--device",          default="auto")
    p.add_argument("--hf_repo",         default="chuannb/bert4rec-amazon-retrieval")
    p.add_argument("--no_upload",       action="store_true")
    p.add_argument("--skip_preprocess", action="store_true",
                   help="Reuse cached numpy arrays from previous run.")
    p.add_argument("--resume",          action="store_true",
                   help="Resume training from last_checkpoint.pt.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    out_dir = Path(args.output_dir)
    pre_dir = out_dir / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_dir.mkdir(exist_ok=True)

    # ── 1. asin2iid ────────────────────────────────────────────────────────
    iid_cache = pre_dir / "asin2iid.json"
    if args.skip_preprocess and iid_cache.exists():
        print("Loading cached asin2iid...", flush=True)
        with open(iid_cache) as f:
            asin2iid = json.load(f)
        print(f"  {len(asin2iid):,} products", flush=True)
    else:
        asin2iid = build_asin2iid(args.products_path)
        with open(iid_cache, "w") as f:
            json.dump(asin2iid, f)
    item_num = max(asin2iid.values())
    print(f"item_num={item_num:,}  |  emb table ≈ "
          f"{item_num * args.hidden_dim * 4 / 1e6:.0f} MB", flush=True)

    # ── 2. Preprocess ──────────────────────────────────────────────────────
    # Note: shares the same numpy cache dir as SASRec if maxlen matches.
    # BERT4Rec uses maxlen=50 by default (different from SASRec's 20),
    # so it builds its own cache under bert4rec/output/preprocessed/.
    train_seqs, train_targets = preprocess_jsonl(
        args.train_path, asin2iid, args.maxlen, pre_dir, "train"
    )
    if args.max_train and len(train_targets) > args.max_train:
        idx = np.random.permutation(len(train_targets))[: args.max_train]
        train_seqs, train_targets = train_seqs[idx], train_targets[idx]
        print(f"  Capped: {len(train_targets):,} records", flush=True)

    # ── 3. Model ───────────────────────────────────────────────────────────
    model = BERT4Rec(
        item_num=item_num,
        maxlen=args.maxlen,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout,
        mask_prob=args.mask_prob,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    mask_token = model.mask_token
    print(f"\nBERT4Rec params={n_params:,}  mask_token={mask_token}", flush=True)

    dataset = BERT4RecDataset(
        train_seqs, train_targets, item_num,
        mask_prob=args.mask_prob,
        mask_token=mask_token,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    temperature = args.temperature

    # ── 4. Training ────────────────────────────────────────────────────────
    print(f"\nTraining: {args.epochs} epochs × {len(dataset):,} records "
          f"(batch={args.batch_size}, mask_prob={args.mask_prob}, temp={args.temperature})\n", flush=True)

    start_epoch = 1
    best_r200   = 0.0
    history     = []

    if args.resume:
        ckpt_path = out_dir / "last_checkpoint.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            best_r200   = ckpt["best_r200"]
            history     = ckpt.get("history", [])
            print(f"Resumed from epoch {ckpt['epoch']}, best_r200={best_r200:.4f}", flush=True)
        else:
            print("--resume: no last_checkpoint.pt found, starting fresh.", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}",
                    unit="batch", dynamic_ncols=True)
        for masked_seqs, pos_ids in pbar:
            masked_seqs = masked_seqs.to(device)
            pos_ids     = pos_ids.to(device)

            loss = model.infonce_loss(masked_seqs, pos_ids, temperature=temperature)
            if loss == 0:
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        print(f"\nEpoch {epoch}  loss={avg_loss:.4f}  {time.perf_counter()-t0:.0f}s", flush=True)

        item_embs = build_item_embs(model, item_num, device)
        recall    = evaluate_recall(
            model, item_embs, args.test_path, asin2iid,
            args.maxlen, args.topk, device,
            sample=args.eval_sample, batch_size=args.eval_batch,
        )
        del item_embs
        r200 = recall.get(200, recall.get(max(args.topk), 0.0))
        history.append({"epoch": epoch, "loss": avg_loss, "recall": recall})

        if r200 > best_r200:
            best_r200 = r200
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "recall": recall, "args": vars(args),
            }, out_dir / "best_model.pt")
            print(f"  ✓ best recall@{max(args.topk)}={r200:.4f} saved", flush=True)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_r200": best_r200,
            "history": history,
        }, out_dir / "last_checkpoint.pt")

    # ── 5. Final eval + FAISS ──────────────────────────────────────────────
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nBest checkpoint: epoch {ckpt['epoch']}", flush=True)

    item_embs    = build_item_embs(model, item_num, device)
    final_recall = evaluate_recall(
        model, item_embs, args.test_path, asin2iid,
        args.maxlen, args.topk, device,
        sample=args.eval_sample, batch_size=args.eval_batch,
    )

    import faiss
    faiss_index = build_faiss_index_cpu(item_embs)
    del item_embs
    faiss.write_index(faiss_index, str(out_dir / "item_faiss.index"))
    print(f"FAISS saved ({(out_dir/'item_faiss.index').stat().st_size/1e6:.0f} MB)", flush=True)

    # Save results & config
    config = dict(
        model_type="BERT4Rec",
        item_num=item_num, maxlen=args.maxlen, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, num_blocks=args.num_blocks,
        dropout=args.dropout, mask_prob=args.mask_prob,
    )
    results = dict(
        model="BERT4Rec",
        paper="Sequential Recommendation with Bidirectional Encoder Representations from Transformer, CIKM 2019",
        github_reference="https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch",
        config=config,
        train_records=len(train_targets),
        best_epoch=ckpt["epoch"],
        bm25_baseline=BM25_BASELINE,
        bert4rec_recall=final_recall,
        training_history=history,
    )
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'═'*58}")
    print("  SUMMARY — BERT4Rec vs BM25 (last-3-titles)")
    print(f"{'═'*58}")
    for k in sorted(args.topk):
        bm  = BM25_BASELINE.get(k, 0.0)
        b4r = final_recall.get(k, 0.0)
        d   = f"+{(b4r-bm)/max(bm,1e-6)*100:.0f}%" if b4r > bm else \
              f"{(b4r-bm)/max(bm,1e-6)*100:.0f}%"
        print(f"  Recall@{k:<4}: BM25={bm:.4f}  BERT4Rec={b4r:.4f}  [{d}]")
    print(f"{'═'*58}")
    print(f"Results → {out_dir}/results.json")

    if not args.no_upload:
        print(f"\nUploading to HuggingFace: {args.hf_repo}...", flush=True)
        upload_hf(out_dir, args.hf_repo)


if __name__ == "__main__":
    main()
