"""
SASRec Retrieval Training — thay thế BM25 Stage-1

Mục tiêu: train SASRec để dùng thay BM25 trong pipeline LM_Submodular_RL.
  BM25 recall@200 ≈ 5%  (sẽ được so sánh trong kết quả)
  SASRec recall@200 → mục tiêu 30-50%

Cách train:
  "User mua [A,B,C] → tiếp theo mua D"
  → cosine_sim(SASRec([A,B,C]), item_emb[D]) nên cao
  → cosine_sim(SASRec([A,B,C]), item_emb[random]) nên thấp

Sau train: item_emb của 2.6M sản phẩm → FAISS index
Inference: user_emb = SASRec(history) → FAISS.search(top_k)

Reference: Kang & McAuley, ICDM 2018 — https://arxiv.org/abs/1808.09781
PyTorch port: https://github.com/pmixer/SASRec.pytorch

Usage:
  # Quick test (500k records, 2 epochs)
  python -m retrieval_models.sasrec.train --max_train 500000 --epochs 2

  # Full run (11.79M records, 5 epochs) — run on GPU
  python -m retrieval_models.sasrec.train --epochs 5

  # Skip preprocessing cache (reuse from previous run)
  python -m retrieval_models.sasrec.train --skip_preprocess
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
from retrieval_models.sasrec.model import SASRec


# ── Paths ──────────────────────────────────────────────────────────────────
PRODUCTS_PATH = "/workspace/amazon/output_amazon/products.jsonl"
TRAIN_PATH    = "/workspace/amazon/dataset/train.jsonl"
TEST_PATH     = "/workspace/amazon/dataset/test.jsonl"

# BM25 baseline từ check_bm25_recall.py (last-3-titles, test split)
BM25_BASELINE = {10: 0.015, 50: 0.030, 100: 0.042, 200: 0.052}


# ===========================================================================
# 1. Build asin → item_id map
# ===========================================================================

def build_asin2iid(products_path: str) -> dict:
    """
    Đọc products.jsonl → {asin: item_id}.
    item_id bắt đầu từ 1 (0 dành cho padding trong SASRec).
    """
    asin2iid = {}
    with open(products_path) as f:
        for line in tqdm(f, desc="products.jsonl", unit="rec", dynamic_ncols=True):
            p = json.loads(line)
            asin2iid[p["asin"]] = p["item_id"] + 1   # 1-indexed; 0 = padding
    print(f"  {len(asin2iid):,} sản phẩm  (ids 1..{max(asin2iid.values()):,})", flush=True)
    return asin2iid


# ===========================================================================
# 2. Preprocess jsonl → numpy arrays (cached)
# ===========================================================================

def preprocess_jsonl(jsonl_path, asin2iid, maxlen, cache_dir, split_name):
    """
    Chuyển đổi records trong jsonl sang (seqs, targets) numpy.
    Kết quả được cache để không phải đọc lại file lần sau.
    """
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
            seq[-len(hist):] = hist          # left-pad
            seqs.append(seq)
            targets.append(target)

    seqs_arr = np.stack(seqs).astype(np.int32)
    tgts_arr = np.array(targets, dtype=np.int32)
    np.save(seq_f, seqs_arr)
    np.save(tgt_f, tgts_arr)
    print(f"  {len(tgts_arr):,} records → {seq_f}", flush=True)
    return seqs_arr, tgts_arr


# ===========================================================================
# 3. Dataset với online negative sampling
# ===========================================================================

class SequenceDataset(Dataset):
    def __init__(self, seqs, targets, item_num, num_neg=1):
        self.seqs     = seqs
        self.targets  = targets
        self.item_num = item_num
        self.num_neg  = num_neg

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        seq    = self.seqs[idx].astype(np.int64)
        target = int(self.targets[idx])
        hist_set = set(seq[seq != 0]) | {target}
        negs = []
        while len(negs) < self.num_neg:
            n = random.randint(1, self.item_num)
            if n not in hist_set:
                negs.append(n)
        return (
            torch.from_numpy(seq),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(negs,   dtype=torch.long),
        )


# ===========================================================================
# 4. Build FAISS index từ item embeddings sau training
# ===========================================================================

def build_faiss_index(model, item_num, device, batch_size=8192):
    import faiss
    model.eval()
    all_embs = []
    ids_full = torch.arange(1, item_num + 1, device=device)
    for start in tqdm(range(0, item_num, batch_size),
                      desc="indexing items", unit="batch", dynamic_ncols=True):
        chunk = ids_full[start: start + batch_size]
        with torch.no_grad():
            emb = nn.functional.normalize(model.item_emb(chunk), dim=-1)
        all_embs.append(emb.cpu().numpy().astype(np.float32))
    all_embs = np.vstack(all_embs)
    index = faiss.IndexFlatIP(all_embs.shape[1])
    index.add(all_embs)
    print(f"  FAISS: {index.ntotal:,} vectors, dim={all_embs.shape[1]}", flush=True)
    return index


# ===========================================================================
# 5. Recall@K evaluation — so sánh với BM25 baseline
# ===========================================================================

def evaluate_recall(model, faiss_idx, jsonl_path, asin2iid, maxlen,
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
            uembs = model.user_embedding(seqs_t).cpu().numpy().astype(np.float32)
        _, indices = faiss_idx.search(uembs, max_k)
        for row, target in zip(indices, tgts_buf):
            result_iids = row + 1    # FAISS row 0 → item_id 1
            for k in topk_list:
                if target in result_iids[:k]:
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
        sas = recall[k]
        tag = f"  (+{(sas-bm)/max(bm,1e-6)*100:.0f}% vs BM25)" if sas > bm else \
              f"  ({(sas-bm)/max(bm,1e-6)*100:.0f}% vs BM25)"
        bar = "█" * min(int(sas * 200), 50)
        print(f"  @{k:<4}: BM25={bm:.4f}  SASRec={sas:.4f}{tag}  {bar}")
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
        print("  Run: huggingface-cli login   then retry", flush=True)


# ===========================================================================
# Args
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--products_path", default=PRODUCTS_PATH)
    p.add_argument("--train_path",    default=TRAIN_PATH)
    p.add_argument("--test_path",     default=TEST_PATH)
    p.add_argument("--output_dir",    default="retrieval_models/sasrec/output")
    # Model
    p.add_argument("--maxlen",     type=int,   default=20)
    p.add_argument("--hidden_dim", type=int,   default=64)
    p.add_argument("--num_heads",  type=int,   default=1)
    p.add_argument("--num_blocks", type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--num_neg",    type=int,   default=1)
    # Training
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch_size", type=int,   default=1024)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--max_train",  type=int,   default=None,
                   help="Giới hạn records train. VD: 500000 để test nhanh.")
    # Eval
    p.add_argument("--topk",        type=int, nargs="+", default=[10, 50, 100, 200])
    p.add_argument("--eval_sample", type=int, default=5000)
    p.add_argument("--eval_batch",  type=int, default=256)
    # Misc
    p.add_argument("--device",  default="auto")
    p.add_argument("--hf_repo", default="chuannb/sasrec-amazon-retrieval")
    p.add_argument("--no_upload",       action="store_true")
    p.add_argument("--skip_preprocess", action="store_true")
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

    # ── 1. asin2iid ───────────────────────────────────────────────────────────
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

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    train_seqs, train_targets = preprocess_jsonl(
        args.train_path, asin2iid, args.maxlen, pre_dir, "train"
    )
    if args.max_train and len(train_targets) > args.max_train:
        idx = np.random.permutation(len(train_targets))[: args.max_train]
        train_seqs, train_targets = train_seqs[idx], train_targets[idx]
        print(f"  Capped: {len(train_targets):,} records", flush=True)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = SASRec(
        item_num=item_num, maxlen=args.maxlen, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, num_blocks=args.num_blocks, dropout_rate=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nSASRec params={n_params:,}", flush=True)

    dataset = SequenceDataset(train_seqs, train_targets, item_num, args.num_neg)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, pin_memory=(device.type == "cuda"),
                         persistent_workers=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # ── 4. Training ───────────────────────────────────────────────────────────
    print(f"\nTraining: {args.epochs} epochs × {len(dataset):,} records "
          f"(batch={args.batch_size})\n", flush=True)

    best_r200  = 0.0
    history    = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}",
                    unit="batch", dynamic_ncols=True)
        for seqs, pos_ids, neg_ids in pbar:
            seqs    = seqs.to(device)
            pos_ids = pos_ids.to(device)
            neg_ids = neg_ids.to(device)
            pos_logits, neg_logits = model(seqs, pos_ids, neg_ids)
            loss = model.bce_loss(pos_logits, neg_logits)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"\nEpoch {epoch}  loss={avg_loss:.4f}  {time.perf_counter()-t0:.0f}s", flush=True)

        faiss_idx = build_faiss_index(model, item_num, device)
        recall = evaluate_recall(
            model, faiss_idx, args.test_path, asin2iid,
            args.maxlen, args.topk, device,
            sample=args.eval_sample, batch_size=args.eval_batch,
        )
        r200 = recall.get(200, recall.get(max(args.topk), 0.0))
        history.append({"epoch": epoch, "loss": avg_loss, "recall": recall})

        if r200 > best_r200:
            best_r200 = r200
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "recall": recall, "args": vars(args),
            }, out_dir / "best_model.pt")
            print(f"  ✓ best recall@{max(args.topk)}={r200:.4f} saved", flush=True)

    # ── 5. Final eval ─────────────────────────────────────────────────────────
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nBest checkpoint: epoch {ckpt['epoch']}", flush=True)

    faiss_idx    = build_faiss_index(model, item_num, device)
    final_recall = evaluate_recall(
        model, faiss_idx, args.test_path, asin2iid,
        args.maxlen, args.topk, device,
        sample=args.eval_sample, batch_size=args.eval_batch,
    )

    # Save FAISS index
    import faiss
    faiss.write_index(faiss_idx, str(out_dir / "item_faiss.index"))
    print(f"FAISS saved ({(out_dir/'item_faiss.index').stat().st_size/1e6:.0f} MB)", flush=True)

    # Save results
    config = dict(model_type="SASRec", item_num=item_num, maxlen=args.maxlen,
                  hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                  num_blocks=args.num_blocks, dropout=args.dropout)
    results = dict(
        model="SASRec",
        paper="Self-Attentive Sequential Recommendation, ICDM 2018",
        github_reference="https://github.com/pmixer/SASRec.pytorch",
        config=config,
        train_records=len(train_targets),
        best_epoch=ckpt["epoch"],
        bm25_baseline=BM25_BASELINE,
        sasrec_recall=final_recall,
        training_history=history,
    )
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'═'*55}")
    print("  SUMMARY — SASRec vs BM25 (last-3-titles)")
    print(f"{'═'*55}")
    for k in sorted(args.topk):
        bm  = BM25_BASELINE.get(k, 0.0)
        sas = final_recall.get(k, 0.0)
        d   = f"+{(sas-bm)/max(bm,1e-6)*100:.0f}%" if sas > bm else f"{(sas-bm)/max(bm,1e-6)*100:.0f}%"
        print(f"  Recall@{k:<4}: BM25={bm:.4f}  SASRec={sas:.4f}  [{d}]")
    print(f"{'═'*55}")
    print(f"Results → {out_dir}/results.json")

    # HuggingFace upload
    if not args.no_upload:
        print(f"\nUploading to HuggingFace: {args.hf_repo}...", flush=True)
        upload_hf(out_dir, args.hf_repo)


if __name__ == "__main__":
    main()
