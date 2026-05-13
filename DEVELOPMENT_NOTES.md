# Development Notes — LM Submodular RL

> Tài liệu này ghi lại trạng thái hiện tại, vấn đề đang giải quyết, và các hướng đi tiếp theo.
> Mục đích: setup lại server mới và tiếp tục đúng chỗ.

---

## 1. Trạng thái server hiện tại (BROKEN — cần fix)

**Vấn đề:** Container thiếu NVIDIA Container Runtime → CUDA không khởi được.

```
/usr/local/nvidia/   ← KHÔNG TỒN TẠI (chưa được mount)
LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64  ← set nhưng path không có
cuInit() → error 999 (CUDA_ERROR_UNKNOWN)
```

**Nguyên nhân:** Container chạy thiếu `--runtime=nvidia` hoặc `--gpus all`.  
**Fix:** Restart container với proper GPU support (xem Section 6).

**Hardware server:**
- GPU: 4x NVIDIA GeForce RTX 3090 (24 GB VRAM mỗi card)
- Driver: 580.95.05, CUDA 13.0
- Python env: `/venv/main/bin/python3` (Python 3.12)
- PyTorch: 2.11.0+cu130 ← đúng version, không cần reinstall
- faiss-cpu: 1.13.2 ✓
- huggingface_hub: 1.8.0 ✓

---

## 2. Data

### Paths trên server hiện tại
```
/workspace/amazon/
├── output_amazon/
│   ├── products.jsonl          # 2,611,529 sản phẩm, ~640 MB
│   │   Format: {"item_id": 0, "asin": "...", "title": "...",
│   │            "categories": "...", "price": 0.0, ...}
│   ├── bm25_index.pkl          # BM25 index đã build, 93 MB
│   └── id_map.json             # ASIN → item_id
└── dataset/
    ├── train.jsonl             # 11,790,525 records
    ├── val.jsonl               # 999,938 records
    └── test.jsonl              # 999,938 records
        Format: {"user_id": "...", "history": [{"asin": "...", "stars": N, "ts": N}],
                 "target_asin": "...", "target_stars": N, "ts": N}
```

### Scale
| | Số lượng |
|---|---|
| Sản phẩm (catalog) | 2,611,529 |
| Train records | 11,790,525 |
| Val/Test records | 999,938 mỗi set |
| Unique target items (test) | ~18,264 |

### Split strategy: Leave-Last-2-Out theo thời gian
```
User [i0, i1, ..., i_{n-1}]:
  train : t = 1..n-3  (rolling window)
  val   : t = n-2     (1 sample/user)
  test  : t = n-1     (1 sample/user)
```

---

## 3. Pipeline hiện tại (LM_Submodular_RL)

```
User history (ASINs)
    │
    ▼ Stage 1 — Retrieval
[BM25] last-3 history titles → top-100 candidates
    │  recall@200 ≈ 5%  ← BOTTLENECK CHÍNH
    ▼ Stage 2 — Reranking
[Qwen3-Reranker-0.6B] p(yes) per candidate
    │
    ▼ Stage 3 — Slate Optimization
[StateEncoder GRU] history → user_state (128-dim)
[RL Actor-Critic]  user_state → (α_t, κ_t)
[RerankerBackedSubmodular] f(S) = α·rel(S) + (1-α)·div(S)
[BudgetedSubmodularGreedy] → slate top-10
```

### Kết quả baseline hiện tại
| Metric | Giá trị | Ghi chú |
|---|---|---|
| BM25 Recall@200 | ~5.2% | Đo trên 2000 test samples |
| BM25 Recall@100 | ~4.2% | |
| Hit@10 (full pipeline) | 1.0% | Epoch 1 best |
| Coverage | 1.3% | Items được gợi ý |

### Vấn đề cốt lõi: Sparse Reward
- 95% training steps có `reward = 0` (vì target không có trong 100 BM25 candidates)
- RL không học được khi không có signal
- inject_target=True là hack training — không dùng khi eval

---

## 4. Hướng đi hiện tại: SASRec Two-Tower Retrieval

**Branch:** `sasrec-backbone`  
**Lý do chọn:** SASRec (ICDM 2018) là sequential recommendation model state-of-art đầu tiên dùng self-attention. Sau khi train, item embeddings → FAISS index thay thế BM25.

### Cách hoạt động
```
Training:
  "User mua [A,B,C] → tiếp theo mua D"
  → cosine_sim(SASRec([A,B,C]), item_emb[D]) nên cao
  → cosine_sim(SASRec([A,B,C]), item_emb[random]) nên thấp
  Loss: Binary Cross-Entropy

Inference:
  user_emb = SASRec(history_ids)  →  FAISS.search(user_emb, top_200)
  [thay BM25(last_3_titles)]
```

### Files
```
retrieval_models/sasrec/
├── model.py     # SASRec architecture (Pre-LN, normalized embeddings)
└── train.py     # Full pipeline: preprocess → train → FAISS → eval → HF upload
```

### Chạy training (cần GPU hoạt động)
```bash
cd /workspace/LM_Submodular_RL

# Quick test: 500k records, 2 epochs (~20 phút trên RTX 3090)
/venv/main/bin/python3 -m retrieval_models.sasrec.train \
    --max_train 500000 --epochs 2 --batch_size 2048

# Full run: tất cả 11.79M records, 5 epochs (~2-3 giờ)
/venv/main/bin/python3 -m retrieval_models.sasrec.train \
    --epochs 5 --batch_size 2048

# Lần sau (skip preprocessing cache)
/venv/main/bin/python3 -m retrieval_models.sasrec.train \
    --skip_preprocess --epochs 5 --batch_size 2048
```

### Outputs sau training
```
retrieval_models/sasrec/output/
├── best_model.pt           # SASRec weights
├── item_faiss.index        # FAISS index (2.6M items × 64 dim ≈ 640 MB)
├── config.json             # Model config
├── results.json            # Recall@K vs BM25 baseline
└── preprocessed/
    ├── asin2iid.json       # ASIN → item_id map (cache)
    ├── train_seqs_L20.npy  # Preprocessed sequences (cache)
    └── train_targets.npy   # Target ids (cache)
```

### Mục tiêu recall
| | BM25 (hiện tại) | SASRec (kỳ vọng) |
|---|---|---|
| Recall@10 | 1.5% | 10-20% |
| Recall@50 | 3.0% | 20-35% |
| Recall@100 | 4.2% | 30-45% |
| Recall@200 | 5.2% | 40-55% |

---

## 5. Hướng đi tiếp theo (sau khi SASRec xong)

### 5.1 Nếu SASRec recall tốt hơn BM25 → Tích hợp vào pipeline

Thay BM25 trong `retrieval/unified_pipeline.py`:
```python
# Thay đoạn này:
candidates = bm25.search(query, top_k=100)

# Bằng:
from retrieval_models.sasrec.retriever import SASRecRetriever
sasrec_retriever = SASRecRetriever.load("retrieval_models/sasrec/output/")
candidates = sasrec_retriever.search(history_ids, top_k=200)
```

Sau đó retrain RL + Submodular với recall cao hơn → kỳ vọng Hit@10 tăng từ 1% lên 5-15%.

### 5.2 Nếu SASRec vẫn chưa đủ → Hybrid BM25 + SASRec

```
SASRec(history_ids)    → 200 candidates
BM25(last_3_titles)    → 100 candidates
RRF fusion             → top-150 (score = Σ 1/(60+rank))
```
Recall@200 hybrid thường > cả hai riêng lẻ.

### 5.3 Tiếp theo: BERT4Rec hoặc TiSASRec

Sau SASRec, thử theo thứ tự:

**BERT4Rec** (CIKM 2019):
- Bidirectional Transformer + Cloze objective (mask random items trong sequence)
- Mạnh hơn SASRec vì nhìn cả trái lẫn phải
- Repo: https://github.com/FeiSun/BERT4Rec
- Branch đề xuất: `bert4rec-backbone`

**TiSASRec** (WSDM 2020):
- SASRec + time-aware attention (thêm time interval giữa các mua hàng)
- Quan trọng cho Amazon vì user behavior có seasonal patterns
- Repo: https://github.com/JiachengLi1995/TiSASRec
- Branch đề xuất: `tisasrec-backbone`

**LightGCN** (SIGIR 2020):
- Graph CF: user-item purchase graph → latent factors → ANN retrieval
- Không cần sequence — học từ co-purchase patterns
- Tốt cho cold items (ít lịch sử)
- Branch đề xuất: `lightgcn-backbone`

### 5.4 Dài hạn: DPO Fine-tune Qwen3-Reranker

Data đã được tạo sẵn (`output_amazon/dpo_pairs/`):
- Train: 263,876 pairs
- Format: `{query, chosen: target_item, rejected: [also_view_items + random]}`

```bash
/venv/main/bin/python3 -m training.dpo_trainer \
    --data_path /workspace/amazon/output_amazon/dpo_pairs/train.jsonl \
    --model_id  Qwen/Qwen3-Reranker-0.6B \
    --method    orpo \
    --output_dir /workspace/amazon/output_amazon/reranker_finetuned \
    --device    cuda
```
Kỳ vọng: Hit@10 tăng thêm 2-4% sau DPO fine-tuning.

### 5.5 Workflow chuẩn cho mỗi approach mới

```
1. git checkout -b <approach-name>-backbone
2. Implement retrieval_models/<approach>/model.py + train.py
3. Train → lưu results.json
4. git add + git commit + git push
5. huggingface-cli login → upload weights lên chuannb/<approach>-amazon-retrieval
6. So sánh recall@K với BM25 và SASRec
7. Nếu tốt hơn → tích hợp vào unified_pipeline.py
```

---

## 6. Setup Server Mới (QUAN TRỌNG)

### Requirements
```bash
# GPU: RTX 3090 hoặc tốt hơn, ≥24GB VRAM
# CUDA: 13.0 (torch 2.11.0+cu130)
# RAM: ≥32GB (preprocessing 11.79M records)
# Disk: ≥100GB (data + model weights + FAISS index)
```

### Docker command đúng cách
```bash
docker run --gpus all \
           --runtime=nvidia \
           -v /path/to/data:/workspace/amazon \
           -v /path/to/code:/workspace/LM_Submodular_RL \
           --shm-size=8g \
           -it your-image bash
```

### Verify CUDA sau khi setup
```bash
/venv/main/bin/python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
x = torch.ones(3,3).cuda()
print('Tensor test OK:', x.device)
"
```

### Install dependencies (nếu cần)
```bash
pip install torch==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install faiss-gpu  # hoặc faiss-cpu nếu không cần GPU FAISS
pip install huggingface_hub tqdm numpy
```

### Data cần có (copy từ server cũ)
```
/workspace/amazon/output_amazon/products.jsonl   # 2.6M sản phẩm
/workspace/amazon/dataset/train.jsonl            # 11.79M records
/workspace/amazon/dataset/val.jsonl
/workspace/amazon/dataset/test.jsonl
/workspace/amazon/output_amazon/bm25_index.pkl  # optional, cho compare
```

### Reproduce kết quả baseline BM25
```bash
cd /workspace/LM_Submodular_RL
/venv/main/bin/python3 check_bm25_recall.py --sample 2000 --topk 10 50 100 200
# Expected: recall@200 ≈ 5.2%
```

---

## 7. GitHub & HuggingFace

- **GitHub:** https://github.com/chuannb/LM_Submodular_RL
  - `main`: code gốc pipeline
  - `30_04`: version đã chạy, có kết quả Hit@10=1%
  - `sasrec-backbone`: SASRec retrieval (branch hiện tại)
  
- **HuggingFace:** `chuannb/`
  - `chuannb/sasrec-amazon-retrieval` ← weights SASRec sau khi train

### Push code lên GitHub
```bash
cd /workspace/LM_Submodular_RL
git remote set-url origin https://YOUR_GITHUB_TOKEN@github.com/chuannb/LM_Submodular_RL.git
git push -u origin sasrec-backbone
```

### Upload weights lên HuggingFace
```bash
huggingface-cli login   # nhập HF token
# Sau khi train xong, train.py tự upload. Hoặc manual:
/venv/main/bin/python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='retrieval_models/sasrec/output',
    repo_id='chuannb/sasrec-amazon-retrieval',
    repo_type='model',
    ignore_patterns=['preprocessed/'],
)
"
```

---

## 8. Key Insights (để không làm lại)

1. **BM25 recall thấp vì sai query** — ghép title của 3 items cuối không đại diện cho purchase intent. Text similarity ≠ purchase prediction.

2. **inject_target=True là hack** — training thêm target vào candidates để có reward, nhưng eval không dùng → Hit@10 tụt về 1%.

3. **Recall@Stage-1 quyết định tất cả** — nếu target không trong top-200 candidates → reward = 0 → RL không học. Fix recall trước, rồi mới tune RL/Submodular sau.

4. **SASRec là cách làm đúng** — học embeddings từ actual purchase sequences. Recall@200 dự kiến 40-55% vs BM25 5.2%.

5. **Submodular + RL là phần novel** — không paper nào kết hợp LLM reranker + RL-controlled α_t + budget constraint. Phần này đúng về concept, chỉ bị bottleneck bởi recall thấp.

6. **Workflow mỗi approach:**  
   `implement → train → save results → git push → HF upload → compare`

---

*Last updated: 2026-05-13 | Branch: sasrec-backbone*
