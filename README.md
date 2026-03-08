# LM Submodular RL — Product Recommendation Pipeline

Hệ thống gợi ý sản phẩm kết hợp 3 tầng:

1. **Retrieval** — BM25 + Qwen3-Embedding-0.6B tìm candidates
2. **Reranking** — Qwen3-Reranker-0.6B + DPO/ORPO fine-tuning từ interaction logs
3. **Slate Optimization** — Submodular function + RL policy tối ưu danh sách k sản phẩm theo relevance, diversity và budget

---

## Cấu trúc thư mục

```
LM_Submodular_RL/
├── run_amazon.py              # Entry point chính — chạy toàn bộ pipeline
├── serve.py                   # FastAPI serving endpoint
├── offline.py                 # Offline training từ logs
│
├── data/
│   └── amazon_loader.py       # Load Amazon reviews + metadata, build DPO pairs
│
├── retrieval/
│   ├── bm25_retriever.py      # BM25 first-stage retrieval (rank-bm25)
│   ├── embedding_retriever.py # Dense retrieval với Qwen3-Embedding-0.6B + FAISS
│   ├── reranker.py            # Qwen3-Reranker-0.6B cross-encoder
│   └── unified_pipeline.py   # BM25 + Dense → RRF Fusion → Reranker → Submodular → RL
│
├── models/
│   ├── submodular.py          # Submodular diversity function f(S) = α*rel + (1-α)*div
│   └── rl_policy.py           # Actor-critic policy π(s) → (α, κ)
│
├── algorithms/
│   ├── greedy_selector.py     # Budgeted submodular greedy
│   ├── trajectory_builder.py  # Build training trajectories từ dataset
│   └── unified_trainer.py     # Joint trainer: RL + submodular
│
├── training/
│   ├── dpo_trainer.py         # DPO/ORPO fine-tuning cho reranker
│   └── reranker_trainer.py    # Pairwise reranker training
│
├── interaction/
│   ├── logger.py              # Log click/cart/purchase events vào SQLite
│   └── preference_converter.py # Convert logs → DPO pairs
│
└── utils/
    ├── encoders.py            # StateEncoder — encode user history thành vector
    └── metrics.py             # Hit@k, NDCG@k, MRR@k, Coverage, ILD
```

---

## Data cần có

```
/workspace/All_Amazon_Review_5_10M_filtered.json   # Reviews (JSONL, ~10M dòng)
/workspace/All_Amazon_Meta_in_Review.json          # Metadata sản phẩm (JSONL)
```

**Format review** (mỗi dòng 1 JSON):
```json
{"reviewerID": "A1X...", "asin": "B017O9P72A", "overall": 4.0,
 "reviewText": "...", "summary": "...", "unixReviewTime": 1514592000}
```

**Format metadata** (mỗi dòng 1 JSON):
```json
{"asin": "B017O9P72A", "title": "LIFX Smart Bulb", "price": "$49.99",
 "brand": "LIFX", "category": ["Electronics", "Smart Home"],
 "description": ["..."], "also_buy": ["B001...", "B002..."],
 "also_view": ["B003...", "B004..."]}
```

---

## Cài đặt

```bash
cd /workspace/LM_Submodular_RL
pip install -r requirements.txt
```

**requirements.txt** bao gồm: `transformers`, `peft`, `trl`, `faiss-cpu`, `rank-bm25`, `sentence-transformers`, `fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `tqdm`.

---

## Chạy pipeline chính

### Smoke test (nhanh, CPU, ~5 phút)

```bash
cd /workspace/LM_Submodular_RL
python run_amazon.py \
    --max_users       200 \
    --epochs          1 \
    --steps_per_epoch 50 \
    --eval_steps      50 \
    --slate_size      10 \
    --device          gpu
```

### Full run (khuyến nghị, GPU)

```bash
python run_amazon.py \
    --max_users       2000 \
    --epochs          3 \
    --steps_per_epoch 200 \
    --slate_size      10 \
    --device          cuda
```

### Full run với Dense Retrieval (Qwen3-Embedding, tốt nhất)

```bash
python run_amazon.py \
    --max_users       2000 \
    --epochs          3 \
    --steps_per_epoch 200 \
    --slate_size      10 \
    --build_dense \
    --device          cuda
```

---

## Các bước pipeline khi chạy `run_amazon.py`

| Bước | Mô tả | Output |
|------|--------|--------|
| 1 | Load reviews (streaming, lấy `max_users` users có ≥5 reviews) | DataFrame |
| 2 | Scan full meta file tìm ASINs khớp → build product catalog | `products.jsonl`, `id_map.json` |
| 3 | Build BM25 index (in-memory, rank-bm25) | `bm25_index.pkl` |
| 4 | *(nếu `--build_dense`)* Build FAISS dense index với Qwen3-Embedding | `dense_index.pkl` |
| 5 | Load Qwen3-Reranker-0.6B | — |
| 6 | Khởi tạo models: Submodular + StateEncoder + RL Policy | — |
| 7 | Build train/val/test splits (leave-last-2-out) | TrajectoryStep lists |
| 7b | Lưu split stats + samples để inspect | `split_inspection/` |
| 7b-2 | Log co-purchase / co-view graph stats | stdout |
| 7c | Build DPO preference pairs (hard negatives từ `also_view`) | `dpo_pairs/*.jsonl` |
| 8 | Train: RL actor-critic + submodular diversity params | `best_unified.pt` |
| 9 | Evaluate trên test set với best checkpoint | — |
| 10 | In metrics + lưu kết quả | `results.json` |

---

## Split strategy

Dùng **leave-last-2-out** theo thứ tự thời gian của từng user:

```
User có n interactions [i0, i1, ..., i_{n-1}]:
  train : t = 1 .. n-3   (rolling window, nhiều samples)
  val   : t = n-2         (1 sample/user)
  test  : t = n-1         (1 sample/user — interaction cuối cùng)
```

Với `min_interactions=5`:
- User 5 reviews → train: 2 steps, val: 1, test: 1
- Cùng user xuất hiện ở cả 3 splits (chuẩn với sequential recommendation — không phải data leak)

---

## DPO pairs (cho fine-tuning reranker)

Output tại `output_amazon/dpo_pairs/{train,val,test}.jsonl`:

```json
{
  "user_id": 42,
  "query": "smart home bulb alexa compatible",
  "chosen":  {"asin": "B017...", "title": "LIFX Smart Bulb", "price": 49.99},
  "rejected": [
    {"asin": "B003...", "title": "Philips Hue", "neg_source": "also_view"},
    {"asin": "B004...", "title": "...",          "neg_source": "also_view"},
    {"asin": "B901...", "title": "...",          "neg_source": "random"},
    {"asin": "B902...", "title": "...",          "neg_source": "random"}
  ],
  "history_ids": [17, 23, 36],
  "copurchase_ids": [55, 78, 102]
}
```

- `also_view` negatives = **hard negatives** (sản phẩm liên quan nhưng không mua)
- `copurchase_ids` = `also_buy` items của target (có thể dùng làm weak positives cho RL)

---

## All arguments

### Data
| Arg | Default | Mô tả |
|-----|---------|--------|
| `--review_path` | `/workspace/All_Amazon_Review_5_10M_filtered.json` | File review JSONL |
| `--meta_path` | `/workspace/All_Amazon_Meta_in_Review.json` | File metadata JSONL |
| `--max_users` | `2000` | Số users load (streaming, dừng sớm) |
| `--max_items` | `50000` | Giới hạn scan meta khi không có `keep_asins` |
| `--history_length` | `10` | Số items lịch sử tối đa encode làm state |

### Retrieval
| Arg | Default | Mô tả |
|-----|---------|--------|
| `--build_dense` | `False` | Bật dense index Qwen3-Embedding-0.6B |
| `--n_bm25` | `100` | BM25 recall size |
| `--n_dense` | `50` | Dense recall size |
| `--n_fuse` | `30` | Candidates sau RRF fusion → reranker |
| `--embed_batch_size` | `16` | Batch size khi embed |
| `--reranker_batch_size` | `8` | Batch size reranker |
| `--slate_size` | `10` | Final slate size k |

### Training
| Arg | Default | Mô tả |
|-----|---------|--------|
| `--epochs` | `3` | Số epoch |
| `--steps_per_epoch` | `200` | Steps mỗi epoch |
| `--eval_steps` | `None` | Giới hạn eval steps (None = all) |
| `--batch_size` | `32` | RL replay batch size |
| `--buffer_size` | `10000` | Replay buffer size |
| `--min_buffer` | `64` | Tối thiểu transitions trước khi train |
| `--lr_rl` | `3e-4` | Learning rate RL policy |
| `--lr_sub` | `1e-3` | Learning rate submodular params |
| `--lr_encoder` | `1e-3` | Learning rate state encoder |
| `--gamma` | `0.99` | RL discount factor |
| `--lambda_sub` | `0.5` | Submodular loss weight |
| `--lambda_rank` | `0.1` | Diversity ranking loss weight |
| `--alpha_init` | `0.7` | Khởi tạo relevance weight α |
| `--device` | `cpu` | `cpu` hoặc `cuda` |
| `--seed` | `42` | Random seed |
| `--output_dir` | `output_amazon` | Thư mục lưu checkpoints và indexes |

---

## Metrics

| Metric | Ý nghĩa |
|--------|---------|
| **Hit@k** | Tỷ lệ target item xuất hiện trong slate k |
| **NDCG@k** | Normalized Discounted Cumulative Gain — thưởng rank cao |
| **MRR@k** | Mean Reciprocal Rank — vị trí trung bình của hit đầu tiên |
| **Coverage** | % catalog items được gợi ý ít nhất 1 lần |
| **ILD** | Intra-List Diversity — độ đa dạng trong slate |

---

## Output files

```
output_amazon/
├── products.jsonl              # Product catalog đã export
├── id_map.json                 # ASIN → item_id mapping
├── bm25_index.pkl              # BM25 index (rank-bm25)
├── dense_index.pkl             # FAISS dense index (nếu --build_dense)
├── best_unified.pt             # Best checkpoint (submodular + encoder + RL)
├── results.json                # Final test metrics
├── split_inspection/
│   ├── stats.json              # Thống kê 3 splits
│   ├── train_samples.json      # 50 mẫu train (có title, history, budget)
│   ├── val_samples.json
│   └── test_samples.json
└── dpo_pairs/
    ├── train.jsonl             # DPO training pairs
    ├── val.jsonl
    └── test.jsonl
```

---

## Serving (sau khi train xong)

```bash
cd /workspace/LM_Submodular_RL
DEVICE=cuda \
BM25_INDEX_PATH=output_amazon/bm25_index.pkl \
MODEL_CKPT=output_amazon/best_unified.pt \
SLATE_SIZE=10 \
uvicorn serve:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
- `POST /search` — `{"query": "smart bulb alexa", "user_id": "U123", "slate_size": 10}`
- `POST /interact` — `{"session_id": "...", "item_id": "B017...", "event": "click"}`
- `GET /session/new` — tạo session mới
- `GET /stats` — interaction statistics

---

## Fine-tune Reranker với DPO (sau khi có DPO pairs)

```bash
python -m training.dpo_trainer \
    --data_path output_amazon/dpo_pairs/train.jsonl \
    --model_id  Qwen/Qwen3-Reranker-0.6B \
    --method    orpo \
    --output_dir output_amazon/reranker_finetuned \
    --device    cuda
```
