"""
Demo: BM25 Retriever — cách hoạt động từng bước

Chạy:
  cd /workspace/LM_Submodular_RL
  python demo_bm25.py
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

from retrieval.bm25_retriever import BM25Retriever, SearchResult, _tokenize, _build_doc_text


INDEX_PATH   = "/workspace/amazon/output_amazon/bm25_index.pkl"
PRODUCTS_PATH = "/workspace/amazon/output_amazon/products.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
def separator(title: str = "") -> None:
    print("\n" + "═" * 60)
    if title:
        print(f"  {title}")
        print("─" * 60)


def show_results(results: List[SearchResult], top: int = 5) -> None:
    for i, r in enumerate(results[:top], 1):
        print(f"  [{i}] score={r.score:6.3f}  id={r.item_id:>7}  {r.title[:55]}")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1: BM25 hoạt động như thế nào — giải thích nhanh
# ─────────────────────────────────────────────────────────────────────────────
separator("PHẦN 1 — BM25 là gì?")
print("""
BM25 (Best Match 25) là thuật toán tìm kiếm dựa trên TF-IDF cải tiến.

  score(q, d) = Σ_{term t in q}  IDF(t) × TF(t, d) × (k1 + 1)
                                            ─────────────────────
                                            TF(t,d) + k1(1 - b + b·|d|/avgdl)

  • TF(t, d)  : term frequency của t trong doc d
  • IDF(t)    : log((N - n_t + 0.5) / (n_t + 0.5))  — từ nào hiếm → điểm cao
  • |d|/avgdl : chuẩn hoá độ dài document (doc ngắn được ưu tiên)
  • k1 = 1.5, b = 0.75  (hằng số mặc định BM25 Okapi)

Kết quả: từ khoá khớp với nhiều terms hiếm → score cao hơn.
""")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2: Text tokenisation — bước đầu tiên
# ─────────────────────────────────────────────────────────────────────────────
separator("PHẦN 2 — Tokenisation")

sample_queries = [
    "Wireless Bluetooth Headphone",
    "Women's Running Shoes Size 8",
    "Harry Potter Book Set",
]

for q in sample_queries:
    tokens = _tokenize(q)
    print(f"  '{q}'")
    print(f"    → {tokens}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3: Document text — cách product được index
# ─────────────────────────────────────────────────────────────────────────────
separator("PHẦN 3 — Product → Searchable Text")

sample_products = [
    {
        "item_id": 1,
        "title": "Sony WH-1000XM5 Wireless Headphone",
        "brand": "Sony",
        "description": "Industry-leading noise cancelling headphones",
        "categories": "Electronics",
        "feature": "30-hour battery life, multipoint connection",
        "price": 349.99,
    },
    {
        "item_id": 2,
        "title": "Nike Air Zoom Pegasus 40 Women",
        "brand": "Nike",
        "description": "Running shoe with React foam cushioning",
        "categories": "Sports & Outdoors",
        "feature": "Lightweight mesh upper, size 6-12",
        "price": 130.00,
    },
]

for p in sample_products:
    text = _build_doc_text(p)
    tokens = _tokenize(text)
    print(f"  Product: '{p['title']}'")
    print(f"  Indexed text: '{text[:80]}...'")
    print(f"  Tokens ({len(tokens)}): {tokens[:12]} ...\n")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4: Build index từ scratch — minh hoạ trực tiếp
# ─────────────────────────────────────────────────────────────────────────────
separator("PHẦN 4 — Build Index Nhỏ Từ Scratch")

mini_products = [
    {"item_id": 0, "title": "Sony WH-1000XM5 Wireless Headphone",
     "brand": "Sony", "description": "Noise cancelling wireless headphone 30h battery",
     "categories": "Electronics", "feature": "Multipoint connection", "price": 349.99},
    {"item_id": 1, "title": "Apple AirPods Pro 2nd Generation",
     "brand": "Apple", "description": "Wireless earbuds active noise cancellation",
     "categories": "Electronics", "feature": "Spatial audio MagSafe charging", "price": 249.00},
    {"item_id": 2, "title": "Nike Air Zoom Pegasus 40 Running Shoes",
     "brand": "Nike", "description": "Road running shoes React foam cushioning",
     "categories": "Sports", "feature": "Lightweight breathable mesh", "price": 130.00},
    {"item_id": 3, "title": "Adidas Ultraboost 23 Running Shoes",
     "brand": "Adidas", "description": "High performance running shoes Boost midsole",
     "categories": "Sports", "feature": "Primeknit upper continental rubber outsole", "price": 190.00},
    {"item_id": 4, "title": "Harry Potter Complete Book Series Box Set",
     "brand": "Scholastic", "description": "All 7 Harry Potter books by J.K. Rowling",
     "categories": "Books", "feature": "Hardcover collector edition", "price": 89.99},
    {"item_id": 5, "title": "Kindle Paperwhite E-Reader 16GB",
     "brand": "Amazon", "description": "E-reader 300ppi display 6.8 inch waterproof",
     "categories": "Electronics", "feature": "3 months Kindle Unlimited included", "price": 139.99},
    {"item_id": 6, "title": "JBL Charge 5 Portable Bluetooth Speaker",
     "brand": "JBL", "description": "Waterproof portable speaker 20h playtime",
     "categories": "Electronics", "feature": "IP67 waterproof powerbank function", "price": 179.95},
    {"item_id": 7, "title": "The Great Gatsby Paperback",
     "brand": "Scribner", "description": "Classic novel by F. Scott Fitzgerald",
     "categories": "Books", "feature": "Paperback 180 pages", "price": 15.00},
]

t0 = time.perf_counter()
mini_bm25 = BM25Retriever.build(mini_products, backend="bm25s")
build_time = (time.perf_counter() - t0) * 1000
print(f"  Built index over {len(mini_products)} products in {build_time:.1f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 5: Search trên mini index — xem scoring
# ─────────────────────────────────────────────────────────────────────────────
separator("PHẦN 5 — Search Trên Mini Index")

demo_queries = [
    ("wireless headphone",   "khớp Sony + Apple (đều wireless)"),
    ("noise cancelling",     "Sony cao hơn AirPods (từ 'noise cancelling' in description)"),
    ("running shoes nike",   "Nike lên đầu vì thêm brand term"),
    ("book",                 "Harry Potter + Great Gatsby"),
    ("bluetooth portable",   "JBL speaker + AirPods"),
    ("sony apple",           "brand matching thuần túy"),
]

for query, explanation in demo_queries:
    results = mini_bm25.search(query, top_k=3)
    print(f"\n  Query: '{query}'  ← {explanation}")
    for i, r in enumerate(results, 1):
        bar = "█" * int(r.score * 3)
        print(f"    [{i}] {r.score:5.2f} {bar:<15}  {r.title[:50]}")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 6: IDF — tại sao từ hiếm quan trọng hơn?
# ─────────────────────────────────────────────────────────────────────────────
separator("PHẦN 6 — IDF: Từ Hiếm Quan Trọng Hơn")

print("""
  Corpus nhỏ (8 docs):
    "wireless" xuất hiện ở 3 docs → IDF thấp  (phổ biến)
    "noise"    xuất hiện ở 2 docs → IDF cao hơn
    "gatsby"   xuất hiện ở 1 doc  → IDF cao nhất (đặc thù)
""")

# "electronics" vs "gatsby" — specific term wins
r1 = mini_bm25.search("electronics", top_k=3)
r2 = mini_bm25.search("gatsby", top_k=3)

print("  search('electronics')  — term phổ biến (nhiều doc):")
for r in r1[:3]:
    print(f"    score={r.score:.3f}  {r.title[:50]}")

print("\n  search('gatsby')       — term hiếm (1 doc):")
for r in r2[:3]:
    print(f"    score={r.score:.3f}  {r.title[:50]}")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 7: top_k và ảnh hưởng
# ─────────────────────────────────────────────────────────────────────────────
separator("PHẦN 7 — Ảnh Hưởng Của top_k")

query = "bluetooth"
for k in [1, 3, 5, 8]:
    results = mini_bm25.search(query, top_k=k)
    print(f"  top_k={k}: {len(results)} kết quả | "
          f"score range [{results[-1].score:.2f}, {results[0].score:.2f}]")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 8: Load index thật — Amazon 2.6M sản phẩm
# ─────────────────────────────────────────────────────────────────────────────
separator("PHẦN 8 — Load Amazon Index Thật (2.6M Products)")

print(f"  Loading {INDEX_PATH} ...")
t0 = time.perf_counter()
real_bm25 = BM25Retriever.load(INDEX_PATH)
load_time = (time.perf_counter() - t0) * 1000
n = len(real_bm25._backend.products)
print(f"  Loaded {n:,} products in {load_time:.0f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 9: Search trên corpus thật + timing
# ─────────────────────────────────────────────────────────────────────────────
separator("PHẦN 9 — Search Thật + Timing")

real_queries = [
    "wireless bluetooth headphone noise cancelling",
    "women running shoes size 8",
    "harry potter book set hardcover",
    "laptop stand ergonomic adjustable",
    "coffee maker automatic programmable",
]

for query in real_queries:
    t0 = time.perf_counter()
    results = real_bm25.search(query, top_k=100)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n  Query: '{query}'  ({elapsed:.1f}ms, {len(results)} results)")
    show_results(results, top=3)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 10: Cách pipeline dùng BM25 — query từ history titles
# ─────────────────────────────────────────────────────────────────────────────
separator("PHẦN 10 — Cách Pipeline Thật Dùng BM25")

print("""
  Trong run_amazon.py, query KHÔNG phải do user gõ.
  Nó được tạo tự động từ tên 3 sản phẩm cuối cùng trong history:

    def make_query(step) -> str:
        titles = [title_map.get(i, "") for i in step.history_ids[-3:]]
        return " ".join(titles)

  Ví dụ: user vừa xem:
    1. "Sony WH-1000XM5 Headphone"
    2. "JBL Charge 5 Speaker"
    3. "Apple AirPods Pro"
  → query = "Sony WH-1000XM5 Headphone JBL Charge 5 Speaker Apple AirPods Pro"
  → BM25 tìm các sản phẩm audio liên quan
  → Top-100 candidates được đưa vào Reranker + Submodular → slate 10 items
""")

history_query = "Sony WH-1000XM5 Headphone JBL Charge 5 Speaker Apple AirPods Pro"
print(f"  Simulated history query:\n    '{history_query}'\n")
results = real_bm25.search(history_query, top_k=10)
print(f"  Top 10 BM25 candidates:")
show_results(results, top=10)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 11: Save / Load workflow
# ─────────────────────────────────────────────────────────────────────────────
separator("PHẦN 11 — Save / Load Index")

import tempfile, os
with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    tmp_path = f.name

mini_bm25.save(tmp_path)
size_kb = os.path.getsize(tmp_path) / 1024
print(f"  Saved mini index (8 products) → {tmp_path}")
print(f"  File size: {size_kb:.1f} KB")

loaded_bm25 = BM25Retriever.load(tmp_path)
r = loaded_bm25.search("headphone", top_k=2)
print(f"  Loaded back — search('headphone'): {[x.title for x in r]}")
os.unlink(tmp_path)

real_size_mb = os.path.getsize(INDEX_PATH) / 1024 / 1024
print(f"\n  Amazon index (2.6M products): {real_size_mb:.0f} MB")


# ─────────────────────────────────────────────────────────────────────────────
separator("TỔNG KẾT")
print("""
  BM25Retriever flow:
  ┌─────────────────────────────────────────────────────────────┐
  │  Build time:                                                 │
  │    products (List[dict])                                     │
  │        → _build_doc_text()  [title+brand+desc+cats+feature] │
  │        → _tokenize()        [lowercase, strip punct]        │
  │        → BM25.index()       [IDF table + term freqs]        │
  │        → save as .pkl                                        │
  │                                                              │
  │  Query time:                                                 │
  │    query string                                              │
  │        → _tokenize()                                        │
  │        → BM25.retrieve()   [score = Σ IDF × TF_normalized]  │
  │        → top-k SearchResult(item_id, score, title, text)    │
  └─────────────────────────────────────────────────────────────┘
""")
