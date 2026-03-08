"""
FastAPI serving endpoint  —  Unified Pipeline

Endpoints:
  POST /search           — query → submodular-optimised slate
  POST /interact         — log user action (click / add-to-cart / ...)
  GET  /session/new      — tạo session mới
  GET  /stats            — interaction statistics

Env vars:
  DEVICE              cpu | cuda
  DB_PATH             path to SQLite interaction DB
  BM25_INDEX_PATH     path to BM25 pickle index
  DENSE_INDEX_PATH    path to dense FAISS index (optional)
  MODEL_CKPT          path to unified model checkpoint (optional)
  SLATE_SIZE          default slate size k (default 10)

Chạy:
  DEVICE=cuda BM25_INDEX_PATH=bm25.pkl uvicorn serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import pickle
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from interaction.logger import EventType, InteractionLogger
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import Qwen3Reranker
from retrieval.unified_pipeline import UnifiedPipeline, UnifiedRLPolicy
from models.submodular import RerankerBackedSubmodular
from utils.encoders import StateEncoder

app = FastAPI(title="Product Search API — Unified Pipeline")

# ---------------------------------------------------------------------------
# Startup: load pipeline once
# ---------------------------------------------------------------------------

_pipeline: Optional[UnifiedPipeline] = None
_logger: Optional[InteractionLogger] = None


@app.on_event("startup")
async def startup():
    global _pipeline, _logger

    device_str = os.getenv("DEVICE", "cpu")
    device = torch.device(device_str)
    db_path = os.getenv("DB_PATH", "interactions.db")
    bm25_index = os.getenv("BM25_INDEX_PATH", "")
    dense_index = os.getenv("DENSE_INDEX_PATH", "")
    model_ckpt = os.getenv("MODEL_CKPT", "")
    slate_size = int(os.getenv("SLATE_SIZE", "10"))
    id_map_path = os.getenv("ID_MAP_PATH", "id_map.json")

    if not bm25_index or not os.path.exists(bm25_index):
        raise RuntimeError(
            "BM25 index not found. Run: python offline.py --action build_index first."
        )
    if not os.path.exists(id_map_path):
        raise RuntimeError(
            f"id_map.json not found at {id_map_path}. "
            "Run offline.py --action build_index to generate it."
        )

    with open(id_map_path) as f:
        id_map: dict = json.load(f)
    num_items = len(id_map)

    _logger = InteractionLogger(db_path)
    bm25 = BM25Retriever.load(bm25_index)
    reranker = Qwen3Reranker(device=device_str)

    dense = None
    if dense_index and os.path.exists(dense_index):
        from retrieval.embedding_retriever import EmbeddingRetriever
        dense = EmbeddingRetriever(device=device_str)
        dense.load_index(dense_index)

    embed_dim = 128
    submodular = RerankerBackedSubmodular(num_items=num_items, embed_dim=64).to(device)
    state_encoder = StateEncoder(num_items=num_items, embed_dim=embed_dim).to(device)
    rl_policy = UnifiedRLPolicy(state_dim=embed_dim).to(device)

    # Load checkpoint if available
    if model_ckpt and os.path.exists(model_ckpt):
        ckpt = torch.load(model_ckpt, map_location=device)
        submodular.load_state_dict(ckpt.get("submodular", {}), strict=False)
        state_encoder.load_state_dict(ckpt.get("state_encoder", {}), strict=False)
        rl_policy.actor.load_state_dict(ckpt.get("rl_actor", {}), strict=False)
        print(f"Loaded checkpoint from {model_ckpt}")

    _pipeline = UnifiedPipeline(
        bm25=bm25,
        reranker=reranker,
        submodular=submodular,
        rl_policy=rl_policy,
        state_encoder=state_encoder,
        id_map=id_map,
        logger=_logger,
        dense=dense,
        device=device,
        slate_size=slate_size,
    )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    session_id: str
    # User history (optional — richer personalisation)
    history_ids: List[int] = []
    history_extras: List[float] = []
    budget: Optional[float] = None
    page: int = 1


class SearchResponse(BaseModel):
    impression_id: str
    results: List[dict]   # {item_id, rel_score, submodular_score, title, position}


class InteractRequest(BaseModel):
    impression_id: str
    item_id: str
    event: str       # "click" | "add_to_cart" | "purchase" | "no_click" | "next_page"
    position: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/session/new")
async def new_session(user_id: Optional[str] = None):
    session_id = _logger.create_session(user_id)
    return {"session_id": session_id}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    results, impression_id = _pipeline.search(
        query=req.query,
        history_ids=req.history_ids or [],
        history_extras=req.history_extras or [],
        session_id=req.session_id,
        budget=req.budget,
        page=req.page,
        deterministic=True,
    )

    return SearchResponse(
        impression_id=impression_id or "",
        results=[
            {
                "item_id": r.item_id,
                "rel_score": round(r.rel_score, 4),
                "submodular_score": round(r.submodular_score, 4),
                "title": r.title,
                "position": r.slate_position,
            }
            for r in results
        ],
    )


@app.post("/interact")
async def interact(req: InteractRequest):
    if _logger is None:
        raise HTTPException(503, "Logger not ready")

    try:
        event = EventType(req.event)
    except ValueError:
        raise HTTPException(400, f"Unknown event type: {req.event}")

    _logger.log_interaction(req.impression_id, req.item_id, event, req.position)
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    if _logger is None:
        raise HTTPException(503, "Logger not ready")
    return _logger.stats()
