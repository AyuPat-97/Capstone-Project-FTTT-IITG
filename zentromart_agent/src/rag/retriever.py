import csv
import pickle
import re
import sys
from pathlib import Path
from typing import Any, List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    DATA_DIR,
    RAG_CACHE,
    RETRIEVAL_TOP_K,
    RRF_K,
    EMBEDDING_MODEL,
    EMBEDDING_FALLBACK,
)

def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize words and hyphenated codes cleanly for BM25."""
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", text) if t]

class HybridRetriever:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.bm25 = None
        self.faiss_index = None
        self.model = None
        self.model_name = None
        self.product_map: Dict[str, str] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        if not (RAG_CACHE / "chunks.pkl").exists():
            # Trigger ingestion if cache doesn't exist
            from src.rag.ingest import ingest
            ingest()

        with open(RAG_CACHE / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        with open(RAG_CACHE / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        self.faiss_index = faiss.read_index(str(RAG_CACHE / "faiss.index"))

        # Load embedding model matching meta
        meta_path = RAG_CACHE / "meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.model_name = meta.get("model_name", EMBEDDING_MODEL)
        else:
            self.model_name = EMBEDDING_MODEL

        cache_folder = Path.home() / ".cache" / "zentromart" / "bge-m3"
        try:
            self.model = SentenceTransformer(self.model_name, cache_folder=str(cache_folder))
        except Exception:
            fallback_folder = Path.home() / ".cache" / "zentromart" / "bge-small"
            self.model = SentenceTransformer(EMBEDDING_FALLBACK, cache_folder=str(fallback_folder))

        # Build product title to SKU map for guaranteed catalog recall
        prod_path = DATA_DIR / "products.csv"
        if prod_path.exists():
            with open(prod_path, mode="r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sku = row.get("sku", "").strip()
                    title = row.get("title", "").strip()
                    if sku and title:
                        self.product_map[title.lower()] = sku.upper()

        self._loaded = True

    def retrieve(
        self,
        query: str,
        session: Optional[Any] = None,
        top_k: int = RETRIEVAL_TOP_K
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        if not query or not query.strip():
            return []

        # Determine order date if session is present
        order_date_str = None
        if session and hasattr(session, "order_data") and session.order_data:
            order_date_str = session.order_data.get("delivered_on") or session.order_data.get("placed_at")
            if order_date_str and "T" in order_date_str:
                order_date_str = order_date_str.split("T")[0]

        # Extract date from query if no session order date (e.g. for past policy queries like PQ0019)
        query_date_str = None
        if not order_date_str and query:
            date_match = re.search(r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b", query, re.I)
            if date_match:
                m_names = {
                    "january": "01", "february": "02", "march": "03", "april": "04",
                    "may": "05", "june": "06", "july": "07", "august": "08",
                    "september": "09", "october": "10", "november": "11", "december": "12"
                }
                d = f"{int(date_match.group(1)):02d}"
                m = m_names[date_match.group(2).lower()]
                query_date_str = f"2026-{m}-{d}"

        # 1. BM25 Sparse Search
        q_tokens = tokenize_for_bm25(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        sparse_ranking = np.argsort(bm25_scores)[::-1]  # descending indices

        # 2. Dense FAISS Search
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        n_candidates = min(len(self.chunks), top_k * 4)
        dense_distances, dense_indices = self.faiss_index.search(q_emb, n_candidates)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[int, float] = {}

        # Add dense ranks
        for rank, idx in enumerate(dense_indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_K + rank + 1))

        # Add sparse ranks
        for rank, idx in enumerate(sparse_ranking[:n_candidates]):
            if bm25_scores[idx] <= 0:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (RRF_K + rank + 1))

        # Boost / directly pull product pseudo-chunks by SKU or title
        found_skus = set()
        for sku_match in re.findall(r"\b([A-Za-z]{2,4}-\d{3,4})\b", query):
            found_skus.add(sku_match.upper())

        q_lower = query.lower()
        for title_lower, sku in self.product_map.items():
            if title_lower in q_lower:
                found_skus.add(sku)

        for sku in found_skus:
            target_sid = f"products.csv:{sku}"
            for idx, c in enumerate(self.chunks):
                if c.get("section_id") == target_sid:
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 10.0

        effective_date_for_version = order_date_str or query_date_str

        # 4. Filter, Deduplicate, and Rank
        scored_candidates = []
        for idx, score in rrf_scores.items():
            chunk = self.chunks[idx]
            sec_id = chunk["section_id"]
            is_superseded = chunk.get("superseded", False)
            effective_from = chunk.get("effective_from", "2026-01-01")

            # Version filter:
            # If chunk is superseded, filter out unless order_date satisfies:
            # effective_from <= order_date < successor_effective_from (2026-08-01 for returns v3)
            if is_superseded:
                if not effective_date_for_version:
                    continue
                # For v3 returns: effective 2026-01-01, replaced by v4 on 2026-08-01
                if not (effective_from <= effective_date_for_version < "2026-08-01"):
                    continue
            else:
                # If current chunk was not yet in effect for this older order
                if order_date_str and effective_from > order_date_str and chunk.get("doc_id") == "POL-RET":
                    continue

            scored_candidates.append({
                "chunk_idx": idx,
                "section_id": sec_id,
                "text": chunk["text"],
                "doc_type": chunk.get("doc_type", "help"),
                "doc_priority": chunk.get("doc_priority", 3),
                "superseded": is_superseded,
                "effective_from": effective_from,
                "score": score,
            })

        # Deduplicate by section_id (keep highest score)
        deduped: Dict[str, Dict[str, Any]] = {}
        for item in scored_candidates:
            sid = item["section_id"]
            if sid not in deduped or item["score"] > deduped[sid]["score"]:
                deduped[sid] = item

        # Sort: score desc, doc_priority asc (lower number = higher priority), effective_from desc
        results = list(deduped.values())
        results.sort(key=lambda x: (-x["score"], x["doc_priority"], -int(x["effective_from"].replace("-", ""))))

        # Cap product pseudo-chunks so they don't crowd out essential policy sections (e.g. WAR-2, RET-2)
        final_chunks = []
        product_count = 0
        max_products = max(1, len(found_skus)) if found_skus else 2
        for r in results:
            is_prod = (r["doc_type"] == "product")
            if is_prod and product_count >= max_products:
                continue
            final_chunks.append({
                "section_id": r["section_id"],
                "text": r["text"],
                "doc_type": r["doc_type"],
                "doc_priority": r["doc_priority"],
                "superseded": r["superseded"],
                "score": r["score"],
            })
            if is_prod:
                product_count += 1
            if len(final_chunks) >= top_k:
                break

        return final_chunks

# Global retriever instance
retriever = HybridRetriever()

def retrieve(query: str, session: Optional[Any] = None, top_k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
    return retriever.retrieve(query, session=session, top_k=top_k)
