import glob
import os
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Add root directory to path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    KB_DIR,
    DATA_DIR,
    RAG_CACHE,
    EMBEDDING_MODEL,
    EMBEDDING_FALLBACK,
)

DOC_PRIORITY_MAP = {
    "meta": 0,
    "policy": 1,
    "agent_guide": 2,
    "product": 2,
    "help": 3,
    "faq": 4,
    "marketing": 5,
}

def load_embedding_model():
    """Try loading BAAI/bge-m3; fall back to bge-small-en-v1.5 if needed."""
    cache_dir = Path.home() / ".cache" / "zentromart" / "bge-m3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Loading primary embedding model: {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(cache_dir))
        return model, EMBEDDING_MODEL
    except Exception as e:
        print(f"Failed to load {EMBEDDING_MODEL}: {e}")
        print(f"Falling back to {EMBEDDING_FALLBACK}...")
        fallback_cache = Path.home() / ".cache" / "zentromart" / "bge-small"
        fallback_cache.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(EMBEDDING_FALLBACK, cache_folder=str(fallback_cache))
        return model, EMBEDDING_FALLBACK

def build_kb_chunks() -> list[dict]:
    """Parse all markdown files in kb/."""
    pattern = re.compile(r'^(##\s+\[([A-Za-z0-9_.\-]+)\][^\n]*)', re.MULTILINE)
    md_files = sorted(glob.glob(str(KB_DIR / "*.md")))
    chunks = []

    for fpath in md_files:
        with open(fpath, "r", encoding="utf-8") as handle:
            raw = handle.read()

        front_matter = {}
        body = raw
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                try:
                    front_matter = yaml.safe_load(parts[1]) or {}
                except Exception:
                    front_matter = {}
                body = parts[2]

        doc_id = front_matter.get("doc_id", os.path.basename(fpath))
        doc_type = front_matter.get("doc_type", "help")
        status = front_matter.get("status", "current")
        effective_from = str(front_matter.get("effective_from", "2026-01-01"))
        version = str(front_matter.get("version", "1.0"))
        superseded = (status == "superseded")
        doc_priority = DOC_PRIORITY_MAP.get(doc_type, 3)

        matches = list(pattern.finditer(body))
        for i, m in enumerate(matches):
            sec_id = m.group(2).strip()
            start_pos = m.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            chunk_text = body[start_pos:end_pos].strip()

            chunks.append({
                "section_id": sec_id,
                "text": chunk_text,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "status": status,
                "superseded": superseded,
                "effective_from": effective_from,
                "version": version,
                "doc_priority": doc_priority,
                "source_file": os.path.basename(fpath),
            })

    return chunks

def build_product_chunks() -> list[dict]:
    """Build pseudo-chunks from products.csv."""
    prod_path = DATA_DIR / "products.csv"
    if not prod_path.exists():
        print(f"Warning: {prod_path} does not exist.")
        return []

    df = pd.read_csv(prod_path, dtype=str)
    chunks = []
    for _, row in df.iterrows():
        sku = row.get("sku", "").strip()
        title = row.get("title", "").strip()
        cat = row.get("return_category", "").strip()
        price = row.get("price_inr", "0").strip()
        fulfilled = row.get("fulfilled_by", "").strip()
        assured = row.get("assured", "False").strip()
        warranty = row.get("warranty_months", "0").strip()

        chunk_text = (
            f"SKU {sku} : {title}. Category: {cat}. Price: ₹{price}.\n"
            f"Fulfilled by: {fulfilled}. Assured: {assured}. Warranty: {warranty} months."
        )

        chunks.append({
            "section_id": f"products.csv:{sku}",
            "text": chunk_text,
            "doc_id": "products.csv",
            "doc_type": "product",
            "status": "current",
            "superseded": False,
            "effective_from": "2026-01-01",
            "version": "1.0",
            "doc_priority": DOC_PRIORITY_MAP.get("product", 2),
            "source_file": "products.csv",
        })

    return chunks

def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize words and hyphenated codes cleanly for BM25."""
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", text) if t]

def ingest():
    """Build and save FAISS and BM25 indexes."""
    RAG_CACHE.mkdir(parents=True, exist_ok=True)
    kb_chunks = build_kb_chunks()
    product_chunks = build_product_chunks()
    all_chunks = kb_chunks + product_chunks

    print(f"Parsed {len(kb_chunks)} KB chunks and {len(product_chunks)} product chunks. Total: {len(all_chunks)} chunks.")

    # 1. BM25
    print("Building BM25 index...")
    tokenized_corpus = [tokenize_for_bm25(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(RAG_CACHE / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # 2. Dense embeddings
    model, model_name = load_embedding_model()
    chunk_texts = [c["text"] for c in all_chunks]
    print(f"Encoding {len(chunk_texts)} chunks using {model_name}...")
    embeddings = model.encode(chunk_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # 3. FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(RAG_CACHE / "faiss.index"))

    # 4. Save chunks and model metadata
    with open(RAG_CACHE / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    with open(RAG_CACHE / "meta.pkl", "wb") as f:
        pickle.dump({"model_name": model_name, "dim": dim, "count": len(all_chunks)}, f)

    md_count = len(glob.glob(str(KB_DIR / "*.md")))
    total_docs = md_count + (1 if product_chunks else 0)
    print(f"Indexed {len(all_chunks)} chunks from {total_docs} documents")

if __name__ == "__main__":
    ingest()
