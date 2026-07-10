"""RAG ingestion script

Usage examples:

python rag_ingest.py --dataset openwebtext --limit 1000 --out_dir rag_index
python rag_ingest.py --source_file docs/mydoc.txt --out_dir rag_index

Produces:
- <out_dir>/index.faiss
- <out_dir>/meta.jsonl

Requires OPENAI_API_KEY in environment or .env
"""
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False
import faiss
from datasets import load_dataset
import openai
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"


def chunk_text(text, chunk_size=1000, chunk_overlap=200, model_name=None):
    """Token-aware chunking when tiktoken is available, otherwise char-based."""
    if not text:
        return []
    if TIKTOKEN_AVAILABLE:
        # Use cl100k_base as a default encoding; some models require different encodings
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        chunks = []
        start = 0
        n = len(toks)
        while start < n:
            end = min(start + chunk_size, n)
            chunk_toks = toks[start:end]
            chunk = enc.decode(chunk_toks)
            chunks.append(chunk)
            if end == n:
                break
            start = end - chunk_overlap
        return chunks
    else:
        # fallback to character-based splitting
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunk = text[start:end]
            chunks.append(chunk)
            if end == length:
                break
            start = end - chunk_overlap
        return chunks


def get_embedding(text):
    resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
    return np.array(resp["data"][0]["embedding"], dtype=np.float32)


def ingest_dataset(dataset_name=None, source_file=None, limit=1000, out_dir="rag_index"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    openai.api_key = api_key

    os.makedirs(out_dir, exist_ok=True)
    metas = []
    vectors = []

    if source_file:
        with open(source_file, "r", encoding="utf-8") as f:
            text = f.read()
        items = [(None, text)]
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)
        items = []
        for i, item in enumerate(ds):
            items.append((item.get("url") if isinstance(item, dict) else None, item.get("text") if isinstance(item, dict) else str(item)))
            if i + 1 >= limit:
                break

    idx = 0
    for src, text in tqdm(items, desc="Documents"):
        chunks = chunk_text(text)
        for chunk in chunks:
            try:
                emb = get_embedding(chunk)
            except Exception as e:
                print(f"Embedding failed for chunk at index {idx}: {e}")
                continue
            vectors.append(emb)
            metas.append({"id": idx, "source": src, "text": chunk})
            idx += 1

    if len(vectors) == 0:
        raise RuntimeError("No embeddings were created")

    emb_dim = vectors[0].shape[0]
    xb = np.vstack(vectors)
    index = faiss.IndexFlatL2(emb_dim)
    index.add(xb)

    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))

    meta_path = os.path.join(out_dir, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as mf:
        for m in metas:
            mf.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Wrote {len(metas)} vectors to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str, help="datasets identifier (e.g. openwebtext)")
    group.add_argument("--source_file", type=str, help="local text file to ingest")
    parser.add_argument("--limit", type=int, default=1000, help="max documents to stream from dataset")
    parser.add_argument("--out_dir", type=str, default="rag_index", help="output directory for index and metadata")
    args = parser.parse_args()

    ingest_dataset(dataset_name=args.dataset, source_file=args.source_file, limit=args.limit, out_dir=args.out_dir)
