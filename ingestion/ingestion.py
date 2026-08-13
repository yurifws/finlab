# Ingest SEC filings into Qdrant for hybrid RAG.
# 1) Fetch 10-K / 10-Q text via Edgar
# 2) Semantically chunk (embed paragraphs → HDBSCAN clusters → token packs)
# 3) Embed each chunk with dense (semantic) + sparse (BM25) + ColBERT (late interaction)
# 4) Upload points to an existing Qdrant collection (run create_collection.py first)
# Query lives in test_query.py — keep ingest and search separate.

import os
import uuid

from dotenv import load_dotenv
from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
)
from qdrant_client import QdrantClient, models
from utils.semantic_chunker import SemanticChunker
from utils.edgar_client import EdgarClient

# Load QDRANT_URL / QDRANT_API_KEY / EDGAR_EMAIL from .env.
load_dotenv()

# Model IDs passed to fastembed (downloaded on first use).
# Dense MiniLM -> 384-dim semantic vectors.
# Sparse BM25 -> term index/weight pairs.
# ColBERT -> multi-vector (one 128-d vector per token); scored with MaxSim at query time.
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

# Must match the collection created by create_collection.py.
COLLECTION_NAME = "financial"
# Soft token budget per semantic chunk (passed to SemanticChunker).
MAX_TOKENS = 300

# Cloud Qdrant client (URL + API key from the environment).
# ColBERT multi-vectors are large payloads; raise timeout past the 5s default.
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120,
)

# Pull latest 10-K and 10-Q for AAPL; each returns metadata + selected Item sections.
edgar_client = EdgarClient(email=os.getenv("EDGAR_EMAIL"))
data_10k = edgar_client.fetch_filing_data(ticker="AAPL", form_type="10-K")
text_10k = edgar_client.get_combined_text(data_10k)

data_10q = edgar_client.fetch_filing_data(ticker="AAPL", form_type="10-Q")
text_10q = edgar_client.get_combined_text(data_10q)

# Cluster related paragraphs by meaning, then pack under MAX_TOKENS.
chunker = SemanticChunker(max_tokens=MAX_TOKENS)

# Keep chunk text + filing metadata together so every Qdrant point is attributable.
all_chunks = []
for data, text in [(data_10k, text_10k), (data_10q, text_10q)]:
    chunks = chunker.create_chunks(text)
    for chunk in chunks:
        all_chunks.append({
            "text": chunk,
            "metadata": data["metadata"],
        })

# Three encoders for hybrid retrieval:
# - dense (MiniLM): semantic similarity — paraphrases / related meaning
# - sparse (BM25): keyword/term matching — tickers, legal jargon, exact phrases
# - ColBERT: late interaction — token-level MaxSim re-ranking for finer relevance
dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)

# Build points: each chunk gets three vectors + payload (original text + metadata).
points = []
for chunk_data in all_chunks:
    chunk = chunk_data["text"]
    metadata = chunk_data["metadata"]

    # passage_embed is for documents; returns an iterator — take the first vector.
    # as_object() -> {"indices": [...], "values": [...]} for Qdrant's SparseVector.
    # ColBERT passage: list of per-token vectors (multi-vector), not a single embedding.
    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_embedding = list(sparse_model.passage_embed([chunk]))[0].as_object()
    colbert_embedding = list(colbert_model.passage_embed([chunk]))[0].tolist()

    point = models.PointStruct(
        # Qdrant point ids must be UUID or unsigned int — string UUID is fine.
        id=str(uuid.uuid4()),
        # Must use the same names as vectors_config / sparse_vectors_config.
        vector={
            "dense": dense_embedding,
            "sparse": sparse_embedding,
            "colbert": colbert_embedding,
        },
        # Payload keeps the raw text + filing metadata so hits are human-readable.
        payload={"text": chunk, "metadata": metadata},
    )
    points.append(point)

# Upload in small batches: each ColBERT point holds many token vectors.
# batch_size=5 avoids WriteTimeout on cloud when payloads are large.
qdrant.upload_points(
    collection_name=COLLECTION_NAME,
    points=points,
    batch_size=5,
)

print(f"Uploaded {len(points)} points to '{COLLECTION_NAME}'.")
