# Hybrid search against the financial collection (run after ingestion.py).
# 1) Prefetch dense + sparse candidates, fuse with RRF
# 2) Re-rank the shortlist with ColBERT MaxSim

import os

from dotenv import load_dotenv
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

# Load QDRANT_URL / QDRANT_API_KEY (and any other secrets) from .env.
load_dotenv()

# Model IDs passed to fastembed (downloaded on first use).
# Dense MiniLM -> 384-dim semantic vectors.
# Sparse BM25 -> term index/weight pairs.
# ColBERT -> multi-vector (one 128-d vector per token); scored with MaxSim at query time.
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

# Named bucket in Qdrant where this financial text will live.
COLLECTION_NAME = "financial"

# Cloud Qdrant client (URL + API key from the environment).
# ColBERT multi-vectors are large payloads; raise timeout past the 5s default.
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120,
)

# Three encoders for hybrid retrieval:
# - dense (MiniLM): semantic similarity — paraphrases / related meaning
# - sparse (BM25): keyword/term matching — tickers, legal jargon, exact phrases
# - ColBERT: late interaction — token-level MaxSim re-ranking for finer relevance
dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)

# Demo query — same pipeline you would run at search time after ingest.
query_text = "what are the main financial risks?"

# query_embed (not passage_embed) is the asymmetric counterpart for search queries.
# Dense -> fixed 384-float list.
# Sparse -> {"indices", "values"} term weights (as_object).
# ColBERT query -> also multi-vector; used only in the final re-rank stage below.
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()
query_colbert = list(colbert_model.query_embed([query_text]))[0].tolist()

# Two-stage retrieval:
# 1) Prefetch: dense + sparse candidate lists, fused with RRF (Reciprocal Rank Fusion)
# 2) Outer query: ColBERT MaxSim re-ranks those fused candidates and returns top 3
results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        models.Prefetch(
            prefetch=[
                # using= must match the named vectors created in create_collection.
                # limit=10: how many candidates each channel contributes before fusion.
                models.Prefetch(query=query_dense, using="dense", limit=10),
                models.Prefetch(
                    # ** unpacks {"indices", "values"} into SparseVector fields.
                    query=models.SparseVector(**query_sparse),
                    using="sparse",
                    limit=10,
                ),
            ],
            # RRF merges the two ranked lists without needing comparable raw scores.
            # limit=20: fused shortlist size passed to ColBERT re-rank.
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
        ),
    ],
    # Re-rank the RRF shortlist with ColBERT (more precise, more expensive).
    query=query_colbert,
    using="colbert",
    limit=3,
)

# Scale scores to [0, 1] relative to the best hit (easier to read than raw MaxSim).
max_score = max(result.score for result in results.points)

# Inspect the top hits: ColBERT re-rank score and a short preview of each chunk.
for r in results.points:
    normalized_score = r.score / max_score
    print(f"Score: {normalized_score}")
    # Truncate payload text so the console stays readable.
    print(f"Text: {r.payload['text'][:100]}...")
    print("-" * 80)
