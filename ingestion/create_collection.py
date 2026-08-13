# Create (or recreate) the Qdrant collection used by ingestion + query.
# Run once before ingestion.py. Named vectors: dense, colbert, sparse.

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Load QDRANT_URL / QDRANT_API_KEY (and any other secrets) from .env.
load_dotenv()

# Named bucket in Qdrant where this financial text will live.
COLLECTION_NAME = "financial"

# Cloud Qdrant client (URL + API key from the environment).
# ColBERT multi-vectors are large payloads; raise timeout past the 5s default.
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120,
)

# Recreate the collection so each run starts clean (demo-friendly, not production).
# delete first — create_collection fails if the named collection already exists.
qdrant.delete_collection(COLLECTION_NAME)
# Named vectors:
# - "dense": single 384-d cosine vector (semantic)
# - "colbert": multi-vector (token-level 128-d), compared with MaxSim
# - "sparse": BM25 inverted index (separate sparse_vectors_config)
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        # size= must match the dense model output dim (MiniLM-L6 = 384).
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "colbert": models.VectorParams(
            # ColBERTv2 token vectors are 128-d; MaxSim = late-interaction score.
            size=128,
            distance=models.Distance.COSINE,
            # Multi-vector: each point stores many token vectors; MaxSim picks best matches.
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    },
    # Sparse lives outside vectors_config — Qdrant treats it as an inverted index.
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
)
