from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from config.settings import settings


class EmbeddingsService:
    def __init__(self):
        self.dense_embedding = TextEmbedding(settings.dense_model)
        self.sparse_embedding = SparseTextEmbedding(settings.sparse_model)
        self.colbert_embedding = LateInteractionTextEmbedding(settings.colbert_model)

    def embed_query(self, query: str):
        dense = list(self.dense_embedding.query_embed([query]))[0].tolist()
        sparse = list(self.sparse_embedding.query_embed([query]))[0].as_object()
        colbert = list(self.colbert_embedding.query_embed([query]))[0].tolist()

        return dense, sparse, colbert
