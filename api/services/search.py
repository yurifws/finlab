from qdrant_client import QdrantClient, models
from models.search import SearchResponse, SearchResult
from services.embeddings import EmbeddingsService

class SearchService:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self.qdrant_client = QdrantClient(
            url=qdrant_url, api_key=qdrant_api_key, timeout=120
        )
        self.collection_name = collection_name
        self.embeddings_service = EmbeddingsService()

    def search(self, query: str, limit: int = 3):
        query_dense, query_sparse, query_colbert = self.embeddings_service.embed_query(query)
        
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(    
                    prefetch=[
                        models.Prefetch(query=query_dense, using="dense", limit=10),
                        models.Prefetch(
                            query=models.SparseVector(**query_sparse),
                            using="sparse",
                            limit=10,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=20,
                ),
            ],
            query=query_colbert,
            using="colbert",
            limit=limit,
        )
        
        points = results.points
        if not points:
            return SearchResponse(results=[])

        max_score = max(result.score for result in points)

        search_results = [
            SearchResult(
                score=result.score / max_score,
                text=result.payload["text"],
                metadata=result.payload["metadata"],
            )
            for result in points
        ]

        return SearchResponse(results=search_results)