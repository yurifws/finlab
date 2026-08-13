from fastapi import FastAPI
from models.search import SearchRequest, SearchResponse
from services.search import SearchService
from config.settings import settings

app = FastAPI(
    title="Financial Search API", description="API for searching financial documents"
)

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collection_name,
)

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    return search_service.search(request.query, request.limit)

@app.get("/")
def root():
    return {"status": "online"}