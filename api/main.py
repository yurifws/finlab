from fastapi import FastAPI
from models.search import SearchRequest, SearchResponse
from models.rag import RAGRequest, RAGResponse
from services.search import SearchService
from services.rag import RAGService
from config.settings import settings

app = FastAPI(
    title="Financial Search API", description="API for searching financial documents"
)

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collection_name,
)

rag_service = RAGService(search_service=search_service)

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    return search_service.search(request.query, request.limit)

@app.post("/rag", response_model=RAGResponse)
def rag(request: RAGRequest):
    return rag_service.generate_answer(request.query, request.limit)

@app.get("/")
def root():
    return {"status": "online"}