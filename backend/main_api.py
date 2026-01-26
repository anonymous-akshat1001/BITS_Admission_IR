import os
import warnings
import logging
import asyncio
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# Suppress warnings FIRST
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - load early
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    logger.error("CRITICAL: Missing QDRANT_URL or QDRANT_API_KEY environment variables!")

COLLECTION_NAME = "hybrid_corpus_v1"
DENSE_VECTOR_NAME = "dense-vector"
SPARSE_VECTOR_NAME = "sparse-vector"
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_RETRIEVER_SEARCH_K = 10
RERANKING_BASE_SEARCH_K = 20
RERANKER_TOP_N = 3

# Global state
rag_components = {}
rag_initialized = False
initialization_in_progress = False
initialization_error = None

# Pydantic Models
class QueryRequest(BaseModel):
    query: str

class SourceDocumentMetadata(BaseModel):
    file_path: Optional[str] = None
    doc_name: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None
    num_pages: Optional[int] = None
    _id: Optional[str] = None
    _collection_name: Optional[str] = None
    score: Optional[float] = None
    class Config:
        extra = 'allow'

class SourceDocumentResponse(BaseModel):
    page_content: str
    metadata: SourceDocumentMetadata

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocumentResponse]

# Helper function to format documents
def format_docs(docs) -> List[SourceDocumentResponse]:
    from langchain_core.documents import Document
    formatted = []
    for doc in docs:
        metadata_dict = doc.metadata.copy()
        score = getattr(doc, 'score', metadata_dict.get('score', None))
        if score is not None:
            metadata_dict['score'] = float(score)
        formatted.append(
            SourceDocumentResponse(
                page_content=doc.page_content,
                metadata=SourceDocumentMetadata(**metadata_dict)
            )
        )
    return formatted

def answer_from_docs(docs) -> str:
    if not docs:
        return "I don't know based on the given information."
    return docs[0].page_content

# Background initialization function
async def initialize_rag_background():
    """Initialize RAG components in the background without blocking"""
    global rag_initialized, initialization_in_progress, initialization_error
    
    if rag_initialized or initialization_in_progress:
        return
    
    initialization_in_progress = True
    logger.info("=" * 60)
    logger.info("BACKGROUND INITIALIZATION: Loading heavy models...")
    logger.info("=" * 60)
    
    try:
        # Run the synchronous initialization in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _initialize_rag_sync)
        
        rag_initialized = True
        initialization_error = None
        logger.info("=" * 60)
        logger.info("ALL MODELS LOADED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        initialization_error = str(e)
        logger.exception(f"RAG initialization failed: {e}")
    finally:
        initialization_in_progress = False

def _initialize_rag_sync():
    """Synchronous initialization logic"""
    if not rag_components.get("qdrant_client"):
        raise Exception("Qdrant client not available")
    
    # Import heavy libraries only when needed
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    
    logger.info("Loading dense embeddings...")
    rag_components["dense_embeddings"] = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    logger.info("Dense embeddings loaded")
    
    logger.info("Loading sparse embeddings...")
    rag_components["sparse_embeddings"] = FastEmbedSparse(
        model_name=SPARSE_MODEL_NAME
    )
    logger.info("Sparse embeddings loaded")
    
    logger.info("Loading cross encoder...")
    rag_components["cross_encoder"] = HuggingFaceCrossEncoder(
        model_name=CROSS_ENCODER_MODEL_NAME
    )
    logger.info("Cross encoder loaded")
    
    # Setup hybrid retriever
    logger.info("Setting up hybrid retriever...")
    vector_store = QdrantVectorStore(
        client=rag_components["qdrant_client"],
        collection_name=COLLECTION_NAME,
        embedding=rag_components["dense_embeddings"],
        sparse_embedding=rag_components["sparse_embeddings"],
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )
    rag_components["hybrid_retriever"] = vector_store.as_retriever(
        search_kwargs={'k': HYBRID_RETRIEVER_SEARCH_K}
    )
    logger.info("✓ Hybrid retriever ready")
    
    # Setup reranking retriever
    logger.info("Setting up reranking retriever...")
    vector_store_rerank = QdrantVectorStore(
        client=rag_components["qdrant_client"],
        collection_name=COLLECTION_NAME,
        embedding=rag_components["dense_embeddings"],
        sparse_embedding=rag_components["sparse_embeddings"],
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME
    )
    base_retriever = vector_store_rerank.as_retriever(
        search_kwargs={"k": RERANKING_BASE_SEARCH_K}
    )
    reranker = CrossEncoderReranker(
        model=rag_components["cross_encoder"],
        top_n=RERANKER_TOP_N
    )
    rag_components["reranking_retriever"] = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker
    )
    logger.info("Reranking retriever ready")

# FastAPI Lifespan - Start background initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("FastAPI startup initiated")
    logger.info(f"PORT environment variable: {os.getenv('PORT', 'NOT SET')}")
    logger.info("=" * 60)
    
    # Initialize Qdrant client
    try:
        from qdrant_client import QdrantClient
        rag_components["qdrant_client"] = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
        logger.info("Qdrant client initialized")
    except Exception as e:
        logger.error(f"✗ Qdrant client failed: {e}")
        rag_components["qdrant_client"] = None
    
    logger.info("Server ready - starting background model loading...")
    logger.info("=" * 60)
    
    # Start background initialization WITHOUT waiting for it
    asyncio.create_task(initialize_rag_background())
    
    yield
    
    logger.info("FastAPI shutdown")
    rag_components.clear()

# Create app instance
app = FastAPI(
    title="Hybrid RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def read_root():
    return {
        "message": "Hybrid RAG API is running",
        "status": "online",
        "models_loaded": rag_initialized,
        "initialization_in_progress": initialization_in_progress
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qdrant_connected": rag_components.get("qdrant_client") is not None,
        "models_loaded": rag_initialized,
        "initialization_in_progress": initialization_in_progress,
        "initialization_error": initialization_error
    }

@app.post("/query/hybrid/", response_model=QueryResponse)
async def query_hybrid(request: QueryRequest):
    # Check if models are ready
    if not rag_initialized:
        if initialization_error:
            raise HTTPException(500, f"Model initialization failed: {initialization_error}")
        raise HTTPException(503, "Models are still loading. Please try again in a moment.")
    
    logger.info(f"Hybrid query: '{request.query[:50]}...'")
    
    if "hybrid_retriever" not in rag_components:
        raise HTTPException(503, "Retriever not initialized")
    
    try:
        # Run retrieval in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None, 
            rag_components["hybrid_retriever"].invoke, 
            request.query
        )
        answer = answer_from_docs(docs)
        logger.info(f"Answer generated: '{answer[:50]}...'")
        
        return QueryResponse(answer=answer, source_documents=format_docs(docs))
    except Exception as e:
        logger.exception(f"Error in hybrid query: {e}")
        raise HTTPException(500, str(e))

@app.post("/query/hybrid-rerank/", response_model=QueryResponse)
async def query_hybrid_rerank(request: QueryRequest):
    # Check if models are ready
    if not rag_initialized:
        if initialization_error:
            raise HTTPException(500, f"Model initialization failed: {initialization_error}")
        raise HTTPException(503, "Models are still loading. Please try again in a moment.")
    
    logger.info(f"Hybrid-rerank query: '{request.query[:50]}...'")
    
    if "reranking_retriever" not in rag_components:
        raise HTTPException(503, "Reranking retriever not initialized")
    
    try:
        # Run retrieval in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None,
            rag_components["reranking_retriever"].invoke,
            request.query
        )
        answer = answer_from_docs(docs)
        logger.info(f"Reranked answer: '{answer[:50]}...'")
        
        return QueryResponse(answer=answer, source_documents=format_docs(docs))
    except Exception as e:
        logger.exception(f"Error in rerank query: {e}")
        raise HTTPException(500, str(e))

# Entry point
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )