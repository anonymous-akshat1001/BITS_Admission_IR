import os

os.environ["FASTEMBED_CACHE_PATH"] = "/opt/render/project/.fastembed"

import warnings
import logging
import asyncio
import gc
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# Aggressive memory optimization
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
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

# Configuration
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

# Global state - MINIMAL
rag_components = {}
hybrid_ready = False
rerank_ready = False

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

# Helper functions
def format_docs(docs) -> List[SourceDocumentResponse]:
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

# LAZY initialization - load ONLY what's needed, WHEN it's needed
def initialize_hybrid_retriever():
    """Load ONLY hybrid retriever components"""
    global hybrid_ready
    
    if hybrid_ready:
        return
    
    logger.info("Loading hybrid retriever components...")
    
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
    
    # Load embeddings
    rag_components["dense_embeddings"] = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        max_length=512
    )
    logger.info("✓ Dense embeddings loaded")
    gc.collect()
    
    rag_components["sparse_embeddings"] = FastEmbedSparse(
        model_name=SPARSE_MODEL_NAME
    )
    logger.info("✓ Sparse embeddings loaded")
    gc.collect()
    
    # Setup retriever
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
    gc.collect()
    
    hybrid_ready = True

def initialize_reranking_retriever():
    """Load reranking components - ONLY if hybrid is already loaded"""
    global rerank_ready
    
    if rerank_ready:
        return
    
    # Ensure hybrid components are loaded first
    if not hybrid_ready:
        initialize_hybrid_retriever()
    
    logger.info("Loading reranking components...")
    
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    
    # Load cross encoder
    rag_components["cross_encoder"] = HuggingFaceCrossEncoder(
        model_name=CROSS_ENCODER_MODEL_NAME
    )
    logger.info("✓ Cross encoder loaded")
    gc.collect()
    
    # Setup reranking retriever
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
    logger.info("✓ Reranking retriever ready")
    gc.collect()
    
    rerank_ready = True

# Background initialization for hybrid only
async def initialize_hybrid_background():
    """Initialize hybrid retriever in background"""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, initialize_hybrid_retriever)
        logger.info("✓✓✓ HYBRID RETRIEVER READY ✓✓✓")
    except Exception as e:
        logger.exception(f"Hybrid initialization failed: {e}")

# FastAPI Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("FastAPI startup - MINIMAL MEMORY MODE")
    logger.info("=" * 60)
    
    # Initialize ONLY Qdrant client
    try:
        from qdrant_client import QdrantClient
        rag_components["qdrant_client"] = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
        logger.info("✓ Qdrant client initialized")
    except Exception as e:
        logger.error(f"✗ Qdrant client failed: {e}")
        rag_components["qdrant_client"] = None
    
    logger.info("✓ Server ready - models load on first query")
    logger.info("=" * 60)
    
    # Start loading hybrid retriever in background
    # asyncio.create_task(initialize_hybrid_background())
    
    yield
    
    logger.info("FastAPI shutdown")
    rag_components.clear()

# Create app instance
app = FastAPI(
    title="Hybrid RAG API - Memory Optimized",
    version="2.0.0",
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
        "message": "Hybrid RAG API - Memory Optimized",
        "status": "online",
        "hybrid_ready": hybrid_ready,
        "rerank_ready": rerank_ready,
        "memory_mode": "ultra-low"
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qdrant_connected": rag_components.get("qdrant_client") is not None,
        "hybrid_ready": hybrid_ready,
        "rerank_ready": rerank_ready
    }

@app.post("/query/hybrid/", response_model=QueryResponse)
async def query_hybrid(request: QueryRequest):
    # Lazy load if not ready
    if not hybrid_ready:
        logger.info("First hybrid query - initializing components...")
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, initialize_hybrid_retriever)
        except Exception as e:
            logger.exception(f"Failed to initialize hybrid retriever: {e}")
            raise HTTPException(500, f"Initialization failed: {str(e)}")
    
    logger.info(f"Hybrid query: '{request.query[:50]}...'")
    
    try:
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None, 
            rag_components["hybrid_retriever"].invoke, 
            request.query
        )
        answer = answer_from_docs(docs)
        logger.info(f"Answer: '{answer[:50]}...'")
        
        return QueryResponse(answer=answer, source_documents=format_docs(docs))
    except Exception as e:
        logger.exception(f"Error in hybrid query: {e}")
        raise HTTPException(500, str(e))

@app.post("/query/hybrid-rerank/", response_model=QueryResponse)
async def query_hybrid_rerank(request: QueryRequest):
    # Lazy load if not ready
    if not rerank_ready:
        logger.info("First rerank query - initializing components...")
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, initialize_reranking_retriever)
        except Exception as e:
            logger.exception(f"Failed to initialize reranking retriever: {e}")
            raise HTTPException(500, f"Initialization failed: {str(e)}")
    
    logger.info(f"Hybrid-rerank query: '{request.query[:50]}...'")
    
    try:
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