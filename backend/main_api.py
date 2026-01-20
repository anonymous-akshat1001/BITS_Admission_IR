import os
import warnings
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient

from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Correct imports
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder



warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# from dotenv import load_dotenv
# load_dotenv()


#Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    logger.error("API keys/URL missing. Set env vars.")

COLLECTION_NAME = "hybrid_corpus_v1"
DENSE_VECTOR_NAME = "dense-vector"
SPARSE_VECTOR_NAME = "sparse-vector"
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"
OPENAI_EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL_NAME = "gpt-3.5-turbo"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_RETRIEVER_SEARCH_K = 10
RERANKING_BASE_SEARCH_K = 20
RERANKER_TOP_N = 3

rag_components = {}

#Helper Functions 
def setup_hybrid_retriever_only(client, collection_name, dense_embed_model, sparse_embed_model):
    logger.info("Setting up standard hybrid retriever...")
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=dense_embed_model, sparse_embedding=sparse_embed_model, retrieval_mode=RetrievalMode.HYBRID, vector_name=DENSE_VECTOR_NAME, sparse_vector_name=SPARSE_VECTOR_NAME)
    retriever = vector_store.as_retriever(search_kwargs={'k': HYBRID_RETRIEVER_SEARCH_K})
    logger.info(f"Standard hybrid retriever setup complete (k={HYBRID_RETRIEVER_SEARCH_K}).")
    return retriever

def setup_reranking_retriever(client, collection_name, dense_embed_model, sparse_embed_model, cross_encoder):
    logger.info("Setting up reranking retriever...")
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=dense_embed_model, sparse_embedding=sparse_embed_model, retrieval_mode=RetrievalMode.HYBRID, vector_name=DENSE_VECTOR_NAME, sparse_vector_name=SPARSE_VECTOR_NAME)
    base_retriever = vector_store.as_retriever(search_kwargs={'k': RERANKING_BASE_SEARCH_K})
    logger.info(f"Reranking base retriever configured (k={RERANKING_BASE_SEARCH_K}).")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=RERANKER_TOP_N)
    logger.info(f"Reranker configured (top_n={RERANKER_TOP_N}).")
    compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)

    logger.info("Reranking retriever setup complete.")
    return compression_retriever

def setup_rag_chain(retriever, llm):
    logger.info("Setting up RAG chain...")
    template ="""
You are a helpful Q&A assistant for BITS Pilani. 
Answer questions only using the provided context. 
If the answer is not in the context, reply with: "I don't know based on the given information." Be clear, concise, and accurate.
Context:{context}
Question: {question}
Answer:
    """
    prompt = PromptTemplate.from_template(template)
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    logger.info("RAG chain setup complete.")
    return chain

#FastAPI Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI startup: Connecting to Qdrant only...")
    try:
        rag_components["qdrant_client"] = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
        logger.info("Qdrant connected.")
    except Exception as e:
        logger.exception(f"Startup failed: {e}")
    yield
    logger.info("FastAPI shutdown.")
    rag_components.clear()


#Pydantic Models
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
    class Config: extra = 'allow'

class SourceDocumentResponse(BaseModel):
    page_content: str
    metadata: SourceDocumentMetadata

def format_docs(docs: List[Document]) -> List[SourceDocumentResponse]:
    formatted = []
    for doc in docs:
        metadata_dict = doc.metadata.copy()
        score = getattr(doc, 'score', metadata_dict.get('score', None))
        if score is not None: metadata_dict['score'] = float(score)
        formatted.append(SourceDocumentResponse(page_content=doc.page_content, metadata=SourceDocumentMetadata(**metadata_dict)))
    return formatted

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocumentResponse]


def initialize_rag():
    if "hybrid_rag_chain" in rag_components:
        return  # already initialized

    logger.info("Lazy RAG initialization started...")

    rag_components["llm"] = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.2
    )

    rag_components["dense_embeddings"] = OpenAIEmbeddings(
        model=OPENAI_EMBED_MODEL,
        openai_api_key=OPENAI_API_KEY,
        disallowed_special=()
    )

    rag_components["sparse_embeddings"] = FastEmbedSparse(
        model_name=SPARSE_MODEL_NAME
    )

    rag_components["cross_encoder"] = HuggingFaceCrossEncoder(
        model_name=CROSS_ENCODER_MODEL_NAME
    )

    rag_components["hybrid_retriever"] = setup_hybrid_retriever_only(
        rag_components["qdrant_client"],
        COLLECTION_NAME,
        rag_components["dense_embeddings"],
        rag_components["sparse_embeddings"]
    )

    rag_components["reranking_retriever"] = setup_reranking_retriever(
        rag_components["qdrant_client"],
        COLLECTION_NAME,
        rag_components["dense_embeddings"],
        rag_components["sparse_embeddings"],
        rag_components["cross_encoder"]
    )

    rag_components["hybrid_rag_chain"] = setup_rag_chain(
        rag_components["hybrid_retriever"],
        rag_components["llm"]
    )

    rag_components["reranking_rag_chain"] = setup_rag_chain(
        rag_components["reranking_retriever"],
        rag_components["llm"]
    )

    logger.info("Lazy RAG initialization complete.")



#App Instance
app = FastAPI(title="Hybrid RAG API", version="1.0.0", lifespan=lifespan)

#Add CORS Middleware
# Allow all origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, 
    allow_methods=["*"],    
    allow_headers=["*"],    
)

# API Endpoints
@app.get("/")
async def read_root(): return {"message": "Welcome to Hybrid RAG API"}

@app.post("/query/hybrid/", response_model=QueryResponse)
async def query_hybrid(request: QueryRequest):
    initialize_rag()
    logger.info(f"Hybrid query: '{request.query[:50]}...'" )
    if "hybrid_rag_chain" not in rag_components:
        raise HTTPException(503, "RAG not initialized")
    try:
        retriever = rag_components["hybrid_retriever"]
        docs = retriever.invoke(request.query)
        chain = rag_components["hybrid_rag_chain"]
        answer = chain.invoke(request.query)
        logger.info(f"Hybrid answer: '{answer[:50]}...'" )
        return QueryResponse(answer=answer, source_documents=format_docs(docs))
    except Exception as e:
        logger.exception(f"Error hybrid query: {e}")
        raise HTTPException(500, str(e))

@app.post("/query/hybrid-rerank/", response_model=QueryResponse)
async def query_hybrid_rerank(request: QueryRequest):
    initialize_rag()
    logger.info(f"Hybrid-rerank query: '{request.query[:50]}...'" )
    if "reranking_rag_chain" not in rag_components:
        raise HTTPException(503, "RAG not initialized")
    try:
        retriever = rag_components["reranking_retriever"]
        docs = retriever.invoke(request.query)
        chain = rag_components["reranking_rag_chain"]
        answer = chain.invoke(request.query)
        logger.info(f"Hybrid-rerank answer: '{answer[:50]}...'" )
        return QueryResponse(answer=answer, source_documents=format_docs(docs))
    except Exception as e:
        logger.exception(f"Error hybrid-rerank query: {e}")
        raise HTTPException(500, str(e))
