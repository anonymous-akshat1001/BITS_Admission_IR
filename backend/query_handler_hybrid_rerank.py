
import os
import warnings
import pprint
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate



from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


warnings.filterwarnings("ignore", category=FutureWarning)


load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "hybrid_corpus_v1"
DENSE_VECTOR_NAME = "dense-vector" 
SPARSE_VECTOR_NAME = "sparse-vector" 

SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1" 
# OPENAI_EMBED_MODEL = "text-embedding-ada-002"


RETRIEVER_SEARCH_K = 20 # Number of initial results to retrieve for reranking
RERANKER_TOP_N = 3      # Number of results after reranking
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Local model


# Initialization
# print("Initializing components (rerank version)...")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

# dense_embeddings = OpenAIEmbeddings(
#     model=OPENAI_EMBED_MODEL,
#     openai_api_key=OPENAI_API_KEY,
#     disallowed_special=()
# )

# sparse_embeddings = FastEmbedSparse(
#     model_name=SPARSE_MODEL_NAME
# )


# llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

print("Initializing embeddings (offline)...")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

dense_embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

sparse_embeddings = FastEmbedSparse(
    model_name=SPARSE_MODEL_NAME
)

print("Initializing CrossEncoder reranker...")
cross_encoder_model = HuggingFaceCrossEncoder(
    model_name=CROSS_ENCODER_MODEL_NAME
)

print("All models initialized.")


print(f"Initializing CrossEncoder model via Langchain: {CROSS_ENCODER_MODEL_NAME}...")
cross_encoder_model = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL_NAME)
print("CrossEncoder model initialized via Langchain.")

print("Core components initialized.")


def setup_reranking_retriever(client, collection_name, dense_embed_model, sparse_embed_model, cross_encoder):
    """
    Set up the Qdrant vector store as base retriever and wrap it
    with a ContextualCompressionRetriever using a CrossEncoderReranker.
    """
    print("Setting up base hybrid retriever (QdrantVectorStore)...")
    #Base Retriever
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embed_model,
        sparse_embedding=sparse_embed_model,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME
    )

    base_retriever = vector_store.as_retriever(
        search_kwargs={'k': RETRIEVER_SEARCH_K} # Retrieve more docs initially
    )
    print(f"Base retriever configured to fetch top {RETRIEVER_SEARCH_K} docs.")

    #Reranker
    print("Setting up CrossEncoderReranker...")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=RERANKER_TOP_N)
    print(f"Reranker configured to return top {RERANKER_TOP_N} docs.")

    print("Setting up ContextualCompressionRetriever...")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )
    print("Contextual Compression Retriever setup complete.")
    return compression_retriever

def setup_rag_chain(retriever, llm):
    """
    Set up the RAG chain using the provided retriever (which will be the compression retriever).
    """
    print("Setting up RAG chain...")
    template = """ You are a kind and helpful QnA assistant. Your job is to assist Faculty and students of BITS Pilani in resolving their doubts and queries based SOLELY on the provided context. If the context doesn't contain the answer, say you don't know.
Context:
{context}

Question: {question}

Answer:
    """
    prompt = PromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain setup complete.")
    return chain

# def query_and_print(chain, retriever, question):
#     """
#     Query the RAG chain, retrieve source documents (post-reranking), and print results.
#     """
#     print("\nQuerying (with Reranking)")
#     print(f"Question: {question}")

#     answer = chain.invoke(question)
#     print(f"\nAnswer:\n{answer}")

#     source_documents = retriever.invoke(question)
#     print(f"\nSource Documents Retrieved (Top {RERANKER_TOP_N} after Reranking)")
#     pprint.pprint(source_documents)
#     print("End of Query")

def query_and_print(retriever, question):
    print("\nQuerying (Hybrid + Reranking)")
    print(f"Question: {question}")

    docs = retriever.invoke(question)

    if not docs:
        print("\nAnswer:\nI don't know based on the given information.")
        return

    print("\nAnswer (Top reranked chunk):\n")
    print(docs[0].page_content)

    print(f"\nSource Documents Retrieved (Top {len(docs)})")
    pprint.pprint(docs)


# Main Execution
def main():
    print("\n--- Initializing RAG System with Reranking (this may take a moment for model loading) ---")
    #Setup Retriever
    retriever = setup_reranking_retriever(
        qdrant_client,
        COLLECTION_NAME,
        dense_embeddings,
        sparse_embeddings,
        cross_encoder_model 
    )

    #Setup RAG Chain
    # rag_chain = setup_rag_chain(retriever, llm)
    print("RAG System with Reranking Ready")

    #Query Loop
    while True:
        try:
            question = input("\nEnter your question (or type 'quit'/'exit' to stop): ")
            question_lower = question.strip().lower()

            if question_lower in ["quit", "exit"]:
                print("Exiting interactive session.")
                break

            if not question.strip():
                print("Please enter a question.")
                continue

            # query_and_print(rag_chain, retriever, question)
            query_and_print(retriever, question)

        except EOFError:
            print("\nExiting interactive session.")
            break
        except KeyboardInterrupt:
            print("\nExiting interactive session.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()
