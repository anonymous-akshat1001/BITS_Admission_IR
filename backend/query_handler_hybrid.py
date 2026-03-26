import os
import warnings
import pprint
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from google import genai

warnings.filterwarnings("ignore", category=FutureWarning)



# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "hybrid_corpus_v1"
DENSE_VECTOR_NAME = "dense-vector" 
SPARSE_VECTOR_NAME = "sparse-vector" 


SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1" 
# OPENAI_EMBED_MODEL = "text-embedding-ada-002"


RETRIEVER_SEARCH_K = 10 

# print("Initializing components...")
# qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

# dense_embeddings = OpenAIEmbeddings(
#     model=OPENAI_EMBED_MODEL,
#     openai_api_key=OPENAI_API_KEY,
#     disallowed_special=()
# )


# sparse_embeddings = FastEmbedSparse(
#     model_name=SPARSE_MODEL_NAME
# )

# llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo") # Or your preferred model
# print("Components initialized.")


print("Initializing components...")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

dense_embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

sparse_embeddings = FastEmbedSparse(
    model_name=SPARSE_MODEL_NAME
)

print("Components initialized.")



def setup_hybrid_retriever(client, collection_name, dense_embed_model, sparse_embed_model):
    """
    Set up the Qdrant vector store and retriever for hybrid search.
    """
    print("Setting up hybrid retriever using QdrantVectorStore...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embed_model,        
        sparse_embedding=sparse_embed_model,  
        retrieval_mode=RetrievalMode.HYBRID, 
        vector_name=DENSE_VECTOR_NAME,         
        sparse_vector_name=SPARSE_VECTOR_NAME  
    )


    retriever = vector_store.as_retriever(
        search_kwargs={'k': RETRIEVER_SEARCH_K}
       
    )
    print("Retriever setup complete.")
    return retriever

GEMINI_SYSTEM_PROMPT = """You are a helpful and accurate Q&A assistant for BITS Pilani admissions.
Your ONLY job is to answer questions strictly based on the context provided below.

Rules you MUST follow:
1. Answer ONLY from the provided context. Do not use any external knowledge.
2. If the question is about a topic NOT covered in the context (e.g., other colleges, general knowledge, unrelated topics), respond with: "I'm sorry, I can only answer questions related to BITS Pilani admissions based on official information."
3. Do not guess, infer, or hallucinate. If the context doesn't have enough information, say: "I don't have enough information in my knowledge base to answer this accurately."
4. Be concise, clear, and helpful.

Context:
{context}

Question: {question}

Answer:"""


def answer_with_gemini(docs, question):
    """Generate an answer using Gemini based on retrieved docs."""
    if not docs:
        return "I don't have enough information in my knowledge base to answer this accurately."

    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    prompt = GEMINI_SYSTEM_PROMPT.format(context=context, question=question)

    try:
        response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "An error occurred while generating the answer."


def query_and_print(retriever, question):
    print("\nQuerying")
    print(f"Question: {question}")

    docs = retriever.invoke(question)

    answer = answer_with_gemini(docs, question)
    print(f"\nAnswer:\n{answer}")

    print("\nSource Documents Retrieved")
    pprint.pprint(docs)



def main():
    print("\nInitializing RAG System (this may take a moment for model loading)")
    retriever = setup_hybrid_retriever(
        qdrant_client,
        COLLECTION_NAME,
        dense_embeddings,
        sparse_embeddings
    )

    # rag_chain = setup_rag_chain(retriever, llm)
    print("RAG System Ready")

    #Interactive Query Loop
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