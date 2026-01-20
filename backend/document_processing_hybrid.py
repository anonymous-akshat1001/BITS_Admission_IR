import os
import re
import json
import datetime
import logging
import warnings
import uuid

from dotenv import load_dotenv

from qdrant_client import QdrantClient, models

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from fastembed import SparseTextEmbedding


# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
# dotenv_path = os.path.join(project_root, '.env')
# load_dotenv(dotenv_path=dotenv_path)

# Logging Setup
LOG_FILE = "processing_hybrid.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Qdrant configuration

QDRANT_URL=os.getenv("QDRANT_URL")


QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "hybrid_corpus_v1" 
DENSE_VECTOR_NAME = "dense-vector"
SPARSE_VECTOR_NAME = "sparse-vector"


DOCUMENT_FOLDER_PATH = "../document_corpus"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150 
TEXT_SPLITTER_SEPARATORS = ["# ", "## ", "### ", "\n\n", "\n", ". ", "! ", "? ", " "]

# Embedding models
SPARSE_MODEL_NAME = "prithvida/Splade_PP_en_v1"
OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_EMBED_DIMENSIONS = 1536

#Initialization
logging.info("Initializing embedding models...")
dense_embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBED_MODEL,
    openai_api_key=OPENAI_API_KEY,
    disallowed_special=()
)
sparse_embeddings = SparseTextEmbedding( 
    model_name=SPARSE_MODEL_NAME,
    batch_size=16, 
    cache_dir="sparse_cache" 
)
logging.info("Embedding models initialized.")

#Helper Functions

def load_checkpoint(checkpoint_path):
    """Load existing checkpoint data"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Checkpoint file {checkpoint_path} is corrupted. Starting fresh.")
            return {}
    logging.info(f"No checkpoint file found at {checkpoint_path}. Starting fresh.")
    return {}

def save_checkpoint(checkpoint_path, checkpoint_data):
    """Save updated checkpoint data"""
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except IOError as e:
        logging.error(f"Error saving checkpoint file {checkpoint_path}: {e}")

def clean_text(text):
    """Clean extracted PDF text and apply case folding"""
    if text is None:
        return ""
    text = text.lower() 
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  #control characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  #non-ASCII characters
    text = re.sub(r'\s+', ' ', text)  #whitespace
    text = re.sub(r'_+', ' ', text)  #underlines
    return text.strip()

def process_single_pdf(file_path):
    """Load, clean, and extract metadata from a single PDF using PyMuPDFLoader."""
    try:
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        if not pages:
             logging.warning(f"No content extracted from {file_path}. Skipping.")
             return None

        full_text = " ".join(page.page_content for page in pages if page.page_content)
        cleaned_text = clean_text(full_text)

        #metadata
        mod_time = os.path.getmtime(file_path)
        mod_date = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
        doc_name = os.path.basename(file_path)
    
        first_page_metadata = pages[0].metadata if pages else {}
        source = first_page_metadata.get('source', file_path) 

        metadata = {
            "file_path": file_path, 
            "doc_name": doc_name,
            "source": source, 
            "date": mod_date,
            "num_pages": len(pages)
        }
        logging.info(f"Processed file: {doc_name} (Last Modified: {mod_date}, Pages: {metadata['num_pages']})")
        # Return a single Document object representing the whole file for splitting
        return Document(page_content=cleaned_text, metadata=metadata)

    except FileNotFoundError:
        logging.error(f"File not found during processing: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None

def load_process_and_split_pdfs_with_checkpointing(folder_path):
    """Load, process, and split PDFs with checkpoint support."""
    abs_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), folder_path))
    if not os.path.isdir(abs_folder_path):
        logging.error(f"Document folder path does not exist or is not a directory: {abs_folder_path}")
        return []

    checkpoint_path = os.path.join(abs_folder_path, ".checkpoint_hybrid.json") 
    checkpoint_data = load_checkpoint(checkpoint_path)
    logging.info(f"Using checkpoint file: {checkpoint_path}")

    current_files = []
    try:
        for f in os.listdir(abs_folder_path):
            if f.endswith('.pdf') and not f.startswith('~'): 
                current_files.append(os.path.join(abs_folder_path, f))
    except Exception as e:
        logging.error(f"Error listing files in {abs_folder_path}: {e}")
        return []

    logging.info(f"Found {len(current_files)} PDF files in {abs_folder_path}")

    docs_to_process = []
    processed_file_paths = set() #
    new_checkpoint_data = {} # Build the new checkpoint data as we go

    for file_path in current_files:
        try:
            mod_time = os.path.getmtime(file_path)
            size = os.path.getsize(file_path)
        except FileNotFoundError:
            logging.warning(f"File not found during checkpoint check: {file_path}. Skipping.")
            continue
        except Exception as e:
            logging.error(f"Error getting stats for {file_path}: {e}. Skipping.")
            continue

        # Default to processing needed
        needs_processing = True
        file_key = file_path 

        # Check against checkpoint data
        if file_key in checkpoint_data:
            stored = checkpoint_data[file_key]
            if isinstance(stored, dict) and stored.get("mod_time") == mod_time and stored.get("size") == size:
                needs_processing = False
                logging.debug(f"Skipping {os.path.basename(file_path)} - checkpoint matches.")
                
                new_checkpoint_data[file_key] = stored
            else:
                 logging.info(f"Change detected for {os.path.basename(file_path)}. Needs processing.")
        else:
             logging.info(f"New file detected: {os.path.basename(file_path)}. Needs processing.")

        if needs_processing:
            doc = process_single_pdf(file_path)
            if doc:
                docs_to_process.append(doc)
                processed_file_paths.add(file_key)
                new_checkpoint_data[file_key] = {
                    "mod_time": mod_time,
                    "size": size
                }
            else:
                 
                 logging.warning(f"Processing failed for {os.path.basename(file_path)}. Checkpoint entry not updated.")
                 
                 if file_key in checkpoint_data:
                      new_checkpoint_data[file_key] = checkpoint_data[file_key]

    logging.info(f"Identified {len(docs_to_process)} new or modified PDFs to process.")

    files_in_checkpoint = set(new_checkpoint_data.keys())
    files_on_disk = set(current_files)
    files_to_remove = files_in_checkpoint - files_on_disk
    if files_to_remove:
        logging.info(f"Removing {len(files_to_remove)} files from checkpoint (deleted from source):")
        for f_rem in files_to_remove:
            logging.info(f" - {os.path.basename(f_rem)}")
            del new_checkpoint_data[f_rem]
            

    
    save_checkpoint(checkpoint_path, new_checkpoint_data)
    logging.info(f"Saved updated checkpoint data to {checkpoint_path}")

    if not docs_to_process:
        logging.info("No new documents needed processing.")
        return [], set() 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=TEXT_SPLITTER_SEPARATORS,
        keep_separator=False, 
        add_start_index=False
    )
    try:
        final_split_docs = text_splitter.split_documents(docs_to_process)
        logging.info(f"Split {len(docs_to_process)} documents into {len(final_split_docs)} chunks.")
    except Exception as e:
        logging.error(f"Error splitting documents: {e}")
        return [], set()

    return final_split_docs, processed_file_paths

def initialize_qdrant_client():
    """Initialize and return the Qdrant client."""
    logging.info(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        
        client.get_collections()
        logging.info("Successfully connected to Qdrant.")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant at {QDRANT_URL}: {e}")
        raise

def ensure_collection_exists(client: QdrantClient):
    """Check if the collection exists, create it if not."""
    logging.info(f"Checking for collection '{COLLECTION_NAME}'...")
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        logging.info(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception as e:
        
        logging.warning(f"Collection '{COLLECTION_NAME}' not found or error accessing: {e}. Attempting creation...")
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=OPENAI_EMBED_DIMENSIONS,
                        distance=models.Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                   SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }
                
            )
            logging.info(f"Successfully created collection '{COLLECTION_NAME}'.")
        except Exception as create_e:
            logging.error(f"Failed to create collection '{COLLECTION_NAME}': {create_e}")
            raise

def delete_points_for_files(client: QdrantClient, file_paths: set):
    """Delete all points associated with the given file paths from Qdrant."""
    if not file_paths:
        return
    logging.info(f"Deleting existing Qdrant points for {len(file_paths)} updated/processed files...")
    try:
       
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.file_path", 
                            match=models.MatchAny(any=list(file_paths))
                        )
                    ]
                )
            ),
            wait=True
        )
        logging.info(f"Deletion request sent for points related to processed files.")
    except Exception as e:
        logging.error(f"Error deleting points from Qdrant for processed files: {e}")
        

def index_documents(client: QdrantClient, docs, processed_file_paths):
    """Generate embeddings and index documents into Qdrant, deleting old points first."""
    if not docs:
        logging.info("No documents provided for indexing.")
        return

    
    delete_points_for_files(client, processed_file_paths)

    
    logging.info(f"Starting indexing for {len(docs)} new document chunks...")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs] 

    # Generate Dense Embeddings
    logging.info("Generating dense embeddings...")
    try:
        dense_vectors = dense_embeddings.embed_documents(texts)
        logging.info(f"Generated {len(dense_vectors)} dense vectors.")
    except Exception as e:
        logging.error(f"Error generating dense embeddings: {e}")
        raise 

    # Generate Sparse Embeddings
    sparse_batch_size = 64 
    sparse_results = []
    num_texts = len(texts)
    logging.info(f"Generating sparse embeddings for {num_texts} texts in batches of {sparse_batch_size}...")
    for i in range(0, num_texts, sparse_batch_size):
        batch_texts = texts[i:min(i + sparse_batch_size, num_texts)]
        if not batch_texts:
            continue
        logging.info(f"Generating sparse embeddings for batch {i//sparse_batch_size + 1}/{(num_texts + sparse_batch_size - 1)//sparse_batch_size} (size {len(batch_texts)}) from index {i}") # Corrected log message slightly
        try:
            
            batch_sparse_results = list(sparse_embeddings.embed(batch_texts))
            sparse_results.extend(batch_sparse_results)
        except Exception as e:
            logging.error(f"Error generating sparse embeddings for batch starting at index {i}: {e}")
            raise e

    logging.info(f"Generated {len(dense_vectors)} dense vectors and {len(sparse_results)} sparse vectors.")

    
    sparse_vectors_for_qdrant = []
    for doc_embedding in sparse_results: 
        
        if doc_embedding and hasattr(doc_embedding, 'indices') and hasattr(doc_embedding, 'values'):
            sparse_vectors_for_qdrant.append(
                models.SparseVector(
                    indices=doc_embedding.indices.tolist(), 
                    values=doc_embedding.values.tolist()   
                )
            )
        else:
             
             logging.warning("Invalid or empty sparse embedding result encountered for a chunk.")
             sparse_vectors_for_qdrant.append(models.SparseVector(indices=[], values=[])) 
   

    if len(dense_vectors) != len(sparse_vectors_for_qdrant) or len(dense_vectors) != len(docs):
         logging.error("Mismatch between number of docs, dense vectors, and sparse vectors.")
         raise ValueError("Mismatch between number of docs, dense vectors, and sparse vectors.")

    points_to_upsert = []
    for i, doc in enumerate(docs):
        # Generate a UUID for the point ID
        point_id = str(uuid.uuid4())

        points_to_upsert.append(
            models.PointStruct(
                id=point_id, 
                payload={
                    "page_content": doc.page_content, 
                    "metadata": doc.metadata, 
                },
                vector={
                    DENSE_VECTOR_NAME: dense_vectors[i],
                    SPARSE_VECTOR_NAME: sparse_vectors_for_qdrant[i],
                }
            )
        )

# upsert in qdrant database
    logging.info(f"Upserting {len(points_to_upsert)} points to Qdrant collection '{COLLECTION_NAME}'...")
    try:
        
        batch_size = 100 
        num_batches = (len(points_to_upsert) + batch_size - 1) // batch_size
        for i in range(0, len(points_to_upsert), batch_size):
            batch = points_to_upsert[i : i + batch_size]
            logging.info(f"Upserting batch {i//batch_size + 1}/{num_batches} (size {len(batch)}) abducted from index {i}") 
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True 
            )
        logging.info("Successfully upserted all points.")
    except Exception as e:
        logging.error(f"Failed to upsert points to Qdrant: {e}")
        raise 

# ain Execution
def main():
    logging.info("Starting Hybrid Document Processing Pipeline")
    try:
        
        final_split_docs, processed_file_paths = load_process_and_split_pdfs_with_checkpointing(DOCUMENT_FOLDER_PATH)

        if not final_split_docs:
            logging.info("Pipeline finished: No new documents were processed or split.")
            return 

        
        qdrant_client = initialize_qdrant_client()

       
        ensure_collection_exists(qdrant_client)

        
        index_documents(qdrant_client, final_split_docs, processed_file_paths)

        logging.info("Hybrid Document Processing Pipeline Finished Successfully")

    except Exception as e:
        logging.exception("Hybrid Document Processing Pipeline Failed")


if __name__ == "__main__":
    main()