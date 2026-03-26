import os
import requests
import urllib3
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# Load environment variables
load_dotenv()

# Suppress SSL warnings for cleaner logs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration parameters
COLLECTION_NAME = "hybrid_corpus_v1"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Source URLs
URLS = [
    # General & BITSAT
    "https://cdn3.digialm.com/EForms/configuredHtml/1823/94103/Index.html?_gl=1*1967jd6*_gcl_au*MTU4NzIwMjAxMi4xNzYzNTUwMzc2LjE1OTU1NDE4MjIuMTc2NTQyNzE5OS4xNzY1NDI4MzY4&utm_source=null&utm_medium=null&utm_campaign=null",
    "https://admissions.bits-pilani.ac.in/Privacy.html?_gl=1*1f0otbj*_gcl_au*NTQ3MjE3ODk4LjE3NzA5NTg5NjA.*_ga*MTE0ODEzOTgxMy4xNzcwOTU4OTYw*_ga_DYQ0HEBE5Z*czE3NzA5NTg5NjAkbzEkZzEkdDE3NzA5NTkwMzUkajYwJGwwJGgxNTI1Mjk1MDE0*_ga_EMYJ78JH5Y*czE3NzA5NTg5NjAkbzEkZzEkdDE3NzA5NTkwMzUkajYwJGwwJGgw",
    "https://admissions.bits-pilani.ac.in/FD/downloads/BITSAT-2026_brochure.pdf?02022026",
    "https://admissions.bits-pilani.ac.in/FD/FD.html",
    "https://admissions.bits-pilani.ac.in/ISA/ISA.html",
    "https://admissions.bits-pilani.ac.in/BT/bt.html",
    "https://admissions.bits-pilani.ac.in/FD/FD.html#FD_programmes",
    "https://admissions.bits-pilani.ac.in/FD/BITSAT_eligibility.html?02122025",
    "https://admissions.bits-pilani.ac.in/FD/downloads/Admission_Modality_FD.pdf?02122025",
    "https://admissions.bits-pilani.ac.in/FD/scholarship.html?02122025",
    "https://admissions.bits-pilani.ac.in/FD/FD_fee.html?25052025",
    "https://admissions.bits-pilani.ac.in/FD/BITSAT_FAQs.html",
    "https://g03.tcsion.com//per/g03/pub/726/EForms/image/ImageDocUpload/71161/3/801578309.pdf",

    # Cutoffs
    "https://admissions.bits-pilani.ac.in/FD/BITSAT_cutOffs.html?FQwp43qOeKhayi8LEQVUtJn3QNZ0TciWLP4NKxNMfcgzQdzcqZCCLqDBZRDnjcsHWFGgSC&yr=2025-2026&eKhayi8LEQwp4NKxN+CfCh+3qOVUtJn3QNZ0TciWLP4",
    "https://admissions.bits-pilani.ac.in/FD/BITSAT_cutOffs.html?FQwp43qOeKhayi8LEQVUtJn3QNZ0TciWLP4NKxNMfcgzQdzcqZCCLqDBZRDnjcsHWFGgSC&yr=2024-2025&eKhayi8LEQwp4NKxN+CfCh+3qOVUtJn3QNZ0TciWLP4",
    "https://admissions.bits-pilani.ac.in/FD/BITSAT_cutOffs.html?FQwp43qOeKhayi8LEQVUtJn3QNZ0TciWLP4NKxNMfcgzQdzcqZCCLqDBZRDnjcsHWFGgSC&yr=2023-2024&eKhayi8LEQwp4NKxN+CfCh+3qOVUtJn3QNZ0TciWLP4",
    "https://admissions.bits-pilani.ac.in/FD/BITSAT_cutOffs.html?FQwp43qOeKhayi8LEQVUtJn3QNZ0TciWLP4NKxNMfcgzQdzcqZCCLqDBZRDnjcsHWFGgSC&yr=2022-2023&eKhayi8LEQwp4NKxN+CfCh+3qOVUtJn3QNZ0TciWLP4",

    # RMIT Collaboration
    "https://admissions.bits-pilani.ac.in/BITS-RMIT/downloads/BITS-RMIT_Admissions_details.pdf?23062025",
    "https://admissions.bits-pilani.ac.in/BITS-RMIT/downloads/BITS-RMIT_Fee_details.pdf?23062025",
    "https://admissions.bits-pilani.ac.in/BITS-RMIT/downloads/BITS-RMIT_FAQs.pdf?23062025",
    "https://admissions.bits-pilani.ac.in/BITS-RMIT/downloads/Application_BITS-RMIT_Higher_Education_Academy.pdf?23062025",
    "https://admissions.bits-pilani.ac.in/BITS-RMIT/downloads/BITS-RMIT_Application_Guide.pdf?23062025",
    "https://admissions.bits-pilani.ac.in/BITS-RMIT/downloads/BITS-RMIT_AcademicCalendar.pdf?05072025",
    "https://admissions.bits-pilani.ac.in/BITS-RMIT/downloads/ICICI_Bank_StudentEcosystemProposal_BITS-RMIT_HEA.pdf?06012025",

    # Board Toppers
    "https://admissions.bits-pilani.ac.in/BT/downloads/Board_Toppers_Admissions.pdf",

    # UB (Buffalo) Collaboration
    "https://admissions.bits-pilani.ac.in/BITS-UB/downloads/Brochure_BITS-ub_Admissions_details.pdf?06012025",
    "https://admissions.bits-pilani.ac.in/BITS-UB/downloads/BITS-UB_Fee_Details_AY_2025-2026.pdf?06012025",
    "https://admissions.bits-pilani.ac.in/BITS-UB/downloads/BITS-UB_FAQs.pdf?06012025",

    # ISU (Iowa State) Collaboration
    "https://admissions.bits-pilani.ac.in/BITS-ISU/downloads/Brochure_BITS-ISU_Admissions_details.pdf?06012025",
    "https://admissions.bits-pilani.ac.in/BITS-ISU/downloads/BITS-ISU_Fee_details_AY_2025-2026.pdf?06062025",
    "https://admissions.bits-pilani.ac.in/BITS-ISU/downloads/BITS-ISU_FAQs.pdf?06012025",
    "https://admissions.bits-pilani.ac.in/BITS-ISU/downloads/BITS-ISU_AcademicCalendar.pdf?05072025",

    # CSP Collaboration
    "https://admissions.bits-pilani.ac.in/BITS-CSP/downloads/Brochure_BITS-CSP_Admissions_Details.pdf?14062025",
    "https://admissions.bits-pilani.ac.in/BITS-CSP/downloads/BITS-CSP_AcademicCalendar.pdf?05072025",
    "https://admissions.bits-pilani.ac.in/BITS-CSP/downloads/BITS-CSP_FAQs.pdf?14062025",
    "https://admissions.bits-pilani.ac.in/BITS-CSP/downloads/BITS-CSP_Questions_asked_in_Webinar-1.pdf?26062025",
    "https://admissions.bits-pilani.ac.in/BITS-CSP/downloads/BITS-CSP_Webinar_27-JUN-2025.pdf?26062025",
    "https://admissions.bits-pilani.ac.in/BITS-CSP/downloads/BITS-CSP_Fee_details.pdf?16062025",

    # RPI Collaboration
    "https://admissions.bits-pilani.ac.in/BITS-RPI/downloads/Brochure_BITS-RPI_Admissions_details.pdf?17062025",
    "https://admissions.bits-pilani.ac.in/BITS-RPI/downloads/BITS-RPI_Fee_details.pdf?17062025",
    "https://admissions.bits-pilani.ac.in/BITS-RPI/downloads/BITS-RPI_FAQs.pdf?17062025",
    "https://admissions.bits-pilani.ac.in/BITS-RPI/downloads/BITS-RPI_AcademicCalendar.pdf?05072025",

    # ISA (International Students)
    "https://admissions.bits-pilani.ac.in/ISA/downloads/ISA_Brochure.pdf?06012025",
    "https://admissions.bits-pilani.ac.in/ISA/downloads/ISA-2026_Timeline.pdf",
    "https://admissions.bits-pilani.ac.in/ISA/downloads/ISA_Fee_Structure_(2025-26).pdf",
    "https://admissions.bits-pilani.ac.in/ISA/downloads/ISA_SAT_Cut-Offs.pdf?06012025",
    "https://admissions.bits-pilani.ac.in/ISA/ISA_FAQs.html?06012025",
    "https://admissions.bits-pilani.ac.in/ISA/downloads/SWIFT_Letter-2025.pdf"
]

# HTML URLs that are known to be JS-rendered (Playwright needed)
JS_RENDERED_DOMAINS = [
    "admissions.bits-pilani.ac.in",
    "cdn3.digialm.com",
]

def is_js_rendered(url):
    """Check if a URL likely needs JS rendering based on domain."""
    for domain in JS_RENDERED_DOMAINS:
        if domain in url:
            return True
    return False

def extract_text_from_html(html_content, url):
    """Extract clean text from raw HTML using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script, style, nav, footer tags - they're noise
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)
    return clean_text

def load_pdf(url, temp_dir):
    """Download and load a PDF file."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, verify=False, timeout=30)
        response.raise_for_status()

        clean_url = url.split("?")[0].lower()
        filename = clean_url.split("/")[-1]
        file_path = os.path.join(temp_dir, filename)

        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"  Processing PDF: {filename}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"  Loaded {len(docs)} pages from PDF")
        return docs

    except Exception as e:
        print(f"  Failed to load PDF {url}: {str(e)[:120]}")
        return []

def load_html_with_playwright(url):
    """Use Playwright to render JS and extract full page text."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            # Wait a bit extra for any lazy-loaded content
            page.wait_for_timeout(2000)
            html_content = page.content()
            browser.close()

        text = extract_text_from_html(html_content, url)

        if not text or len(text) < 50:
            print(f"  Warning: Very little text extracted from {url}")
            return []

        print(f"  Extracted {len(text)} characters from HTML (JS-rendered)")
        doc = Document(
            page_content=text,
            metadata={"source": url}
        )
        return [doc]

    except Exception as e:
        print(f"  Failed to load HTML (Playwright) {url}: {str(e)[:120]}")
        return []

def load_html_static(url):
    """Load a static HTML page using requests + BeautifulSoup."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, verify=False, timeout=15)
        response.raise_for_status()

        text = extract_text_from_html(response.text, url)

        if not text or len(text) < 50:
            print(f"  Warning: Very little text extracted from {url}")
            return []

        print(f"  Extracted {len(text)} characters from static HTML")
        doc = Document(
            page_content=text,
            metadata={"source": url}
        )
        return [doc]

    except Exception as e:
        print(f"  Failed to load static HTML {url}: {str(e)[:120]}")
        return []

def download_and_load(url, temp_dir):
    """Route URL to appropriate loader based on type."""
    clean_url = url.split("?")[0].lower()

    if clean_url.endswith(".pdf"):
        return load_pdf(url, temp_dir)
    elif is_js_rendered(url):
        return load_html_with_playwright(url)
    else:
        return load_html_static(url)

def ingest_urls():
    """Download, process, and upload documents to Qdrant vector store."""
    all_documents = []

    print(f"\nStarting ingestion for {len(URLS)} URLs")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, url in enumerate(URLS, 1):
            print(f"\n[{i}/{len(URLS)}] {url[:80]}...")
            docs = download_and_load(url, temp_dir)
            if docs:
                all_documents.extend(docs)
                print(f"  -> Added {len(docs)} document(s). Total so far: {len(all_documents)}")
            else:
                print(f"  -> No content loaded, skipping.")

    print("\n" + "=" * 60)

    if not all_documents:
        print("No documents were loaded. Exiting.")
        return

    print(f"\nTotal documents loaded: {len(all_documents)}")

    print("\nSplitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Generated {len(chunks)} chunks")

    print("\nInitializing embedding models...")
    dense_embeddings = FastEmbedEmbeddings(model_name=DENSE_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense-vector": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse-vector": models.SparseVectorParams()
            }
        )

    print("Uploading data to Qdrant (this may take a while)...")
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense-vector",
        sparse_vector_name="sparse-vector",
        force_recreate=True
    )

    print("\nIngestion completed successfully!")
    print(f"Total chunks indexed: {len(chunks)}")

if __name__ == "__main__":
    ingest_urls()