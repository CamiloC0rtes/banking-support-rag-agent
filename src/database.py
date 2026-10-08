import glob
import logging
import os

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("blossom_agent.database")

# ------------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------------
BASE_DIR = os.getcwd()
DATA_PATH = os.getenv("DATA_PATH", os.path.join(BASE_DIR, "data"))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(BASE_DIR, "chroma_db"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

os.makedirs(CHROMA_PATH, exist_ok=True)

# ------------------------------------------------------------------
# SINGLETONS
# ------------------------------------------------------------------
_RETRIEVER_INSTANCE = None
_CHROMA_CLIENT = None

# ------------------------------------------------------------------
# CHROMA CLIENT
# ------------------------------------------------------------------
def get_chroma_client():
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        try:
            _CHROMA_CLIENT = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=chromadb.Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            logger.info(f"ChromaDB initialized at {CHROMA_PATH}")
        except Exception as e:
            logger.error(f"Error opening ChromaDB: {e}")
            raise
    return _CHROMA_CLIENT

# ------------------------------------------------------------------
# DOCUMENT FILTERING
# ------------------------------------------------------------------
ALLOWED_DOCS = [
    "Magic Training — Back Office",
    "Magic Training (Member admin)",
    "Magic Phone Banking Training",
    "Personal Banking Training",
    "Login — Security items",
    "Sign Up (Set Up Online Access)",
    "Detailed_Security_Protocol",
]

def _get_tags(text: str) -> str:
    mapping = {
        "password": ["password", "reset", "credential"],
        "lockout": ["lockout", "locked", "attempts", "suspend"],
        "mfa": ["verification", "code", "mfa", "otp", "prompt"],
        "remember_me": ["remember me", "cadence", "session"],
        "username": ["username", "id", "login name"],
    }
    text_lower = text.lower()
    found = [
        tag for tag, keywords in mapping.items()
        if any(k in text_lower for k in keywords)
    ]
    return ", ".join(found) if found else ""

# ------------------------------------------------------------------
# PDF LOADING
# ------------------------------------------------------------------
def _load_pdf_documents():
    logger.info(f"Loading PDFs from {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        logger.warning("DATA_PATH does not exist.")
        return []

    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    logger.info(f"Found PDFs: {pdf_files}")

    documents = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)

        if not any(doc_name in filename for doc_name in ALLOWED_DOCS):
            logger.info(f"Skipping non-whitelisted PDF: {filename}")
            continue

        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for d in docs:
                d.metadata = {
                    "source": filename,
                    "page": d.metadata.get("page", 0) + 1,
                    "tags": _get_tags(d.page_content),
                }

            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {filename}")

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")

    return documents

# ------------------------------------------------------------------
# INGESTION
# ------------------------------------------------------------------
def run_ingestion(force_rebuild: bool = False) -> bool:
    db_file = os.path.join(CHROMA_PATH, "chroma.sqlite3")

    if not force_rebuild and os.path.exists(db_file):
        logger.info("Existing Chroma DB found. Skipping ingestion.")
        return True

    logger.info("Starting ingestion process...")

    documents = _load_pdf_documents()
    if not documents:
        logger.warning("No documents found to ingest.")
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(
            model=os.getenv(
                "EMBEDDING_MODEL_NAME",
                "text-embedding-3-small",
            )
        ),
        client=get_chroma_client(),
        collection_name="blossom_security_v1",
    )

    logger.info(f"Vector store successfully created at {CHROMA_PATH}")
    return True

# ------------------------------------------------------------------
# RETRIEVER
# ------------------------------------------------------------------
_VECTOR_STORE = None


def get_vector_store() -> Chroma:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = Chroma(
            client=get_chroma_client(),
            collection_name="blossom_security_v1",
            embedding_function=OpenAIEmbeddings(
                model=os.getenv(
                    "EMBEDDING_MODEL_NAME",
                    "text-embedding-3-small",
                )
            ),
        )
    return _VECTOR_STORE


def get_active_retriever():
    global _RETRIEVER_INSTANCE

    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = get_vector_store().as_retriever(
            search_kwargs={"k": RETRIEVAL_K}
        )

    return _RETRIEVER_INSTANCE


async def search_with_scores(query: str, k: int = RETRIEVAL_K) -> list:
    """Return [(Document, relevance_score)] sorted by relevance (0..1, higher is better)."""
    return await get_vector_store().asimilarity_search_with_relevance_scores(query, k=k)
