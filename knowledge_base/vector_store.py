import os
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

# Embedding model (384-dimensional, fast and compact ONNX runtime < 200MB RAM)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "fullstack_devdocs"
VECTOR_SIZE = 384

_embeddings = None

def get_embeddings():
    """Singleton getter for lightweight FastEmbed ONNX embeddings."""
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings

def get_qdrant_client() -> tuple[QdrantClient, str]:
    """
    Connect to Qdrant Cloud if credentials exist,
    otherwise cleanly fall back to local disk persistence.
    """
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip().strip('"').strip("'")

    if url and api_key:
        client = QdrantClient(url=url, api_key=api_key)
        mode = f"Cloud ({url})"
    else:
        local_path = os.path.join(os.path.dirname(__file__), "..", "qdrant_db")
        os.makedirs(local_path, exist_ok=True)
        client = QdrantClient(path=local_path)
        mode = f"Local Disk ({local_path})"

    return client, mode

def init_vector_store(collection_name: str = COLLECTION_NAME) -> QdrantVectorStore:
    """Initialize or load the vector store."""
    client, mode = get_qdrant_client()
    embeddings = get_embeddings()

    # Ensure collection exists
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    return vector_store
