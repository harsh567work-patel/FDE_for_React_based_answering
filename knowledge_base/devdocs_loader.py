import os
import json
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge_base.vector_store import init_vector_store, get_qdrant_client

DEVDOCS_INDEX_URL = "https://devdocs.io/docs.json"
DOCS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "raw_docs")

# Core Full Stack documentation catalog mapped to DevDocs slugs
FULLSTACK_CATALOG = {
    "fastapi": {"slug": "fastapi", "name": "FastAPI"},
    "react": {"slug": "react", "name": "React"},
    "express": {"slug": "express", "name": "Express"},
    "typescript": {"slug": "typescript", "name": "TypeScript"},
    "node": {"slug": "node", "name": "Node.js"},
    "html": {"slug": "html", "name": "HTML"},
    "css": {"slug": "css", "name": "CSS"},
    "git": {"slug": "git", "name": "Git"},
    "python": {"slug": "python~3.12", "name": "Python"},
    "postgresql": {"slug": "postgresql~18", "name": "PostgreSQL"},
}

def get_devdocs_manifest() -> dict:
    """Fetch the DevDocs manifest containing release and mtime hashes."""
    response = requests.get(DEVDOCS_INDEX_URL, timeout=15)
    response.raise_for_status()
    docs = response.json()
    return {d["slug"]: d for d in docs}

def download_docset(slug: str, manifest: dict | None = None) -> dict:
    """Download or load cached DevDocs documentation database for a slug."""
    os.makedirs(DOCS_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(DOCS_CACHE_DIR, f"{slug.replace('~', '_')}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if manifest is None:
        manifest = get_devdocs_manifest()

    if slug not in manifest:
        raise ValueError(f"Slug '{slug}' not found in DevDocs manifest.")

    mtime = manifest[slug].get("mtime", "")
    url = f"https://documents.devdocs.io/{slug}/db.json?{mtime}"
    print(f"Downloading DevDocs documentation for '{slug}'...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return data

def parse_html_to_markdown(html_content: str) -> str:
    """Clean and parse HTML into structured text with code blocks."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove scripts, styles, and navigation
    for el in soup(["script", "style", "nav", "footer"]):
        el.decompose()

    # Format code blocks clearly
    for pre in soup.find_all("pre"):
        code = pre.get_text()
        pre.replace_with(f"\n```\n{code}\n```\n")

    return soup.get_text(separator="\n", strip=True)

def ingest_docset(tech_key: str, max_pages: int | None = None, batch_size: int = 50) -> int:
    """
    Download, parse, chunk, and embed a specific technology docset into the knowledge base.
    """
    if tech_key not in FULLSTACK_CATALOG:
        raise ValueError(f"Unknown tech '{tech_key}'. Available: {list(FULLSTACK_CATALOG.keys())}")

    target = FULLSTACK_CATALOG[tech_key]
    slug = target["slug"]
    name = target["name"]

    manifest = get_devdocs_manifest()
    # Resolve fallback if specific version slug changed
    if slug not in manifest:
        matches = [s for s in manifest.keys() if s.startswith(tech_key)]
        slug = matches[0] if matches else slug

    db_data = download_docset(slug, manifest)
    pages = list(db_data.items())
    if max_pages:
        pages = pages[:max_pages]

    print(f"Processing {len(pages)} pages for {name} ({slug})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    documents: list[Document] = []
    for path, html in pages:
        clean_text = parse_html_to_markdown(html)
        if not clean_text or len(clean_text) < 40:
            continue

        chunks = splitter.split_text(clean_text)
        for chunk_idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": "devdocs",
                    "technology": name,
                    "slug": slug,
                    "path": path,
                    "chunk": chunk_idx,
                    "url": f"https://devdocs.io/{slug}/{path}"
                }
            )
            documents.append(doc)

    print(f"Created {len(documents)} chunks. Ingesting into Qdrant Vector Store...")
    vector_store = init_vector_store()
    
    total_ingested = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(batch)
        total_ingested += len(batch)
        print(f"  Ingested {total_ingested}/{len(documents)} chunks...")

    return total_ingested

if __name__ == "__main__":
    import sys
    tech = sys.argv[1] if len(sys.argv) > 1 else "fastapi"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(f"Starting test ingestion for '{tech}' (limit: {limit} pages)...")
    ingested = ingest_docset(tech, max_pages=limit)
    print(f"Done! Successfully loaded {ingested} chunks for '{tech}'.")
