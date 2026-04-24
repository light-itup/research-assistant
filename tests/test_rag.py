"""Test RAG module."""
import os
os.environ.pop('SSL_CERT_FILE', None)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
from src.rag import (
    load_documents,
    split_documents,
    create_embedder,
    create_vector_index,
    query_index,
)

# Test data path
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"


def test_rag_pipeline():
    """Test the complete RAG pipeline."""
    print("=" * 50)
    print("Testing RAG Pipeline")
    print("=" * 50)

    # 1. Check knowledge base
    files = list(TEST_DATA_DIR.glob("*"))
    print(f"\n[1] Knowledge base files: {len(files)} found")
    for f in files:
        print(f"   - {f.name}")

    if not files:
        print("\n⚠️  No files in knowledge base. Add some documents to test.")
        print("   Supported: .txt, .md, .pdf, .docx, .pptx, .html, .json, .yaml")
        return

    # 2. Load documents
    print("\n[2] Loading documents...")
    docs = load_documents(directory=str(TEST_DATA_DIR))
    print(f"   Loaded {len(docs)} documents")
    for doc in docs[:2]:
        text = doc.text[:100].replace("\n", " ")
        print(f"   - {text}...")

    # 3. Split documents
    print("\n[3] Splitting documents...")
    chunks = split_documents(docs, chunk_size=256, chunk_overlap=50)
    print(f"   Created {len(chunks)} chunks")

    # 4. Create embedder
    print("\n[4] Creating embedder...")
    embedder = create_embedder()
    print(f"   Model: {embedder.model_name}")

    # 5. Create vector index
    print("\n[5] Creating vector index...")
    index = create_vector_index(docs, embed_model=embedder, store_locally=False)
    print(f"   Index created with {len(index.docstore.docs)} nodes")

    # 6. Query test
    print("\n[6] Testing query...")
    results = query_index(index, "What is this about?", top_k=2)
    print(f"   Found {len(results)} results")
    for i, r in enumerate(results[:2]):
        text = r.text[:150].replace("\n", " ")
        print(f"   [{i+1}] {text}...")

    print("\n" + "=" * 50)
    print("RAG Pipeline Test Complete!")
    print("=" * 50)


if __name__ == "__main__":
    test_rag_pipeline()
