"""
Loads PDFs from developer documentation, chunks them, and stores vectors in MongoDB.
"""

import os
import certifi
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration
DB_NAME = "sc_dev_docs"
COLLECTION_NAME = "dev_docs"
INDEX_NAME = "docs"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_pdf_files(docs_dir: str) -> list[Path]:
    docs_path = Path(docs_dir)
    pdf_files = sorted(docs_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    return pdf_files


def load_and_chunk_pdfs(pdf_files: list[Path]) -> list:
    """Load PDFs and split into chunks."""
    all_documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            documents = loader.load()

            # Add source metadata
            for doc in documents:
                doc.metadata["source_file"] = pdf_path.name

            # Split documents into chunks
            chunks = text_splitter.split_documents(documents)
            all_documents.extend(chunks)
            print(f"  -> Created {len(chunks)} chunks")

        except Exception as e:
            print(f"  -> Error processing {pdf_path.name}: {e}")

    print(f"\nTotal chunks created: {len(all_documents)}")
    return all_documents


def setup_mongodb_collection():
    """Set up MongoDB collection for vector storage."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    db = client[DB_NAME]

    # Create collection if it doesn't exist
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        # Clear existing documents for fresh ingestion
        db[COLLECTION_NAME].delete_many({})
        print(f"Cleared existing documents in: {COLLECTION_NAME}")

    return client, db[COLLECTION_NAME]


def create_vector_store(collection, documents: list):
    """Create embeddings and store in MongoDB Atlas Vector Search."""
    if not documents:
        raise ValueError("create_vector_store called with empty documents list")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    print("\nCreating embeddings and storing in MongoDB...")

    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME
    )

    print(f"Successfully stored {len(documents)} document chunks in MongoDB")
    return vector_store


def print_vector_search_index_instructions():
    """Print instructions for creating the vector search index in MongoDB Atlas."""
    print("\n" + "=" * 70)
    print("IMPORTANT: Create Vector Search Index in MongoDB Atlas")
    print("=" * 70)
    print("""
To enable vector search, create an index in MongoDB Atlas with this definition:

1. Go to MongoDB Atlas → Your Cluster → Atlas Search → Create Search Index
2. Select "JSON Editor" and use this configuration:

{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}

3. Set the index name to: naive
4. Select database: rag_playbook
5. Select collection: naive_rag

After creating the index, wait for it to become "Active" before running queries.
""")
    print("=" * 70)


def main():
    """Main ingestion pipeline."""
    print("=" * 50)
    print("Dev Docs - Ingestion Pipeline")
    print("=" * 50)

    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    docs_dir = Path(__file__).resolve().parent / "docs"

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    # Step 1: Get PDF files
    pdf_files = get_pdf_files(docs_dir)

    if not pdf_files:
        raise ValueError("No PDF files found in docs directory")

    # Step 2: Load and chunk PDFs
    documents = load_and_chunk_pdfs(pdf_files)

    if not documents:
        raise ValueError("No document chunks produced by load_and_chunk_pdfs; check your PDFs and chunking settings")
    # Step 3: Setup MongoDB
    client, collection = setup_mongodb_collection()

    try:
        # Step 4: Create embeddings and store in vector database
        vector_store = create_vector_store(collection, documents)
    
        print("\n✅ Ingestion complete!")
        print(f"   Database: {DB_NAME}")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Documents stored: {len(documents)}")

        # Print instructions for creating the vector search index
        print_vector_search_index_instructions()

    finally:
        client.close()


if __name__ == "__main__":
    main()
