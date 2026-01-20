import os
import certifi
from dotenv import load_dotenv
from collections import defaultdict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pymongo import MongoClient

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration
DB_NAME = "sc_dev_docs"
COLLECTION_NAME = "dev_docs"
INDEX_NAME = "docs"


# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 5

# Default hybrid weights
DEFAULT_BM25_WEIGHT = 0.5
DEFAULT_VECTOR_WEIGHT = 0.5
RRF_K = 60

RAG_PROMPT_TEMPLATE = """You are a helpful engineering onboarding assistant that answers questions based on developer documentation from Confluence.

Search Method: Hybrid (combining keyword and semantic search)

Use ONLY the information from the context below to answer the question. If the context doesn't contain enough information, acknowledge what you can answer and what's missing.

Context:
{context}

Question: {question}

Answer:"""

# Global cache for BM25 (expensive to rebuild)
_bm25_cache = {
    "documents": None,
}


def get_mongo_client():
    return MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())


def load_documents_for_bm25() -> list[Document]:
    """Load documents from MongoDB for BM25 indexing."""
    if _bm25_cache["documents"] is not None:
        return _bm25_cache["documents"]

    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]

    try:
        documents = []
        for doc in collection.find():
            text = doc.get("text", "")
            if not text:
                continue

            metadata = {
                "source_file": doc.get("source_file", "Unknown"),
                "page": doc.get("page", 0),
            }

            if "topic_buckets" in doc:
                metadata["topic_buckets"] = doc["topic_buckets"]

            documents.append(Document(
                page_content=text,
                metadata=metadata
            ))

        _bm25_cache["documents"] = documents
        return documents
    finally:
        client.close()


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    weights: list[float],
    k: int = RRF_K
) -> list[Document]:
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    doc_scores = defaultdict(float)
    doc_map = {}

    for list_idx, doc_list in enumerate(result_lists):
        weight = weights[list_idx]
        for rank, doc in enumerate(doc_list):
            doc_key = hash(doc.page_content)
            rrf_score = weight / (k + rank + 1)
            doc_scores[doc_key] += rrf_score

            if doc_key not in doc_map:
                doc_map[doc_key] = doc

    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


def hybrid_search(
    query: str,
    k: int = TOP_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    topic_filter: str = None,
) -> list[Document]:
    """
    Perform hybrid search with optional topic filtering.

    Args:
        topic_filter: Filter by topic (e.g., "backend", "frontend", "testing")
    """
    documents = load_documents_for_bm25()

    if not documents:
        raise ValueError("No documents found. Run ingestion first!")

    # Apply topic filter to BM25 documents if specified
    bm25_docs = documents
    if topic_filter:
        filtered = [
            doc for doc in documents
            if topic_filter.lower() in [t.lower() for t in doc.metadata.get("topic_buckets", [])]
        ]
        if filtered:
            bm25_docs = filtered

    # BM25 search
    bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=k * 3)
    bm25_results = bm25_retriever.invoke(query)

    # Vector search
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    try:
        if topic_filter:
            # Use pre_filter for MongoDB Atlas vector search
            vector_results = vector_store.similarity_search(
                query=query,
                k=k * 3,
                pre_filter={"topic_buckets": topic_filter}
            )
            # Fall back to unfiltered if no results
            if not vector_results:
                vector_results = vector_store.similarity_search(query=query, k=k * 3)
        else:
            vector_results = vector_store.similarity_search(query=query, k=k * 3)
    finally:
        client.close()

    # Combine with RRF
    combined = reciprocal_rank_fusion(
        result_lists=[bm25_results, vector_results],
        weights=[bm25_weight, vector_weight]
    )

    return combined[:k]


def format_context(documents: list) -> str:
    """Format documents into context string."""
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        parts.append(
            f"[Document {i}]\n"
            f"Source: {source}, Page: {page})\n"
            f"Content:\n{doc.page_content}\n"
        )
    return "\n---\n".join(parts)


def create_rag_chain():
    """Create the RAG chain."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )

    return prompt | llm | StrOutputParser()


def generate_answer(
    question: str,
    top_k: int = TOP_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    topic_filter: str = None,
    verbose: bool = False,
) -> dict:
    documents = hybrid_search(
        query=question,
        k=top_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        topic_filter=topic_filter,
    )

    if not documents:
        return {
            "answer": "I couldn't find any relevant information.",
            "sources": [],
            "search_method": f"Hybrid (BM25: {bm25_weight}, Vector: {vector_weight})"
        }

    context = format_context(documents)
    chain = create_rag_chain()
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    search_method = f"Hybrid (BM25: {bm25_weight}, Vector: {vector_weight})"
    if topic_filter:
        search_method += f", Topic: {topic_filter}"

    response = {
        "answer": answer,
        "sources": [
            {
                "file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
            }
            for doc in documents
        ],
        "search_method": search_method,
    }

    if verbose:
        response["retrieved_documents"] = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]

    return response


def interactive_mode():
    """Run interactive Q&A with hybrid search."""
    print("\n" + "=" * 60)
    print("Hybrid Search RAG - Interactive Q&A")
    print("=" * 60)
    print("\n📌 Using existing documents (no ingestion needed)")
    print(f"   Collection: {DB_NAME}.{COLLECTION_NAME}")
    print("\nCommands:")
    print("  weights:0.7,0.3    - Set BM25/Vector weights")
    print("  topic:backend      - Filter by topic (backend, frontend, testing, api, etc.)")
    print("  topic:clear        - Clear topic filter")
    print("  quit               - Exit")
    print("-" * 60)

    print("\n🔄 Loading documents for BM25 index...")
    docs = load_documents_for_bm25()
    print(f"   Loaded {len(docs)} documents")

    # Show available topics
    all_topics = set()
    for doc in docs:
        all_topics.update(doc.metadata.get("topic_buckets", []))
    if all_topics:
        print(f"   Available topics: {', '.join(sorted(all_topics))}")

    bm25_weight = DEFAULT_BM25_WEIGHT
    vector_weight = DEFAULT_VECTOR_WEIGHT
    topic_filter = None

    while True:
        print()
        status = f"⚖️  Weights: BM25={bm25_weight}, Vector={vector_weight}"
        if topic_filter:
            status += f" | 🏷️  Topic: {topic_filter}"
        print(status)
        user_input = input(">> ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if user_input.startswith('weights:'):
            try:
                weights = user_input.split(':')[1].split(',')
                bm25_weight = float(weights[0])
                vector_weight = float(weights[1])
                print(f"✅ Weights updated: BM25={bm25_weight}, Vector={vector_weight}")
            except Exception:
                print("❌ Invalid format. Use 'weights:0.7,0.3'")
            continue

        if user_input.startswith('topic:'):
            topic_value = user_input.split(':')[1].strip()
            if topic_value.lower() == 'clear':
                topic_filter = None
                print("✅ Topic filter cleared")
            else:
                topic_filter = topic_value
                print(f"✅ Topic filter set: {topic_filter}")
            continue

        print("\n🔀 Searching with hybrid retrieval...")
        print("🤖 Generating answer...\n")

        try:
            result = generate_answer(
                user_input,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
                topic_filter=topic_filter,
            )

            print("-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])

            print(f"\n📚 Sources ({len(result['sources'])} documents):")
            for s in result["sources"]:
                print(f"  • {s['file']}, Page: {s['page']})")

            print(f"\n🔀 {result['search_method']}")

        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run example queries or interactive mode."""
    print("=" * 60)
    print("Hybrid Search RAG - Generation")
    print("=" * 60)
    print("\n📌 NO INGESTION REQUIRED")
    print(f"   Reusing collection: {DB_NAME}.{COLLECTION_NAME}")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    examples = [
        {
            "question": "How can I perform barcode scanning on an emulated Android device?",
            "note": "BM25 helps with exact match"
        },
        {
            "question": "Why are code standards important in the ServiceCore fron",
            "note": "Vector helps with semantic meaning"
        },
        {
            "question": "SC Laravel Migration",
            "note": "Hybrid combines terms + meaning"
        },
    ]

    print("\n📋 Example Queries:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex['question'][:45]}... ({ex['note']})")

    print("\n" + "-" * 50)
    choice = input("Enter 1-3 for examples, 'i' for interactive, or your question: ").strip()

    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3']:
        ex = examples[int(choice) - 1]
        print(f"\n📝 Question: {ex['question']}")
        print(f"💡 Note: {ex['note']}")

        print("\n🔀 Retrieving with hybrid search...")
        print("🤖 Generating answer...\n")

        result = generate_answer(ex['question'])

        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])

        print(f"\n📚 Sources:")
        for s in result["sources"]:
            print(f"  • {s['file']}")
    elif choice:
        print(f"\n📝 Question: {choice}")
        print("\n🔀 Retrieving with hybrid search...")

        result = generate_answer(choice)

        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])

        print(f"\n📚 Sources:")
        for s in result["sources"]:
            print(f"  • {s['file']}")
    else:
        print("\n👋 No input. Run again to try!")


if __name__ == "__main__":
    main()
