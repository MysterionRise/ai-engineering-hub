"""Semantic Search Engine Implementation.

A production-ready semantic search engine that demonstrates:
- Document ingestion and chunking
- Embedding generation
- Vector similarity search
- Result ranking and filtering

This project showcases how to build a real-world search system
using LLM embeddings.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from ai_hub import count_tokens, get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@dataclass
class Document:
    """A document in the search index."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """A search result with relevance score."""

    document: Document
    score: float
    highlights: list[str] = field(default_factory=list)


class SemanticSearchEngine:
    """A semantic search engine using embeddings."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialize the search engine.

        Args:
            embedding_model: OpenAI embedding model to use.
            chunk_size: Maximum tokens per chunk.
            chunk_overlap: Token overlap between chunks.
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = OpenAI()
        self.documents: dict[str, Document] = {}

        logger.info(
            "search_engine_initialized",
            model=embedding_model,
            chunk_size=chunk_size,
        )

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk.

        Returns:
            List of text chunks.
        """
        # Simple sentence-based chunking
        sentences = text.replace("\n", " ").split(". ")
        chunks = []
        current_chunk: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = count_tokens(sentence, "gpt-4o")

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
                # Keep overlap
                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_tokens = count_tokens(s, "gpt-4o")
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        return chunks

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity."""
        import math

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        chunk: bool = True,
    ) -> list[str]:
        """Add a document to the search index.

        Args:
            content: Document content.
            metadata: Optional metadata.
            chunk: Whether to chunk the document.

        Returns:
            List of document IDs added.
        """
        metadata = metadata or {}
        doc_ids = []

        if chunk:
            chunks = self._chunk_text(content)
            logger.info("document_chunked", chunks=len(chunks))
        else:
            chunks = [content]

        for i, chunk_text in enumerate(chunks):
            doc_id = self._generate_id(chunk_text)
            embedding = self._get_embedding(chunk_text)

            doc = Document(
                id=doc_id,
                content=chunk_text,
                metadata={**metadata, "chunk_index": i},
                embedding=embedding,
            )

            self.documents[doc_id] = doc
            doc_ids.append(doc_id)

        logger.info("documents_added", count=len(doc_ids))
        return doc_ids

    def add_documents_batch(
        self,
        documents: list[dict[str, Any]],
    ) -> int:
        """Add multiple documents in batch.

        Args:
            documents: List of dicts with 'content' and optional 'metadata'.

        Returns:
            Number of document chunks added.
        """
        total_added = 0
        for doc in documents:
            ids = self.add_document(
                content=doc["content"],
                metadata=doc.get("metadata"),
            )
            total_added += len(ids)
        return total_added

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for relevant documents.

        Args:
            query: Search query.
            top_k: Number of results to return.
            min_score: Minimum similarity score.
            filter_metadata: Optional metadata filters.

        Returns:
            List of search results.
        """
        if not self.documents:
            return []

        query_embedding = self._get_embedding(query)

        # Score all documents
        scored: list[tuple[float, Document]] = []
        for doc in self.documents.values():
            # Apply metadata filter
            if filter_metadata:
                match = all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue

            if doc.embedding:
                score = self._cosine_similarity(query_embedding, doc.embedding)
                if score >= min_score:
                    scored.append((score, doc))

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build results
        results = []
        for score, doc in scored[:top_k]:
            # Generate highlights (simple keyword matching)
            query_words = set(query.lower().split())
            highlights = []
            for sentence in doc.content.split(". "):
                if any(word in sentence.lower() for word in query_words):
                    highlights.append(sentence.strip())

            results.append(
                SearchResult(
                    document=doc,
                    score=score,
                    highlights=highlights[:3],
                )
            )

        logger.info("search_complete", query=query[:50], results=len(results))
        return results

    def save_index(self, path: Path) -> None:
        """Save the search index to disk.

        Args:
            path: Path to save the index.
        """
        data = {
            "model": self.embedding_model,
            "documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "embedding": doc.embedding,
                }
                for doc in self.documents.values()
            ],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

        logger.info("index_saved", path=str(path), documents=len(self.documents))

    def load_index(self, path: Path) -> None:
        """Load a search index from disk.

        Args:
            path: Path to the index file.
        """
        with open(path) as f:
            data = json.load(f)

        self.embedding_model = data["model"]
        self.documents = {}

        for doc_data in data["documents"]:
            doc = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                embedding=doc_data["embedding"],
            )
            self.documents[doc.id] = doc

        logger.info("index_loaded", path=str(path), documents=len(self.documents))

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "total_documents": len(self.documents),
            "embedding_model": self.embedding_model,
            "total_tokens": sum(count_tokens(d.content, "gpt-4o") for d in self.documents.values()),
        }


def main() -> None:
    """Demonstrate the semantic search engine."""
    print("=" * 60)
    print("Semantic Search Engine Demo")
    print("=" * 60)

    # Initialize
    engine = SemanticSearchEngine()

    # Add sample documents
    sample_docs = [
        {
            "content": """Python is a high-level, interpreted programming language known for its
            clear syntax and readability. It was created by Guido van Rossum and first released
            in 1991. Python supports multiple programming paradigms including procedural,
            object-oriented, and functional programming.""",
            "metadata": {"topic": "programming", "language": "python"},
        },
        {
            "content": """Machine learning is a subset of artificial intelligence that enables
            systems to learn and improve from experience without being explicitly programmed.
            It focuses on developing algorithms that can access data and use it to learn
            for themselves. Common applications include image recognition, natural language
            processing, and recommendation systems.""",
            "metadata": {"topic": "ai", "subtopic": "machine-learning"},
        },
        {
            "content": """JavaScript is a programming language that enables interactive web pages.
            It was created by Brendan Eich in 1995 and has become one of the core technologies
            of the World Wide Web. JavaScript is used for both client-side and server-side
            development with frameworks like React, Vue, and Node.js.""",
            "metadata": {"topic": "programming", "language": "javascript"},
        },
    ]

    print("\n--- Adding Documents ---")
    total = engine.add_documents_batch(sample_docs)
    print(f"Added {total} document chunks")

    stats = engine.get_stats()
    print(f"Index stats: {stats}")

    # Search examples
    queries = [
        "Who created Python?",
        "What is machine learning used for?",
        "web development frameworks",
    ]

    for query in queries:
        print(f"\n--- Search: '{query}' ---")
        results = engine.search(query, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (score: {result.score:.3f}):")
            print(f"  Content: {result.document.content[:100]}...")
            if result.highlights:
                print(f"  Highlights: {result.highlights[0][:80]}...")

    # Filtered search
    print("\n--- Filtered Search (topic=programming) ---")
    results = engine.search(
        "programming language features",
        filter_metadata={"topic": "programming"},
    )
    for result in results:
        print(f"  - {result.document.metadata.get('language', 'unknown')}: {result.score:.3f}")


if __name__ == "__main__":
    main()
