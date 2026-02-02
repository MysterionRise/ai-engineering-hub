"""Simple RAG (Retrieval-Augmented Generation) Example.

This example demonstrates a basic RAG pattern using ChromaDB for vector storage
and semantic search to augment LLM responses with relevant context.

Features demonstrated:
- Document chunking and embedding
- Vector storage with ChromaDB
- Semantic similarity search
- Context-augmented generation
"""

import hashlib
from typing import Any

from ai_hub import Message, OpenAIProvider, count_tokens, get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


# Sample knowledge base (in production, this would come from files/databases)
KNOWLEDGE_BASE = [
    {
        "title": "Python Basics",
        "content": """Python is a high-level, interpreted programming language known for its
        clear syntax and readability. It was created by Guido van Rossum and first released
        in 1991. Python supports multiple programming paradigms including procedural,
        object-oriented, and functional programming. It has a large standard library and
        an active community that contributes thousands of third-party packages.""",
    },
    {
        "title": "Python Data Types",
        "content": """Python has several built-in data types: integers (int), floating-point
        numbers (float), strings (str), booleans (bool), lists, tuples, dictionaries (dict),
        and sets. Lists are mutable ordered sequences, while tuples are immutable. Dictionaries
        store key-value pairs and are highly optimized for lookups. Sets store unique elements
        and support mathematical set operations.""",
    },
    {
        "title": "Python Functions",
        "content": """Functions in Python are defined using the 'def' keyword. They can accept
        positional arguments, keyword arguments, *args for variable positional arguments, and
        **kwargs for variable keyword arguments. Python functions are first-class objects,
        meaning they can be passed as arguments, returned from other functions, and assigned
        to variables. Lambda functions provide a way to create small anonymous functions.""",
    },
    {
        "title": "Python Classes",
        "content": """Python supports object-oriented programming with classes. Classes are
        defined using the 'class' keyword. The __init__ method is the constructor. Python
        supports single and multiple inheritance. Special methods (dunder methods) like
        __str__, __repr__, and __eq__ customize class behavior. Python uses duck typing,
        meaning an object's suitability is determined by its methods and properties rather
        than its type.""",
    },
    {
        "title": "Python Error Handling",
        "content": """Python uses try-except blocks for error handling. You can catch specific
        exceptions or use a bare except clause. The 'finally' block runs regardless of whether
        an exception occurred. The 'else' block runs if no exception was raised. Custom
        exceptions can be created by inheriting from the Exception class. The 'raise' keyword
        is used to raise exceptions manually.""",
    },
]


class SimpleRAG:
    """Simple RAG implementation using in-memory vector storage."""

    def __init__(self, embedding_model: str = "text-embedding-3-small") -> None:
        """Initialize the RAG system.

        Args:
            embedding_model: OpenAI embedding model to use.
        """
        self.provider = OpenAIProvider()
        self.embedding_model = embedding_model
        self.documents: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []

        # Try to use ChromaDB if available, otherwise use simple cosine similarity
        try:
            import chromadb

            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"},
            )
            self.use_chroma = True
            logger.info("using_chromadb")
        except ImportError:
            self.use_chroma = False
            logger.info("chromadb_not_available_using_simple_similarity")

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        from openai import OpenAI

        client = OpenAI()
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def add_documents(self, documents: list[dict[str, str]]) -> None:
        """Add documents to the knowledge base.

        Args:
            documents: List of documents with 'title' and 'content' keys.
        """
        for doc in documents:
            doc_id = hashlib.md5(doc["content"].encode()).hexdigest()[:8]
            combined_text = f"{doc['title']}\n{doc['content']}"

            if self.use_chroma:
                embedding = self._get_embedding(combined_text)
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[combined_text],
                    metadatas=[{"title": doc["title"]}],
                )
            else:
                # Simple in-memory storage
                embedding = self._get_embedding(combined_text)
                self.documents.append({"id": doc_id, "text": combined_text, "title": doc["title"]})
                self.embeddings.append(embedding)

            logger.info("document_added", title=doc["title"], id=doc_id)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search for relevant documents.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of relevant documents with scores.
        """
        query_embedding = self._get_embedding(query)

        if self.use_chroma:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            return [
                {
                    "text": doc,
                    "title": meta["title"],
                    "score": 1 - dist,  # Convert distance to similarity
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        else:
            # Simple similarity search
            scores = [(self._cosine_similarity(query_embedding, emb), i) for i, emb in enumerate(self.embeddings)]
            scores.sort(reverse=True)

            return [
                {
                    "text": self.documents[i]["text"],
                    "title": self.documents[i]["title"],
                    "score": score,
                }
                for score, i in scores[:top_k]
            ]

    def query(self, question: str, top_k: int = 3) -> str:
        """Query the RAG system.

        Args:
            question: User question.
            top_k: Number of context documents to retrieve.

        Returns:
            Generated answer.
        """
        # Retrieve relevant documents
        results = self.search(question, top_k=top_k)
        logger.info("documents_retrieved", count=len(results))

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"[Source {i}: {doc['title']}]\n{doc['text']}")

        context = "\n\n".join(context_parts)

        # Check token count
        context_tokens = count_tokens(context)
        logger.info("context_tokens", tokens=context_tokens)

        # Generate answer
        messages = [
            Message.system(
                "You are a helpful assistant. Answer questions based on the provided context. "
                "If the context doesn't contain relevant information, say so. "
                "Cite your sources by referring to [Source N] when using information."
            ),
            Message.user(f"Context:\n{context}\n\nQuestion: {question}"),
        ]

        response = self.provider.complete(messages, temperature=0.3)
        return response.content or ""


def main() -> None:
    """Run the RAG example."""
    print("=" * 60)
    print("Simple RAG Example")
    print("=" * 60)

    # Initialize RAG system
    rag = SimpleRAG()

    # Add knowledge base
    print("\n--- Adding Documents ---")
    rag.add_documents(KNOWLEDGE_BASE)
    print(f"Added {len(KNOWLEDGE_BASE)} documents")

    # Example queries
    queries = [
        "How do I handle errors in Python?",
        "What are the different data types in Python?",
        "How do I create a class with inheritance?",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        answer = rag.query(query)
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
