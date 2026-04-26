"""
RAG Retriever — Searches indexed documents for relevant context.
Used by AI modules to enrich prompts with document knowledge.
"""

from typing import Dict, List, Any, Optional
from loguru import logger


class RAGRetriever:
    """Retrieves relevant document chunks from ChromaDB."""

    def __init__(self, config: Dict[str, Any]):
        rag_cfg = config.get("rag", {})
        self.enabled = rag_cfg.get("enabled", False)
        self.persist_dir = rag_cfg.get("chroma_persist_dir", "./data/chroma_store")
        self.top_k = rag_cfg.get("top_k", 5)

        self._collection = None
        self._client = None

    def _get_collection(self):
        """Lazy-load ChromaDB collection."""
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings

                self._client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.persist_dir,
                    anonymized_telemetry=False,
                ))
                self._collection = self._client.get_or_create_collection(
                    name="ws_documents",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.error("Failed to connect to ChromaDB: {0}", str(e))
                raise

        return self._collection

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query (user's question)
            top_k: Number of results to return (default from config)
            document_type: Filter by document type (market_commentary, ips, etc.)

        Returns:
            List of dicts: [{"content": "...", "title": "...", "score": 0.85, ...}]
        """
        if not self.enabled:
            return []

        k = top_k or self.top_k

        try:
            collection = self._get_collection()

            # Build where filter
            where_filter = None
            if document_type:
                where_filter = {"document_type": document_type}

            results = collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter,
            )

            # Parse results
            chunks = []
            if results and results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0

                    # ChromaDB returns distance (lower = more similar for cosine)
                    # Convert to similarity score (1 - distance)
                    score = max(0, 1 - distance)

                    chunks.append({
                        "content": doc,
                        "title": meta.get("title", ""),
                        "document_id": meta.get("document_id", ""),
                        "document_type": meta.get("document_type", ""),
                        "chunk_index": meta.get("chunk_index", ""),
                        "score": round(score, 4),
                    })

            logger.debug("RAG search | query='{0}' | results={1}",
                         query[:60], len(chunks))
            return chunks

        except Exception as e:
            logger.error("RAG search failed: {0}", str(e))
            return []

    # ------------------------------------------------------------------
    # Build Context
    # ------------------------------------------------------------------

    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_type: Optional[str] = None,
        min_score: float = 0.3,
    ) -> str:
        """
        Get relevant context as a formatted string for prompt injection.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            document_type: Filter by type
            min_score: Minimum similarity score to include

        Returns:
            Formatted string of relevant context, or empty string if nothing found
        """
        chunks = self.search(query, top_k, document_type)

        # Filter by minimum score
        relevant = [c for c in chunks if c["score"] >= min_score]

        if not relevant:
            return ""

        lines = ["RELEVANT CONTEXT FROM DOCUMENTS:"]
        for i, chunk in enumerate(relevant):
            lines.append("[Source: {0} (relevance: {1:.0%})]".format(
                chunk["title"], chunk["score"]))
            lines.append(chunk["content"])
            lines.append("")

        context = "\n".join(lines)
        logger.info("RAG context | {0} relevant chunks | total_chars={1}",
                     len(relevant), len(context))
        return context
