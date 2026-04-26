"""
RAG Indexer — Indexes documents into ChromaDB for vector search.
Splits documents into chunks, generates embeddings, and stores them.
"""

import os
import re
from typing import Dict, List, Any, Optional
from loguru import logger


class RAGIndexer:
    """Indexes documents into ChromaDB for retrieval."""

    def __init__(self, config: Dict[str, Any]):
        rag_cfg = config.get("rag", {})
        self.enabled = rag_cfg.get("enabled", False)
        self.persist_dir = rag_cfg.get("chroma_persist_dir", "./data/chroma_store")
        self.embedding_model = rag_cfg.get("embedding_model", "all-MiniLM-L6-v2")
        self.chunk_size = rag_cfg.get("chunk_size", 500)
        self.chunk_overlap = rag_cfg.get("chunk_overlap", 50)

        self._collection = None
        self._client = None

    def _get_collection(self):
        """Lazy-load ChromaDB collection."""
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings

                # Ensure persist directory exists
                os.makedirs(self.persist_dir, exist_ok=True)

                self._client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.persist_dir,
                    anonymized_telemetry=False,
                ))
                self._collection = self._client.get_or_create_collection(
                    name="ws_documents",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("ChromaDB collection initialized at {0}", self.persist_dir)
            except ImportError:
                logger.error("chromadb not installed. Run: pip install chromadb")
                raise
            except Exception as e:
                logger.error("Failed to initialize ChromaDB: {0}", str(e))
                raise

        return self._collection

    # ------------------------------------------------------------------
    # Text Chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks by word count.
        Each chunk is ~chunk_size words with chunk_overlap word overlap.
        """
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            # Move start forward by (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean text for indexing — remove excessive whitespace, special chars."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove very long sequences of special characters
        text = re.sub(r'[=\-_]{5,}', '', text)
        return text.strip()

    # ------------------------------------------------------------------
    # Index Document
    # ------------------------------------------------------------------

    def index_document(
        self,
        document_id: str,
        document_type: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Index a document into ChromaDB.

        Args:
            document_id: Unique identifier for the document
            document_type: Category (market_commentary, ips, factsheet, policy)
            title: Document title
            content: Plain text content of the document
            metadata: Additional metadata key-value pairs

        Returns:
            Number of chunks indexed
        """
        if not self.enabled:
            logger.warning("RAG is disabled. Set rag.enabled=true in config.yaml")
            return 0

        collection = self._get_collection()
        cleaned = self._clean_text(content)
        chunks = self._chunk_text(cleaned)

        if not chunks:
            logger.warning("No chunks generated for document: {0}", document_id)
            return 0

        # Prepare IDs, documents, and metadata for each chunk
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = "{0}_chunk_{1}".format(document_id, i)
            ids.append(chunk_id)
            documents.append(chunk)

            chunk_meta = {
                "document_id": document_id,
                "document_type": document_type,
                "title": title,
                "chunk_index": str(i),
                "total_chunks": str(len(chunks)),
            }
            if metadata:
                chunk_meta.update(metadata)
            metadatas.append(chunk_meta)

        # Upsert (add or update) into ChromaDB
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Indexed document '{0}' ({1}) — {2} chunks",
                     title, document_type, len(chunks))
        return len(chunks)

    # ------------------------------------------------------------------
    # Delete Document
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str):
        """Delete all chunks of a document from the index."""
        if not self.enabled:
            return

        collection = self._get_collection()
        # Get all chunk IDs for this document
        results = collection.get(
            where={"document_id": document_id},
        )
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info("Deleted document '{0}' — {1} chunks removed",
                         document_id, len(results["ids"]))

    # ------------------------------------------------------------------
    # Index from File
    # ------------------------------------------------------------------

    def index_from_pdf(self, file_path: str, document_id: str,
                       document_type: str = "general") -> int:
        """Extract text from PDF and index it."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            title = os.path.basename(file_path)
            return self.index_document(
                document_id=document_id,
                document_type=document_type,
                title=title,
                content=text,
                metadata={"source_file": file_path},
            )
        except Exception as e:
            logger.error("Failed to index PDF '{0}': {1}", file_path, str(e))
            return 0

    def index_from_text_file(self, file_path: str, document_id: str,
                             document_type: str = "general") -> int:
        """Index a plain text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            title = os.path.basename(file_path)
            return self.index_document(
                document_id=document_id,
                document_type=document_type,
                title=title,
                content=content,
                metadata={"source_file": file_path},
            )
        except Exception as e:
            logger.error("Failed to index text file '{0}': {1}", file_path, str(e))
            return 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.enabled:
            return {"enabled": False, "total_chunks": 0}

        try:
            collection = self._get_collection()
            count = collection.count()
            return {
                "enabled": True,
                "total_chunks": count,
                "persist_dir": self.persist_dir,
                "embedding_model": self.embedding_model,
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}
