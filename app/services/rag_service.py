"""RAG service using LanceDB for vector storage."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import lancedb
from lancedb.pydantic import LanceModel, Vector
import pyarrow as pa

from app.config import get_settings
from app.services.ollama_client import get_ollama_client
from app.utils.file_parser import chunk_text, parse_file, scan_directory


class Document(LanceModel):
    """Document model for LanceDB."""

    id: str
    content: str
    source: str
    chunk_index: int
    embedding: Vector(768)  # nomic-embed-text dimension
    created_at: str
    rating: Optional[int] = None


class RAGService:
    """RAG service for document indexing and retrieval."""

    def __init__(self):
        self.settings = get_settings()
        self.db_path = self.settings.lancedb_path
        self.embedding_model = self.settings.embedding_model
        self._db: Optional[lancedb.DBConnection] = None
        self._table: Optional[lancedb.table.Table] = None
        self._last_ingest: Optional[datetime] = None

    def _get_db(self) -> lancedb.DBConnection:
        """Get or create database connection."""
        if self._db is None:
            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(self.db_path)
        return self._db

    async def _get_or_create_table(self) -> lancedb.table.Table:
        """Get or create the documents table."""
        db = self._get_db()
        table_name = "documents"

        try:
            return db.open_table(table_name)
        except Exception:
            # Table doesn't exist, create it with schema
            return db.create_table(table_name, schema=Document)

    async def index_chunks(
        self,
        chunks: list[str],
        source: str,
        start_index: int = 0,
    ) -> int:
        """Index text chunks into the vector store.

        Args:
            chunks: List of text chunks to index.
            source: Source file path or identifier.
            start_index: Starting index for chunk numbering.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            return 0

        ollama = get_ollama_client()
        table = await self._get_or_create_table()

        # Generate embeddings in batches
        batch_size = 20
        documents = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = await ollama.embed_batch(batch)

            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                doc = {
                    "id": f"{source}_{start_index + i + j}",
                    "content": chunk,
                    "source": source,
                    "chunk_index": start_index + i + j,
                    "embedding": embedding,
                    "created_at": datetime.utcnow().isoformat(),
                    "rating": None,
                }
                documents.append(doc)

        # Add to table
        if documents:
            table.add(documents)

        return len(documents)

    async def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        chunk_size: int = 500,
    ) -> tuple[int, int, list[str], list[str]]:
        """Ingest documents from a directory.

        Args:
            directory: Path to directory containing documents.
            recursive: Whether to scan subdirectories.
            chunk_size: Size of text chunks.

        Returns:
            Tuple of (files_indexed, total_chunks, files_processed, errors).
        """
        files = scan_directory(directory, recursive=recursive)
        files_processed = []
        errors = []
        total_chunks = 0

        for file_path in files:
            try:
                # Parse file content
                content = parse_file(file_path)

                # Chunk the content
                chunks = chunk_text(content, chunk_size=chunk_size)

                # Index chunks
                indexed = await self.index_chunks(chunks, source=file_path)
                total_chunks += indexed
                files_processed.append(file_path)

            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")

        self._last_ingest = datetime.utcnow()

        return len(files_processed), total_chunks, files_processed, errors

    async def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict]:
        """Search for relevant documents.

        Args:
            query: Search query text.
            k: Number of results to return.

        Returns:
            List of matching documents with content and metadata.
        """
        ollama = get_ollama_client()
        table = await self._get_or_create_table()

        # Generate query embedding
        query_embedding = await ollama.embed(query)

        # Search using vector similarity
        results = (
            table.search(query_embedding)
            .limit(k)
            .to_list()
        )

        return [
            {
                "content": r["content"],
                "source": r["source"],
                "chunk_index": r["chunk_index"],
                "score": r.get("_distance", 0),
            }
            for r in results
        ]

    async def search_by_embedding(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict]:
        """Search using a pre-computed embedding.

        Args:
            query_embedding: Query embedding vector.
            k: Number of results to return.

        Returns:
            List of matching documents.
        """
        table = await self._get_or_create_table()

        results = (
            table.search(query_embedding)
            .limit(k)
            .to_list()
        )

        return [
            {
                "content": r["content"],
                "source": r["source"],
                "chunk_index": r["chunk_index"],
                "score": r.get("_distance", 0),
            }
            for r in results
        ]

    async def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        db = self._get_db()
        try:
            db.drop_table("documents")
        except Exception:
            pass  # Table doesn't exist
        self._table = None

    async def get_stats(self) -> dict:
        """Get statistics about the knowledge base.

        Returns:
            Dictionary with stats including document count, chunk count, etc.
        """
        try:
            table = await self._get_or_create_table()
            count = table.count_rows()

            # Get unique sources
            try:
                df = table.to_pandas()
                unique_sources = df["source"].nunique() if len(df) > 0 else 0
            except Exception:
                unique_sources = 0

            return {
                "total_documents": unique_sources,
                "total_chunks": count,
                "last_ingest": self._last_ingest,
                "embedding_model": self.embedding_model,
            }
        except Exception:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "last_ingest": None,
                "embedding_model": self.embedding_model,
            }

    async def add_rating(
        self,
        prompt: str,
        enhanced_prompt: str,
        rating: int,
        feedback: Optional[str] = None,
    ) -> None:
        """Store a rating for an enhanced prompt.

        Args:
            prompt: Original prompt.
            enhanced_prompt: Enhanced prompt.
            rating: User rating (1-5).
            feedback: Optional feedback text.
        """
        ollama = get_ollama_client()
        table = await self._get_or_create_table()

        # Create embedding for the original prompt
        embedding = await ollama.embed(prompt)

        # Store as a rated sample
        doc = {
            "id": f"rating_{datetime.utcnow().timestamp()}",
            "content": json.dumps({
                "original": prompt,
                "enhanced": enhanced_prompt,
                "feedback": feedback,
            }),
            "source": "rating",
            "chunk_index": 0,
            "embedding": embedding,
            "created_at": datetime.utcnow().isoformat(),
            "rating": rating,
        }

        table.add([doc])


# Singleton instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get singleton RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
