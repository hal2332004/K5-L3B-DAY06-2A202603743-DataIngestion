"""Store and search vectors using Qdrant Cloud with metadata filtering."""
import os
import uuid
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from .models import Chunk


class QdrantVectorStore:
    """Qdrant Cloud implementation supporting vector search and payload filtering."""

    def __init__(self, database_url: str | None = None, dimensions: int = 1536) -> None:
        # Load environment
        root_env = Path(__file__).resolve().parents[2] / ".env"
        starter_env = Path(__file__).resolve().parents[2] / "starter" / ".env"
        if root_env.exists():
            load_dotenv(root_env)
        elif starter_env.exists():
            load_dotenv(starter_env)
        else:
            load_dotenv()

        self.url = database_url or os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        if not self.url or not self.api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")

        self.dimensions = dimensions
        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=20)

    def create_schema(self, strategy: str, dimensions: int | None = None) -> None:
        """Create or recreate collection for a strategy and build payload indexes for filtering."""
        col_name = f"strategy_{strategy.lower()}"
        dims = dimensions or self.dimensions
        if self.client.collection_exists(col_name):
            print(f"[Store] Recreating existing collection: {col_name}")
            self.client.delete_collection(col_name)

        print(f"[Store] Creating collection: {col_name} (dimensions={dims}, metric=Cosine)")
        self.client.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=dims,
                distance=models.Distance.COSINE
            )
        )

        # Build payload indexes required for metadata filtering
        self.client.create_payload_index(
            collection_name=col_name,
            field_name="chapter",
            field_schema=models.PayloadSchemaType.INTEGER
        )
        self.client.create_payload_index(
            collection_name=col_name,
            field_name="section",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]],
               batch_size: int = 100) -> None:
        """Validate lengths/dimensions and write deterministic IDs to avoid duplicates."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors.")

        col_name = f"strategy_{strategy.lower()}"
        total = len(chunks)

        for i in range(0, total, batch_size):
            chunk_batch = chunks[i:i + batch_size]
            vec_batch = vectors[i:i + batch_size]

            points: list[models.PointStruct] = []
            for ch, vec in zip(chunk_batch, vec_batch):
                # Deterministic UUID from chunk_id ensures idempotent re-runs
                pt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, ch.chunk_id))
                payload = {
                    "chunk_id": ch.chunk_id,
                    "document_id": ch.document_id,
                    "source": ch.source,
                    "page": ch.page,
                    "text": ch.text,
                    **ch.metadata
                }
                points.append(models.PointStruct(id=pt_id, vector=vec, payload=payload))

            self.client.upsert(collection_name=col_name, points=points)

        print(f"[Store] Successfully indexed {total} points into {col_name}")

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Rank by cosine similarity, optionally applying metadata filters."""
        col_name = f"strategy_{strategy.lower()}"

        query_filter = None
        if filters:
            conditions = []
            for key, val in filters.items():
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=val))
                )
            if conditions:
                query_filter = models.Filter(must=conditions)

        res = self.client.query_points(
            collection_name=col_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )

        results: list[dict[str, Any]] = []
        for pt in res.points:
            payload = pt.payload or {}
            results.append({
                "chunk_id": payload.get("chunk_id", str(pt.id)),
                "text": payload.get("text", ""),
                "score": float(pt.score),
                "page": payload.get("page", 0),
                "source": payload.get("source", ""),
                "metadata": payload
            })

        return results


# Backward compatibility alias
PgVectorStore = QdrantVectorStore
