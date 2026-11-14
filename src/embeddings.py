"""Embeddings provider implementations:
- OpenAIEmbedder (text-embedding-3-small, 1536d) for Strategies A, B, C, D
- JinaLateChunker (jina-embeddings-v3, 1024d with late_chunking=True) for Strategy E
"""
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Protocol
from dotenv import load_dotenv
from openai import OpenAI


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, question: str) -> list[float]:
        ...


class OpenAIEmbedder:
    """OpenAI API Embedder (text-embedding-3-small, 1536 dimensions)."""

    def __init__(self, model_name: str = "text-embedding-3-small", dimensions: int = 1536) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing from environment / .env file.")
        
        self.model_name = model_name
        self.dimensions = dimensions
        self.client = OpenAI(api_key=api_key)

    def embed_documents(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            clean_batch = [t if t.strip() else " " for t in batch]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=clean_batch,
                dimensions=self.dimensions
            )
            batch_vectors = [item.embedding for item in response.data]
            all_vectors.extend(batch_vectors)

        return all_vectors

    def embed_query(self, question: str) -> list[float]:
        clean_q = question.strip() if question.strip() else " "
        response = self.client.embeddings.create(
            model=self.model_name,
            input=clean_q,
            dimensions=self.dimensions
        )
        return response.data[0].embedding


class JinaLateChunker:
    """Jina AI Embeddings v3 with native Late Chunking (1024 dimensions)."""

    def __init__(self, model_name: str = "jina-embeddings-v3", dimensions: int = 1024) -> None:
        load_dotenv()
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY is missing from environment / .env file.")
        self.model_name = model_name
        self.dimensions = dimensions
        self.api_url = "https://api.jina.ai/v1/embeddings"

    def embed_late_chunking_page(self, chunk_texts: list[str]) -> list[list[float]]:
        """Send all chunks belonging to a document/page simultaneously with late_chunking=True."""
        if not chunk_texts:
            return []

        clean_chunks = [t if t.strip() else " " for t in chunk_texts]
        payload = {
            "model": self.model_name,
            "dimensions": self.dimensions,
            "late_chunking": True,
            "input": clean_chunks
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            vectors = [item["embedding"] for item in data["data"]]
            return vectors

    def embed_query(self, question: str) -> list[float]:
        """Embed search query with retrieval.query LoRA adapter."""
        clean_q = question.strip() if question.strip() else " "
        payload = {
            "model": self.model_name,
            "dimensions": self.dimensions,
            "task": "retrieval.query",
            "input": [clean_q]
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["data"][0]["embedding"]


# Backward compatibility
MyEmbedder = OpenAIEmbedder
