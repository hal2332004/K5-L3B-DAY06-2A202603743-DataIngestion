"""OpenAI Embeddings provider implementation (1536 dimensions)."""
import os
from typing import Protocol
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class OpenAIEmbedder:
    """Real embedder using OpenAI API (text-embedding-3-small, 1536 dimensions)."""

    def __init__(self, model_name: str = "text-embedding-3-small", dimensions: int = 1536) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing from environment / .env file.")
        
        self.model_name = model_name
        self.dimensions = dimensions
        self.client = OpenAI(api_key=api_key)

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed list of texts in batches using OpenAI API."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Replace empty strings with single space to avoid API validation errors
            clean_batch = [t if t.strip() else " " for t in batch]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=clean_batch,
                dimensions=self.dimensions
            )
            # Ensure order matching batch
            batch_vectors = [item.embedding for item in response.data]
            all_vectors.extend(batch_vectors)

        return all_vectors

    def embed_query(self, question: str) -> list[float]:
        """Embed a single query question."""
        clean_q = question.strip() if question.strip() else " "
        response = self.client.embeddings.create(
            model=self.model_name,
            input=clean_q,
            dimensions=self.dimensions
        )
        return response.data[0].embedding


# Alias to satisfy starter contract
MyEmbedder = OpenAIEmbedder
