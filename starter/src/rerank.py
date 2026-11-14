"""Cohere Reranker implementation using Cohere API v2 (rerank-v3.5)."""
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any
from dotenv import load_dotenv


class CohereReranker:
    """Reranker using Cohere API v2 (model: rerank-v3.5)."""

    def __init__(self, model: str = "rerank-v3.5", api_key: str | None = None) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is missing from environment / .env file.")
        self.model = model
        self.api_url = "https://api.cohere.com/v2/rerank"

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict[str, Any]]:
        """Rerank documents against the query.
        
        Returns a list of dicts: [{"index": int, "relevance_score": float}, ...]
        ordered by relevance_score descending.
        """
        if not documents:
            return []

        clean_query = query.strip() if query.strip() else " "
        clean_docs = [d if d.strip() else " " for d in documents]
        top_n = min(top_n or len(documents), len(documents))

        payload = {
            "model": self.model,
            "query": clean_query,
            "documents": clean_docs,
            "top_n": top_n
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        for attempt in range(1, 5):
            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("results", [])
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 4:
                    wait_time = 12 * attempt
                    print(f"[Cohere] Rate limited (429). Retrying in {wait_time}s (attempt {attempt}/4)...")
                    time.sleep(wait_time)
                else:
                    raise e
            except Exception as e:
                if attempt < 3:
                    time.sleep(2)
                else:
                    raise e
        return []
