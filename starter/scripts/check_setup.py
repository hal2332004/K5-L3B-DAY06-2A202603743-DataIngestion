"""Setup verification script for Day 06 Lab.

Tests:
1. Environment variables from .env
2. OpenAI API connection & text-embedding-3-small (1536 dims)
3. Qdrant Cloud cluster connection
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root or starter/
root_env = Path(__file__).resolve().parents[1] / ".env"
starter_env = Path(__file__).resolve().parents[1] / "starter" / ".env"

if root_env.exists():
    load_dotenv(root_env)
    print(f"Loaded config from: {root_env}")
elif starter_env.exists():
    load_dotenv(starter_env)
    print(f"Loaded config from: {starter_env}")
else:
    load_dotenv()
    print("Warning: .env not found in root or starter/")

openai_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")
model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

print("=" * 60)
print(" DAY 06 LAB SETUP CHECK (OpenAI 1536d + Qdrant Cloud)")
print("=" * 60)

errors = []

# 1. Check OpenAI API Key & Embedding
if not openai_key or openai_key.startswith("your_"):
    print("[FAIL] OPENAI_API_KEY is not configured in .env")
    errors.append("OPENAI_API_KEY missing")
else:
    print(f"[...] Testing OpenAI embedding model: {model_name} ({dimensions}d)...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.embeddings.create(
            model=model_name,
            input="Test sentence for AI Engineering Day 06 lab."
        )
        vec = response.data[0].embedding
        if len(vec) == dimensions:
            print(f"[OK] OpenAI Embedding success! Vector length: {len(vec)}")
        else:
            print(f"[WARN] Vector length {len(vec)} != expected {dimensions}")
    except Exception as e:
        print(f"[FAIL] OpenAI API call failed: {e}")
        errors.append(f"OpenAI error: {e}")

# 2. Check Qdrant Cloud
if not qdrant_url or qdrant_url.startswith("https://your-"):
    print("[FAIL] QDRANT_URL is not configured in .env")
    errors.append("QDRANT_URL missing")
elif not qdrant_key or qdrant_key.startswith("your_"):
    print("[FAIL] QDRANT_API_KEY is not configured in .env")
    errors.append("QDRANT_API_KEY missing")
else:
    print(f"[...] Testing Qdrant Cloud connection at: {qdrant_url}...")
    try:
        from qdrant_client import QdrantClient
        qclient = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=10)
        collections = qclient.get_collections()
        print(f"[OK] Qdrant Cloud connected! Existing collections: {len(collections.collections)}")
    except Exception as e:
        print(f"[FAIL] Qdrant Cloud connection failed: {e}")
        errors.append(f"Qdrant error: {e}")

print("=" * 60)
if errors:
    print(f"Setup NOT complete. Please fix the issues above in your .env file.")
    sys.exit(1)
else:
    print("[SUCCESS] All systems verified! Ready to run ingestion and benchmark.")
    sys.exit(0)

