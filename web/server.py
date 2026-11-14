"""Lightweight Python HTTP backend for RAG Chatbot demo.

Endpoints:
- GET /: Serves web/index.html
- POST /api/chat: Performs RAG retrieval on Qdrant Cloud + LLM generation
- GET /api/benchmark: Returns the 20 benchmark questions for easy testing
"""
import http.server
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Add project root to sys.path so we can import starter.src
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from starter.src.embeddings import OpenAIEmbedder
from starter.src.store import QdrantVectorStore

# Load environment
root_env = project_root / ".env"
starter_env = project_root / "starter" / ".env"
if root_env.exists():
    load_dotenv(root_env)
elif starter_env.exists():
    load_dotenv(starter_env)
else:
    load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
embedder = OpenAIEmbedder(model_name="text-embedding-3-small", dimensions=1536)
store = QdrantVectorStore(dimensions=1536)
llm_client = OpenAI(api_key=openai_api_key)

# Load benchmark questions if available
benchmark_file = project_root / "benchmark" / "questions.json"
benchmark_questions = []
if benchmark_file.exists():
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_questions = json.load(f)


class RAGHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "index.html not found")
        elif self.path == "/api/benchmark":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(benchmark_questions, ensure_ascii=False).encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                query = data.get("query", "").strip()
                strategy = data.get("strategy", "C").upper()
                top_k = int(data.get("top_k", 3))
                custom_filter = data.get("filter")

                if not query:
                    self.send_json({"error": "Empty query"}, status=400)
                    return

                # 1. Embed query
                q_vec = embedder.embed_query(query)

                # 2. Retrieve from Qdrant Cloud
                filters = custom_filter if strategy == "C" else None
                hits = store.search(strategy=strategy, query_vector=q_vec, top_k=top_k, filters=filters)

                # 3. Construct context
                context_blocks = []
                for idx, h in enumerate(hits, start=1):
                    p = h.get("page", "?")
                    meta = h.get("metadata", {})
                    sec = meta.get("section", "general")
                    txt = h.get("text", "")
                    context_blocks.append(f"[Excerpt {idx} - Page {p} - Section: {sec}]\n{txt}")

                context_str = "\n\n".join(context_blocks)

                # 4. Generate Answer using OpenAI LLM
                system_prompt = (
                    "You are an expert AI Engineering assistant with authoritative knowledge from Chip Huyen's "
                    "book 'AI Engineering: Building Applications with Foundation Models' (Chapters 1 & 2).\n"
                    "Use ONLY the following retrieved excerpts to answer the user's question accurately.\n"
                    "Always cite the relevant page numbers using [Page X] right next to the corresponding fact.\n"
                    "If the answer cannot be found in the provided excerpts, state clearly that the retrieved context "
                    "does not contain sufficient information to answer."
                )

                user_prompt = f"Retrieved Context:\n{context_str}\n\nUser Question: {query}\n\nAnswer:"

                try:
                    chat_resp = llm_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2
                    )
                    answer = chat_resp.choices[0].message.content
                except Exception as llm_err:
                    answer = f"(Retrieved successfully from Qdrant Cloud, but LLM generation encountered: {llm_err})"

                response_payload = {
                    "query": query,
                    "strategy": strategy,
                    "answer": answer,
                    "hits": hits
                }
                self.send_json(response_payload)

            except Exception as e:
                self.send_json({"error": str(e)}, status=500)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def send_json(self, payload: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def main():
    port = int(os.getenv("PORT", 8000))
    server = http.server.HTTPServer(("0.0.0.0", port), RAGHandler)
    print("=" * 65)
    print(f" RAG CHATBOT SERVER RUNNING AT: http://localhost:{port}")
    print(f" Open http://localhost:{port} in your browser to test!")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()

