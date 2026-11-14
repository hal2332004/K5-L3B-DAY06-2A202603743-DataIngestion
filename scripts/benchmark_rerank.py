"""Benchmark Strategies A, B, C with and without Cohere Reranker (rerank-v3.5)."""
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tqdm import tqdm
from starter.src.embeddings import OpenAIEmbedder
from starter.src.evaluation import evaluate_query, summarize
from starter.src.rerank import CohereReranker
from starter.src.store import QdrantVectorStore


def main():
    print("=" * 75)
    print(" BENCHMARKING STRATEGIES A, B, C WITH COHERE RERANKER (v3.5)")
    print("=" * 75)

    benchmark_path = Path("benchmark/questions.json")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[Setup] Loaded {len(questions)} questions ({sum(1 for q in questions if q['split'] == 'dev')} dev, {sum(1 for q in questions if q['split'] == 'test')} test)")

    embedder = OpenAIEmbedder(model_name="text-embedding-3-small", dimensions=1536)
    reranker = CohereReranker(model="rerank-v3.5")
    store = QdrantVectorStore(dimensions=1536)

    # Pre-embed queries
    print("\n--- Pre-embedding 20 benchmark queries with OpenAI ---")
    query_vectors = {q["id"]: embedder.embed_query(q["question"]) for q in questions}

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    rerank_results = {}

    for s_name in ["A", "B", "C"]:
        existing_trace_file = results_dir / f"traces_strategy_{s_name}_rerank.json"
        # If A or B already completed successfully without fallback, reuse
        if s_name in ["A", "B"] and existing_trace_file.exists():
            with open(existing_trace_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            rerank_traces = saved_data["traces"]
            rerank_summary = saved_data["summary"]

            # Load base summary
            with open(results_dir / f"traces_strategy_{s_name}.json", "r", encoding="utf-8") as bf:
                base_summary = json.load(bf)["summary"]

            rerank_results[s_name] = {
                "base": base_summary,
                "rerank": rerank_summary
            }
            print(f"[Reuse] Loaded existing rerank traces for Strategy {s_name} (Dev H1: {rerank_summary['dev']['hit_1_ratio']}, Dev H3: {rerank_summary['dev']['hit_3_ratio']})")
            continue

        print(f"\n[Evaluating Strategy {s_name}] Retrieving Top-10 candidates from Qdrant Cloud...")
        base_traces = []
        rerank_traces = []

        for q in tqdm(questions, desc=f"Strategy {s_name}"):
            q_vec = query_vectors[q["id"]]
            q_filter = q.get("filter") if s_name == "C" else None

            # Retrieve top 10 candidates
            candidates = store.search(strategy=s_name, query_vector=q_vec, top_k=10, filters=q_filter)

            # 1. Base (No Rerank): Take top 3
            base_hits = candidates[:3]
            base_trace = evaluate_query(question=q, hits=base_hits)
            base_traces.append(base_trace)

            # 2. With Cohere Rerank: Rerank the 10 candidates down to top 3
            doc_texts = [c.get("text", "") for c in candidates]
            try:
                cohere_res = reranker.rerank(query=q["question"], documents=doc_texts, top_n=3)
                rerank_hits = []
                for cr in cohere_res:
                    orig_idx = cr["index"]
                    candidate_copy = dict(candidates[orig_idx])
                    candidate_copy["original_rank"] = candidate_copy.get("rank", orig_idx + 1)
                    candidate_copy["original_score"] = candidate_copy.get("score")
                    candidate_copy["score"] = cr["relevance_score"]
                    candidate_copy["reranked"] = True
                    rerank_hits.append(candidate_copy)
            except Exception as e:
                print(f"Warning: Rerank failed for {q['id']} ({e}), falling back to base hits.")
                rerank_hits = base_hits

            rerank_trace = evaluate_query(question=q, hits=rerank_hits)
            rerank_traces.append(rerank_trace)

            # Cohere trial tier has 10 requests/minute limit -> sleep 6.2s between calls
            time.sleep(6.2)

        base_summary = summarize(base_traces)
        rerank_summary = summarize(rerank_traces)

        rerank_results[s_name] = {
            "base": base_summary,
            "rerank": rerank_summary
        }

        # Save trace files
        with open(results_dir / f"traces_strategy_{s_name}_rerank.json", "w", encoding="utf-8") as f:
            json.dump({
                "strategy": s_name,
                "model": "rerank-v3.5",
                "summary": rerank_summary,
                "traces": rerank_traces
            }, f, indent=2, ensure_ascii=False)

    # Save summary json
    summary_path = results_dir / "rerank_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rerank_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Rerank metrics saved to {summary_path}")

    # Print Comparison Table
    print("\n" + "=" * 95)
    print(" RERANKER IMPACT BENCHMARK TABLE (A, B, C: Base vs +Cohere Rerank-v3.5)")
    print("=" * 95)
    print(f"{'Strategy':<10} | {'Split':<6} | {'Hit@1 (Base)':<13} | {'Hit@1 (+Rerank)':<15} | {'Delta H@1':<10} | {'Hit@3 (Base)':<13} | {'Hit@3 (+Rerank)':<15} | {'Delta H@3'}")
    print("-" * 95)

    for s in ["A", "B", "C"]:
        for sp in ["dev", "test", "all"]:
            b = rerank_results[s]["base"][sp]
            r = rerank_results[s]["rerank"][sp]

            d_h1 = r["hit_1_rate"] - b["hit_1_rate"]
            d_h3 = r["hit_3_rate"] - b["hit_3_rate"]

            d_h1_str = f"+{d_h1:.1f}%" if d_h1 > 0 else (f"{d_h1:.1f}%" if d_h1 < 0 else "0.0%")
            d_h3_str = f"+{d_h3:.1f}%" if d_h3 > 0 else (f"{d_h3:.1f}%" if d_h3 < 0 else "0.0%")

            b_h1_str = f"{b['hit_1_ratio']} ({b['hit_1_rate']}%)"
            r_h1_str = f"{r['hit_1_ratio']} ({r['hit_1_rate']}%)"

            b_h3_str = f"{b['hit_3_ratio']} ({b['hit_3_rate']}%)"
            r_h3_str = f"{r['hit_3_ratio']} ({r['hit_3_rate']}%)"

            print(f"{s:<10} | {sp:<6} | {b_h1_str:<13} | {r_h1_str:<15} | {d_h1_str:<10} | {b_h3_str:<13} | {r_h3_str:<15} | {d_h3_str}")
        print("-" * 95)


if __name__ == "__main__":
    main()
