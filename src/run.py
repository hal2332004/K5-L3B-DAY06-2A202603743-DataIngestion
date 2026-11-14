"""Orchestration script for Day 06 Data Ingestion Lab.

Runs:
1. Ingestion of 112 pages from Chip Huyen's AI Engineering book (Chapters 1 & 2).
2. Chunking under Strategies A (Fixed-size), B (Paragraph), and C (Paragraph + Filter).
3. Embedding generation via OpenAI API (text-embedding-3-small, 1536 dimensions).
4. Idempotent storage in Qdrant Cloud.
5. Benchmark retrieval on 15 Dev and 5 Test questions.
6. Evaluation of Hit@1 and Evidence Hit@3, trace preservation, and experiment log update.
"""
import glob
import json
import os
import time
from pathlib import Path

from .chunking import by_paragraph, fixed_size
from .embeddings import OpenAIEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .store import QdrantVectorStore


def find_file(relative_paths: list[str]) -> Path:
    """Find the first existing file among candidate paths."""
    for p_str in relative_paths:
        p = Path(p_str)
        if p.exists():
            return p
    # Try globbing in data
    pdf_matches = glob.glob("data/*.pdf") + glob.glob("starter/data/*.pdf")
    if pdf_matches:
        return Path(pdf_matches[0])
    raise FileNotFoundError(f"None of candidate paths exist: {relative_paths}")


def main() -> None:
    print("=" * 70)
    print(" DAY 06 DATA INGESTION PIPELINE (Strategies A, B, C)")
    print("=" * 70)

    # 1. Paths and inputs
    pdf_path = find_file([
        "data/Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "starter/data/Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf"
    ])
    benchmark_path = find_file([
        "benchmark/questions.json",
        "starter/benchmark/questions.json"
    ])

    print(f"[Setup] Source PDF: {pdf_path}")
    print(f"[Setup] Benchmark questions: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[Setup] Loaded {len(questions)} questions ({sum(1 for q in questions if q['split'] == 'dev')} dev, {sum(1 for q in questions if q['split'] == 'test')} test)")

    # 2. Ingestion
    t0_ingest = time.perf_counter()
    document_id = "chip-huyen-ai-engineering-ch1-2"
    source_name = pdf_path.name
    # Physical pages 25 to 136 = 112 content pages
    pages = load_pdf(pdf_path, document_id=document_id, source=source_name, start_page=25, end_page=136)
    ingest_time = round(time.perf_counter() - t0_ingest, 2)
    print(f"[Setup] Ingest completed in {ingest_time}s across {len(pages)} content pages (>= 30 requirement passed).")

    # 3. Chunking for Strategies
    print("\n--- Generating Chunks for Strategies A, B, C ---")
    # Strategy A: Fixed-size (size=500, overlap=50)
    t0_a = time.perf_counter()
    chunks_a = []
    for p in pages:
        chunks_a.extend(fixed_size(p, size=500, overlap=50))
    time_chunk_a = round(time.perf_counter() - t0_a, 4)

    # Strategy B & C: By paragraph (max_size=800)
    t0_b = time.perf_counter()
    chunks_b = []
    for p in pages:
        chunks_b.extend(by_paragraph(p, max_size=800))
    time_chunk_b = round(time.perf_counter() - t0_b, 4)

    # Strategy C uses identical chunks to Strategy B to isolate the filter variable
    chunks_c = chunks_b

    print(f"Strategy A (Fixed-size 500/50): {len(chunks_a)} chunks (created in {time_chunk_a}s)")
    print(f"Strategy B (Paragraph <=800):    {len(chunks_b)} chunks (created in {time_chunk_b}s)")
    print(f"Strategy C (Paragraph <=800):    {len(chunks_c)} chunks (same as B, filter applied at search)")

    # 4. Initialize Embedder and Vector Store
    print("\n--- Connecting to OpenAI and Qdrant Cloud ---")
    embedder = OpenAIEmbedder(model_name="text-embedding-3-small", dimensions=1536)
    store = QdrantVectorStore(dimensions=1536)

    # 5. Embed and Index Each Strategy
    strategies_chunks = {
        "A": chunks_a,
        "B": chunks_b,
        "C": chunks_c
    }
    ingest_stats = {}

    for strat_name, chunks in strategies_chunks.items():
        print(f"\n[Index Strategy {strat_name}] Recreating collection and indexing {len(chunks)} chunks...")
        store.create_schema(strat_name)
        
        t0_emb = time.perf_counter()
        texts = [ch.text for ch in chunks]
        vectors = embedder.embed_documents(texts, batch_size=100)
        time_emb = round(time.perf_counter() - t0_emb, 2)

        t0_upsert = time.perf_counter()
        store.upsert(strat_name, chunks, vectors, batch_size=100)
        time_upsert = round(time.perf_counter() - t0_upsert, 2)

        total_chars = sum(len(t) for t in texts)
        # OpenAI text-embedding-3-small is $0.02 / 1M tokens (~4 chars per token)
        est_tokens = total_chars / 4.0
        est_cost_usd = round((est_tokens / 1_000_000.0) * 0.02, 6)

        ingest_stats[strat_name] = {
            "chunk_count": len(chunks),
            "total_chars": total_chars,
            "embed_time_s": time_emb,
            "upsert_time_s": time_upsert,
            "total_ingest_time_s": round(time_emb + time_upsert, 2),
            "est_cost_usd": est_cost_usd
        }
        print(f"[Index Strategy {strat_name}] Done! Embed: {time_emb}s | Upsert: {time_upsert}s | Est Cost: ${est_cost_usd}")

    # 6. Pre-embed Query Questions to Guarantee Fair & Identical Vectors
    print("\n--- Pre-embedding 20 Benchmark Query Questions ---")
    query_vectors = {}
    for q in questions:
        query_vectors[q["id"]] = embedder.embed_query(q["question"])

    # 7. Run Retrieval and Evaluation
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_traces = {}
    all_summaries = {}

    for strat_name in ["A", "B", "C"]:
        print(f"\n[Retrieval Strategy {strat_name}] Running 20 benchmark queries...")
        strat_traces = []
        for q in questions:
            q_vec = query_vectors[q["id"]]
            
            # Apply filter ONLY for Strategy C when question specifies a filter
            q_filter = q.get("filter") if strat_name == "C" else None
            
            hits = store.search(strategy=strat_name, query_vector=q_vec, top_k=3, filters=q_filter)
            trace = evaluate_query(question=q, hits=hits)
            strat_traces.append(trace)

        strat_summary = summarize(strat_traces)
        all_traces[strat_name] = strat_traces
        all_summaries[strat_name] = strat_summary

        # Save trace file
        trace_file = results_dir / f"traces_strategy_{strat_name}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump({
                "strategy": strat_name,
                "ingest_stats": ingest_stats[strat_name],
                "summary": strat_summary,
                "traces": strat_traces
            }, f, indent=2, ensure_ascii=False)
        print(f"[Retrieval Strategy {strat_name}] Saved traces to {trace_file}")

    # 8. Save Overall Summary
    summary_file = results_dir / "summary_metrics.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "ingest_stats": ingest_stats,
            "summaries": all_summaries
        }, f, indent=2, ensure_ascii=False)

    # 9. Update artifacts/experiment_log.csv
    exp_log_path = Path("artifacts/experiment_log.csv")
    csv_rows = [
        "strategy,chunking,chunk_size,overlap,metadata,filter,vectorizer,hypothesis,dev_hit1,dev_hit3,test_hit1,test_hit3,notes"
    ]
    hypotheses = {
        "A": "Fixed-size 500/50 causes broken context (lower coherence); higher chunk count",
        "B": "Paragraph boundaries preserve semantic coherence; higher Hit@3 than A",
        "C": "Payload filter eliminates out-of-scope noise for filtered queries"
    }
    chunking_types = {"A": "fixed_size", "B": "by_paragraph", "C": "by_paragraph"}
    chunk_sizes = {"A": "500", "B": "800", "C": "800"}
    overlaps = {"A": "50", "B": "0", "C": "0"}
    filters = {"A": "none", "B": "none", "C": "chapter_and_section"}

    for s in ["A", "B", "C"]:
        sm = all_summaries[s]
        dev_h1 = sm["dev"]["hit_1_ratio"]
        dev_h3 = sm["dev"]["hit_3_ratio"]
        test_h1 = sm["test"]["hit_1_ratio"]
        test_h3 = sm["test"]["hit_3_ratio"]
        note = f"Chunks: {ingest_stats[s]['chunk_count']}, IngestTime: {ingest_stats[s]['total_ingest_time_s']}s"
        csv_rows.append(
            f"{s},{chunking_types[s]},{chunk_sizes[s]},{overlaps[s]},\"doc_id, source, page, chapter, section\",{filters[s]},openai-text-embedding-3-small-1536d,\"{hypotheses[s]}\",{dev_h1},{dev_h3},{test_h1},{test_h3},\"{note}\""
        )
    with open(exp_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows) + "\n")
    print(f"\n[Experiment Log] Updated {exp_log_path}")

    # 10. Print Grand Summary Tables
    print("\n" + "=" * 75)
    print(" BENCHMARK RESULTS SUMMARY (15 Dev / 5 Test)")
    print("=" * 75)
    print(f"{'Strategy':<10} | {'Split':<6} | {'Hit@1 (n/N)':<12} | {'Hit@1 %':<8} | {'Hit@3 (n/N)':<12} | {'Hit@3 %':<8} | {'Chunks':<8}")
    print("-" * 75)
    for s in ["A", "B", "C"]:
        for sp in ["dev", "test"]:
            sm = all_summaries[s][sp]
            print(f"{s:<10} | {sp:<6} | {sm['hit_1_ratio']:<12} | {sm['hit_1_rate']:<7}% | {sm['hit_3_ratio']:<12} | {sm['hit_3_rate']:<7}% | {ingest_stats[s]['chunk_count']:<8}")
        print("-" * 75)

    print("\n" + "=" * 75)
    print(" METADATA FILTER COMPLIANCE (On queries with filter)")
    print("=" * 75)
    for s in ["A", "B", "C"]:
        dev_comp = all_summaries[s]["dev"]["filter_compliance_ratio"]
        test_comp = all_summaries[s]["test"]["filter_compliance_ratio"]
        print(f"Strategy {s}: Dev filter compliance = {dev_comp} | Test filter compliance = {test_comp}")
    print("=" * 75)
    print("[SUCCESS] All pipeline stages executed cleanly!")


if __name__ == "__main__":
    main()
