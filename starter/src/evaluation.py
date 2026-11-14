"""Evaluate retrieved evidence and preserve per-query traces."""
import re
from typing import Any


def norm_eval(text: str) -> str:
    """Normalize text for evidence matching."""
    text = text.replace("\u00ad", "")
    text = re.sub(r'(\w+)[\u2010\u2013\u2014-]\s+(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and enough gold evidence; keep hit@1 and hit@3 trace."""
    gold_page = question["gold_page"]
    gold_quote_clean = norm_eval(question["evidence_quote"])

    annotated_hits = []
    hit_at_1 = False
    hit_at_3 = False

    for rank, hit in enumerate(hits, start=1):
        hit_page = hit.get("page")
        hit_text_clean = norm_eval(hit.get("text", ""))

        # Hit condition: belongs to gold page AND contains the gold evidence quote
        page_match = (hit_page == gold_page)
        evidence_match = (gold_quote_clean in hit_text_clean)
        is_hit = page_match and evidence_match

        if is_hit:
            if rank == 1:
                hit_at_1 = True
            if rank <= 3:
                hit_at_3 = True

        annotated_hits.append({
            "rank": rank,
            "chunk_id": hit.get("chunk_id"),
            "page": hit_page,
            "score": hit.get("score"),
            "text": hit.get("text"),
            "page_match": page_match,
            "evidence_match": evidence_match,
            "is_hit": is_hit,
            "metadata": hit.get("metadata", {})
        })

    trace = {
        "question_id": question["id"],
        "split": question["split"],
        "type": question.get("type", "unknown"),
        "question": question["question"],
        "expected_answer": question.get("expected_answer", ""),
        "gold_page": gold_page,
        "evidence_quote": question["evidence_quote"],
        "filter": question.get("filter"),
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hits": annotated_hits
    }
    return trace


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    summary = {}
    splits = ["dev", "test", "all"]

    for sp in splits:
        if sp == "all":
            subset = traces
        else:
            subset = [t for t in traces if t["split"] == sp]

        n = len(subset)
        if n == 0:
            continue

        hit1_count = sum(1 for t in subset if t["hit_at_1"])
        hit3_count = sum(1 for t in subset if t["hit_at_3"])

        # Filter queries subset
        filter_subset = [t for t in subset if t.get("filter") is not None]
        filter_n = len(filter_subset)
        filter_compliance_count = 0
        for t in filter_subset:
            # Check if all top-3 hits comply with the filter condition
            q_filter = t["filter"]
            comply_all = True
            for h in t["hits"]:
                meta = h.get("metadata", {})
                for k, v in q_filter.items():
                    if meta.get(k) != v:
                        comply_all = False
                        break
            if comply_all:
                filter_compliance_count += 1

        summary[sp] = {
            "total_queries": n,
            "hit_1_count": hit1_count,
            "hit_1_ratio": f"{hit1_count}/{n}",
            "hit_1_rate": round(hit1_count / n * 100, 2),
            "hit_3_count": hit3_count,
            "hit_3_ratio": f"{hit3_count}/{n}",
            "hit_3_rate": round(hit3_count / n * 100, 2),
            "filter_queries_count": filter_n,
            "filter_compliance_ratio": f"{filter_compliance_count}/{filter_n}" if filter_n > 0 else "N/A"
        }

    return summary
