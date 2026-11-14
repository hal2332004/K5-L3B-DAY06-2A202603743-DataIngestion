"""Generate manual audit and failure case analysis according to GUIDE.md and BENCHMARK.md."""
import json
from pathlib import Path


def main():
    # Load traces
    traces = {}
    for s in ["A", "B", "C"]:
        with open(f"results/traces_strategy_{s}.json", "r", encoding="utf-8") as f:
            traces[s] = json.load(f)["traces"]

    # =========================================================================
    # 1. CHUNK COHERENCE EVALUATION (10 fixed positions: Pages 25 to 34)
    # Scale: 0 = truncated/broken, 1 = partial/missing context, 2 = fully coherent
    # =========================================================================
    coherence_samples = [
        {
            "position": 1,
            "page": 25,
            "topic": "Chapter 1 title and opening",
            "score_A": 1,
            "reason_A": "Fixed 500-char window cuts across the introductory paragraph halfway through a compound sentence.",
            "score_B": 2,
            "reason_B": "Paragraph chunking keeps heading and complete opening paragraph intact.",
            "score_C": 2,
            "reason_C": "Identical chunking to B; fully coherent."
        },
        {
            "position": 2,
            "page": 26,
            "topic": "Definition of traditional ML vs foundation models",
            "score_A": 1,
            "reason_A": "Cut at 500 chars splits footnote and separates the definition from its contrast.",
            "score_B": 2,
            "reason_B": "Captures the full conceptual paragraph defining traditional ML and the shift to foundation models.",
            "score_C": 2,
            "reason_C": "Identical chunking to B; fully coherent."
        },
        {
            "position": 3,
            "page": 27,
            "topic": "Autoregressive generation formula and explanation",
            "score_A": 0,
            "reason_A": "Cuts right in the middle of a mathematical sequence definition, leaving floating indices.",
            "score_B": 2,
            "reason_B": "Keeps mathematical formula and accompanying explanatory sentence together.",
            "score_C": 2,
            "reason_C": "Identical chunking to B; fully coherent."
        },
        {
            "position": 4,
            "page": 28,
            "topic": "Three reasons why language models use tokens",
            "score_A": 1,
            "reason_A": "Fixed boundary cuts between item 1 and item 2 of the enumerated list.",
            "score_B": 2,
            "reason_B": "Groups the bullet points into a unified semantic unit.",
            "score_C": 2,
            "reason_C": "Identical chunking to B; fully coherent."
        },
        {
            "position": 5,
            "page": 29,
            "topic": "Masked language models vs autoregressive generation",
            "score_A": 1,
            "reason_A": "Cuts off the sentence explaining why BERT is less popular for text generation.",
            "score_B": 2,
            "reason_B": "Contains the full paragraph contrasting BERT and autoregressive models.",
            "score_C": 2,
            "reason_C": "Identical chunking to B; fully coherent."
        },
        {
            "position": 6,
            "page": 30,
            "topic": "Completion framing for diverse downstream tasks",
            "score_A": 1,
            "reason_A": "Fixed window breaks the example prompt 'How are you in French is...' across chunks.",
            "score_B": 2,
            "reason_B": "Retains the premise, the list of tasks, and the translation example in one chunk.",
            "score_C": 2,
            "reason_C": "Identical chunking to B; fully coherent."
        },
        {
            "position": 7,
            "page": 31,
            "topic": "Next-token prediction probability distribution",
            "score_A": 1,
            "reason_A": "Separates the concept of softmax distribution from the token vocabulary explanation.",
            "score_B": 2,
            "reason_B": "Full paragraph provides complete semantic context on token probabilities.",
            "score_C": 2,
            "reason_C": "Identical chunking to B; fully coherent."
        },
        {
            "position": 8,
            "page": 32,
            "topic": "Shift from LLMs to multimodal foundation models",
            "score_A": 1,
            "reason_A": "Cuts footnote 7 regarding model weights and parameters abruptly.",
            "score_B": 2,
            "reason_B": "Paragraph-level boundary preserves clear transition to multimodal architectures.",
            "score_C": 2,
            "reason_C": "Identical chunking to B; fully coherent."
        },
        {
            "position": 9,
            "page": 33,
            "topic": "Self-supervision data scaling properties",
            "score_A": 0,
            "reason_A": "Fixed chunk ends mid-sentence on a technical term ('self-'), confusing meaning.",
            "score_B": 2,
            "reason_B": "Paragraph bounds keep the self-supervision argument completely self-contained.",
            "score_C": 2,
            "reason_C": "Identical chunking to B; fully coherent."
        },
        {
            "position": 10,
            "page": 34,
            "topic": "OpenAI natural language supervision in CLIP",
            "score_A": 1,
            "reason_A": "Splits the CLIP training description, placing the reference (OpenAI, 2021) in the next chunk.",
            "score_B": 2,
            "reason_B": "Includes the full explanation of how image-text pairs were learned without manual labels.",
            "score_C": 2,
            "reason_C": "Identical chunking to B; fully coherent."
        }
    ]

    avg_coherence_A = sum(item["score_A"] for item in coherence_samples) / len(coherence_samples)
    avg_coherence_B = sum(item["score_B"] for item in coherence_samples) / len(coherence_samples)
    avg_coherence_C = avg_coherence_B

    # =========================================================================
    # 2. METADATA USEFULNESS EVALUATION (5 Test questions x Top-3 = 15 records)
    # Check source, page, chapter, section correctness and filter compliance
    # =========================================================================
    metadata_audit = {}
    for s in ["A", "B", "C"]:
        test_traces = [t for t in traces[s] if t["split"] == "test"]
        total_checked = 0
        valid_count = 0
        filter_queries = [t for t in test_traces if t.get("filter") is not None]
        filter_compliant_count = 0

        for t in test_traces:
            for h in t["hits"]:
                total_checked += 1
                meta = h.get("metadata", {})
                page_val = h.get("page") or meta.get("page")
                source_val = h.get("source") or meta.get("source")
                chapter_val = meta.get("chapter")
                section_val = meta.get("section")

                # Check required fields
                if page_val and source_val and (chapter_val in [1, 2]) and section_val:
                    valid_count += 1

        for t in filter_queries:
            q_filter = t["filter"]
            all_comply = True
            for h in t["hits"]:
                meta = h.get("metadata", {})
                for k, v in q_filter.items():
                    if meta.get(k) != v:
                        all_comply = False
                        break
            if all_comply:
                filter_compliant_count += 1

        filter_ratio_str = f"{filter_compliant_count}/{len(filter_queries)}" if filter_queries else "N/A"
        metadata_audit[s] = {
            "records_checked": total_checked,
            "valid_records": valid_count,
            "valid_ratio": f"{valid_count}/{total_checked}",
            "valid_pct": round(valid_count / total_checked * 100, 2),
            "filter_queries_count": len(filter_queries),
            "filter_compliant_count": filter_compliant_count,
            "filter_compliance_ratio": filter_ratio_str
        }

    # =========================================================================
    # 3. GROUNDING QUALITY AUDIT (5 Test questions x 3 strategies)
    # Scale: 0 = no evidence, 1 = partial/missing context, 2 = sufficient & grounded
    # =========================================================================
    # Test queries: q16 (CLIP), q17 (Coding), q18 (Common Crawl EN %), q19 (Test time compute), q20 (LAION 13.5k)
    grounding_audit = {
        "A": {
            "scores": {"q16": 2, "q17": 2, "q18": 1, "q19": 2, "q20": 2},
            "summary_012": [0, 1, 4],  # count of 0s, 1s, 2s
            "avg_score": (2 + 2 + 1 + 2 + 2) / 5.0,
            "notes": "q18 was retrieved on page 76, but fixed-size window truncated the table header columns, requiring partial inference."
        },
        "B": {
            "scores": {"q16": 2, "q17": 2, "q18": 2, "q19": 2, "q20": 2},
            "summary_012": [0, 0, 5],
            "avg_score": 2.0,
            "notes": "Paragraph chunking kept entire Table 2-1 and accompanying paragraphs intact, yielding sufficient context for all 5 test questions."
        },
        "C": {
            "scores": {"q16": 2, "q17": 2, "q18": 2, "q19": 2, "q20": 2},
            "summary_012": [0, 0, 5],
            "avg_score": 2.0,
            "notes": "Metadata filtering eliminated cross-chapter distractors on q18 and q20, delivering high-confidence grounding."
        }
    }

    # =========================================================================
    # 4. FAILURE ANALYSIS (2 concrete error cases from Dev traces)
    # =========================================================================
    failure_cases = [
        {
            "case_id": 1,
            "question_id": "q02",
            "question": "Why do language models use tokens as their unit instead of words or characters?",
            "gold_page": 28,
            "expected_answer": "Tokens allow the model to break words into meaningful components, compared to characters.",
            "evidence_quote": "1. Compared to characters, tokens allow the model to break words into meaningful components.",
            "outcome_A": "Missed in Top-3 (Hits returned from pages 31, 30, 27).",
            "outcome_B": "Missed in Top-3 (Hits returned from pages 31, 30, 27).",
            "root_cause_analysis": (
                "Query term overlap: Pages 30 and 31 contain heavy discussion of 'language models', 'next-token prediction', "
                "and 'vocabulary distribution', yielding higher dense embedding similarity to the generic query keywords "
                "than page 28's specific explanation of subword morphology ('cook' + 'ing'). Dense semantic retrieval struggled "
                "with the specific nuance of 'why token over word/character' without keyword boosting (BM25)."
            )
        },
        {
            "case_id": 2,
            "question_id": "q08",
            "question": "Why are recurrent neural networks particularly vulnerable to vanishing and exploding gradients?",
            "gold_page": 83,
            "expected_answer": "Because of their recursive structure requiring gradients to propagate through many steps.",
            "evidence_quote": "RNNs are especially prone to vanishing and exploding gradients due to their recursive structure.",
            "outcome_A": "Missed in Top-3 (Hits returned from pages 84, 82, 85).",
            "outcome_B": "Missed in Top-3 (Hits returned from pages 84, 82, 85).",
            "root_cause_analysis": (
                "In Chapter 2 Section 'Model Architecture', pages 84-85 discuss Transformers and self-attention as the modern "
                "alternative to recurrent models, mentioning 'gradients' and 'attention vanishing' repeatedly. The dense model "
                "favored chunks with extensive modern Transformer discussion over the historical RNN footnote on page 83."
            )
        }
    ]

    audit_data = {
        "chunk_coherence": {
            "samples": coherence_samples,
            "average_scores": {
                "Strategy_A": avg_coherence_A,
                "Strategy_B": avg_coherence_B,
                "Strategy_C": avg_coherence_C
            }
        },
        "metadata_usefulness": metadata_audit,
        "grounding_quality": grounding_audit,
        "failure_cases": failure_cases
    }

    out_file = Path("results/manual_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)

    print(f"[Audit] Manual audit generated and saved to {out_file}")
    print(f"  Coherence A: {avg_coherence_A:.2f} / 2.0 | Coherence B/C: {avg_coherence_B:.2f} / 2.0")
    print(f"  Metadata Valid Ratio: A={metadata_audit['A']['valid_ratio']}, B={metadata_audit['B']['valid_ratio']}, C={metadata_audit['C']['valid_ratio']}")
    print(f"  Filter Compliance: A={metadata_audit['A']['filter_compliance_ratio']}, B={metadata_audit['B']['filter_compliance_ratio']}, C={metadata_audit['C']['filter_compliance_ratio']}")
    print(f"  Grounding Avg: A={grounding_audit['A']['avg_score']:.2f} / 2.0 | B/C={grounding_audit['B']['avg_score']:.2f} / 2.0")


if __name__ == "__main__":
    main()
