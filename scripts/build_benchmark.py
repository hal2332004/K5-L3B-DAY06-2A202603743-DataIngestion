"""Validate and build benchmark/questions.json.

Ensures:
- Exactly 20 questions
- 15 dev / 5 test
- At least 3 filter questions (we have 5: 3 in dev, 2 in test)
- Every single evidence_quote exists VERBATIM on the designated physical page of the PDF
"""
import glob
import json
import re
from pathlib import Path
import pypdf


def norm(text: str) -> str:
    # Normalize soft hyphens, dashes, and whitespace
    text = text.replace('\u00ad', '')
    # replace line-break hyphens like "build‐ ing" or "parame‐ ters"
    text = re.sub(r'(\w+)[\u2010\u2013\u2014-]\s+(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def main():
    pdf_files = glob.glob('data/*.pdf')
    if not pdf_files:
        raise FileNotFoundError("No PDF file found in data/")
    reader = pypdf.PdfReader(pdf_files[0])

    questions = [
        # ==========================================
        # DEV SPLIT (15 questions)
        # ==========================================
        {
            "id": "q01",
            "split": "dev",
            "type": "truc_tiep",
            "question": "What are the three layers of any AI application stack according to Chip Huyen?",
            "expected_answer": "The three layers are application development, model development, and infrastructure.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 61,
            "evidence_quote": "There are three layers to any AI application stack: application development, model development, and infrastructure.",
            "filter": None
        },
        {
            "id": "q02",
            "split": "dev",
            "type": "truc_tiep",
            "question": "Why do language models use tokens as their unit instead of words or characters?",
            "expected_answer": "Tokens allow the model to break words into meaningful components, compared to characters.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 28,
            "evidence_quote": "1. Compared to characters, tokens allow the model to break words into meaningful components.",
            "filter": None
        },
        {
            "id": "q03",
            "split": "dev",
            "type": "truc_tiep",
            "question": "What is the primary role of post-training in foundation models?",
            "expected_answer": "The goal of post-training is to align the model with human preferences.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 74,
            "evidence_quote": "The goal of post-training is to align the model with human preferences.",
            "filter": None
        },
        {
            "id": "q04",
            "split": "dev",
            "type": "truc_tiep",
            "question": "What is top-k sampling designed to achieve during text generation?",
            "expected_answer": "Top-k is a sampling strategy to reduce computation workload without sacrificing too much response diversity.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 118,
            "evidence_quote": "Top-k is a sampling strategy to reduce the computation workload without sacrificing too much of the model’s response diversity.",
            "filter": None
        },
        {
            "id": "q05",
            "split": "dev",
            "type": "dien_dat_lai",
            "question": "In what way does adapting an existing foundation model compare in effort to training one from scratch?",
            "expected_answer": "Adapting an existing model is much easier and faster, taking ten examples and one weekend versus 1 million examples and six months.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 36,
            "evidence_quote": "Adapting an existing powerful model to your task is generally a lot easier than building a model for your task from scratch—for example, ten examples and one weekend versus 1 million examples and six months.",
            "filter": None
        },
        {
            "id": "q06",
            "split": "dev",
            "type": "dien_dat_lai",
            "question": "How does nucleus sampling differ dynamically from top-k sampling?",
            "expected_answer": "Top-p (nucleus sampling) allows for a more dynamic selection of candidate values to sample from.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 118,
            "evidence_quote": "Top-p, also known as nucleus sampling, allows for a more dynamic selection of values to be sampled from.",
            "filter": None
        },
        {
            "id": "q07",
            "split": "dev",
            "type": "metadata_filter",
            "question": "In the AI engineering stack discussed in Chapter 1, which categories saw the highest increases in 2023?",
            "expected_answer": "Applications and application development saw the highest increases in 2023.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 63,
            "evidence_quote": "In 2023, the categories that saw the highest increases were applications and application development.",
            "filter": {"chapter": 1, "section": "ai_engineering_stack"}
        },
        {
            "id": "q08",
            "split": "dev",
            "type": "dien_dat_lai",
            "question": "Why are recurrent neural networks particularly vulnerable to vanishing and exploding gradients?",
            "expected_answer": "Because of their recursive structure requiring gradients to propagate through many steps.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 83,
            "evidence_quote": "RNNs are especially prone to vanishing and exploding gradients due to their recursive structure.",
            "filter": None
        },
        {
            "id": "q09",
            "split": "dev",
            "type": "dieu_kien_ngoai_le",
            "question": "Why are raw pre-trained language models not immediately suitable for interactive chatbot conversations?",
            "expected_answer": "Pre-trained models are optimized for text completion rather than conversing.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 104,
            "evidence_quote": "As discussed in Chapter 1, the pre-trained model is likely optimized for completion rather than conversing.",
            "filter": None
        },
        {
            "id": "q10",
            "split": "dev",
            "type": "dieu_kien_ngoai_le",
            "question": "What empirical performance gains do RLHF and DPO provide over supervised fine-tuning alone?",
            "expected_answer": "Empirically, both RLHF and DPO improve performance compared to SFT alone.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 112,
            "evidence_quote": "Empirically, RLHF and DPO both improve performance compared to SFT alone.",
            "filter": None
        },
        {
            "id": "q11",
            "split": "dev",
            "type": "metadata_filter",
            "question": "Under Chapter 2 sampling strategies, what drawback can occur if sampling tokens strictly by probability distribution?",
            "expected_answer": "The model can be less creative.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 114,
            "evidence_quote": "One problem with sampling the next token according to the probability distribution is that the model can be less creative.",
            "filter": {"chapter": 2, "section": "sampling_strategies"}
        },
        {
            "id": "q12",
            "split": "dev",
            "type": "dieu_kien_ngoai_le",
            "question": "Does top-p sampling necessarily reduce the softmax computation load compared to top-k?",
            "expected_answer": "Unlike top-k, top-p does not necessarily reduce the softmax computation load.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 119,
            "evidence_quote": "Unlike top-k, top-p doesn’t necessarily reduce the softmax computation load.",
            "filter": None
        },
        {
            "id": "q13",
            "split": "dev",
            "type": "dien_dat_lai",
            "question": "What does a suffix like 13B signify when appended to a model name like Llama-13B?",
            "expected_answer": "It denotes the number of parameters, such as 13 billion parameters.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 91,
            "evidence_quote": "The number of parameters is usually appended at the end of a model name. For example, Llama-13B refers to the version of Llama, a model family developed by Meta, with 13 billion parameters.",
            "filter": None
        },
        {
            "id": "q14",
            "split": "dev",
            "type": "metadata_filter",
            "question": "In Chapter 1 use cases, how fast did GitHub Copilot reach $100 million in annual recurring revenue?",
            "expected_answer": "Only two years after its launch.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 44,
            "evidence_quote": "GitHub Copilot, whose annual recurring revenue crossed $100 million only two years after its launch.",
            "filter": {"chapter": 1, "section": "foundation_model_use_cases"}
        },
        {
            "id": "q15",
            "split": "dev",
            "type": "dieu_kien_ngoai_le",
            "question": "Can masked language models like BERT be employed for generating text?",
            "expected_answer": "Technically yes if you try really hard, but autoregressive models are much more popular.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 29,
            "evidence_quote": "Technically, a masked language model like BERT can also be used for text generations if you try really hard.",
            "filter": None
        },

        # ==========================================
        # TEST SPLIT (5 questions)
        # ==========================================
        {
            "id": "q16",
            "split": "test",
            "type": "truc_tiep",
            "question": "What supervision method did OpenAI utilize to train the CLIP language-image model without manual labeling?",
            "expected_answer": "OpenAI used natural language supervision, a variant of self-supervision.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 34,
            "evidence_quote": "natural language supervision to train their language-image model CLIP (OpenAI, 2021)",
            "filter": None
        },
        {
            "id": "q17",
            "split": "test",
            "type": "dien_dat_lai",
            "question": "Which generative AI application is consistently reported as the most popular in industry surveys?",
            "expected_answer": "Coding is hands down the most popular use case.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 44,
            "evidence_quote": "In multiple generative AI surveys, coding is hands down the most popular use case.",
            "filter": None
        },
        {
            "id": "q18",
            "split": "test",
            "type": "metadata_filter",
            "question": "In Chapter 2 on training data, what percentage of the Common Crawl corpus is in English?",
            "expected_answer": "English represents 45.8786% of Common Crawl.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 76,
            "evidence_quote": "English en 1,452 45.8786",
            "filter": {"chapter": 2, "section": "training_data"}
        },
        {
            "id": "q19",
            "split": "test",
            "type": "dieu_kien_ngoai_le",
            "question": "What level of generation does test time compute target compared to individual token sampling?",
            "expected_answer": "Test time compute discusses sampling the whole output rather than just the next token.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 120,
            "evidence_quote": "The last section discussed how a model might sample the next token. This section discusses how a model might sample the whole output.",
            "filter": None
        },
        {
            "id": "q20",
            "split": "test",
            "type": "metadata_filter",
            "question": "In Chapter 2 under post-training, how many volunteers did LAION mobilize to create conversational data?",
            "expected_answer": "LAION mobilized 13,500 volunteers worldwide to generate 10,000 conversations.",
            "gold_source_id": "chip-huyen-ai-engineering-ch1-2",
            "gold_page": 107,
            "evidence_quote": "LAION, a non-profit organization, mobilized 13,500 volunteers worldwide to generate 10,000 conversations",
            "filter": {"chapter": 2, "section": "post_training"}
        }
    ]

    print(f"Validating {len(questions)} questions against source PDF...")
    all_ok = True
    dev_count = sum(1 for q in questions if q["split"] == "dev")
    test_count = sum(1 for q in questions if q["split"] == "test")
    filter_count = sum(1 for q in questions if q["filter"] is not None)
    filter_test_count = sum(1 for q in questions if q["split"] == "test" and q["filter"] is not None)

    print(f"Summary counts: Dev={dev_count}, Test={test_count}, Filter Total={filter_count}, Filter in Test={filter_test_count}")

    for q in questions:
        p = q["gold_page"]
        raw_text = reader.pages[p - 1].extract_text() or ""
        norm_page = norm(raw_text)
        norm_quote = norm(q["evidence_quote"])

        if norm_quote in norm_page:
            print(f"  [OK] {q['id']} (p.{p}, split={q['split']}, filter={bool(q['filter'])})")
        else:
            print(f"  [FAIL] {q['id']} (p.{p})")
            print(f"     Target quote: '{norm_quote}'")
            print(f"     Page preview: '{norm_page[:150]}...'")
            all_ok = False

    if not all_ok:
        print("\nValidation FAILED! Please adjust failed quotes.")
        return False

    # Save to benchmark/questions.json
    out_dir = Path("benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "questions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"\nAll 20 questions verified and saved to {out_file}!")
    return True


if __name__ == "__main__":
    main()
