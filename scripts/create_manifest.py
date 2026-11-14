"""Generate data/manifest.json for Chapters 1 & 2 of Chip Huyen's AI Engineering."""
import glob
import json
from pathlib import Path
import pypdf


def main():
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        raise FileNotFoundError("No PDF file in data/")
    
    reader = pypdf.PdfReader(pdf_files[0])
    start_page = 25
    end_page = 136
    pages_info = []
    total_chars = 0

    for p in range(start_page, end_page + 1):
        text = reader.pages[p - 1].extract_text() or ""
        c_len = len(text.strip())
        total_chars += c_len
        chapter = 1 if p <= 72 else 2
        pages_info.append({
            "page": p,
            "chapter": chapter,
            "char_count": c_len
        })

    manifest = {
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "title": "AI Engineering: Building Applications with Foundation Models",
        "author": "Chip Huyen",
        "publisher": "O'Reilly Media",
        "publication_year": 2025,
        "source_file": pdf_files[0].replace("\\", "/"),
        "license_or_permission": "Educational / Academic Fair Use for VinAI Course Laboratory Research",
        "accessed_at": "2026-11-14",
        "scope_description": "Chapters 1 and 2 (Physical pages 25 to 136, inclusive). Chapter 1: Introduction to AI Engineering (pp. 25-72). Chapter 2: Understanding Foundation Models (pp. 73-136). Pages 1-24 (covers, copyright, TOC, preface) are excluded as non-content.",
        "total_content_pages": len(pages_info),
        "total_characters": total_chars,
        "page_range": {
            "start_physical_page": start_page,
            "end_physical_page": end_page
        },
        "pages": pages_info
    }

    out_file = Path("data/manifest.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest written successfully: {len(pages_info)} content pages, {total_chars:,} characters total.")


if __name__ == "__main__":
    main()

