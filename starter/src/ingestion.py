"""Read and normalize source data from PDF.

Extracts physical pages, normalizes Unicode and whitespace,
preserves paragraph boundaries, and tags chunks with metadata (chapter, section).
"""
import re
import unicodedata
from pathlib import Path
import pypdf

from .models import DocumentPage


def get_section_for_page(page: int) -> tuple[int, str]:
    """Determine chapter and section based on book outline."""
    if page <= 72:
        chapter = 1
        if page < 40:
            section = "the_rise_of_ai_engineering"
        elif page < 52:
            section = "foundation_model_use_cases"
        elif page < 61:
            section = "planning_ai_applications"
        else:
            section = "ai_engineering_stack"
    else:
        chapter = 2
        if page < 82:
            section = "training_data"
        elif page < 102:
            section = "model_architecture"
        elif page < 112:
            section = "post_training"
        elif page < 120:
            section = "sampling_strategies"
        elif page < 123:
            section = "test_time_compute"
        elif page < 129:
            section = "structured_outputs"
        else:
            section = "probabilistic_nature_of_ai"
    return chapter, section


def normalize_text(text: str) -> str:
    """Normalize Unicode (NFC), clean line-break hyphens, and preserve paragraph structure."""
    if not text:
        return ""
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Remove soft hyphens
    text = text.replace("\u00ad", "")
    # Normalize CRLF to LF
    text = text.replace("\r\n", "\n")
    # Fix hyphenated words broken across line breaks (e.g. "build‐\n ing" -> "building")
    text = re.sub(r'(\w+)[\u2010\u2013\u2014-]\s*\n\s*(\w+)', r'\1\2', text)
    # Collapse intra-line multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse more than 2 consecutive newlines into 2 (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_pdf(path: Path, document_id: str, source: str,
             start_page: int = 25, end_page: int = 136) -> list[DocumentPage]:
    """Extract physical pages within range, normalize text, and retain metadata."""
    reader = pypdf.PdfReader(str(path))
    total_physical = len(reader.pages)
    
    pages: list[DocumentPage] = []
    actual_end = min(end_page, total_physical)
    
    empty_pages = []
    for p_num in range(start_page, actual_end + 1):
        raw_text = reader.pages[p_num - 1].extract_text() or ""
        cleaned = normalize_text(raw_text)
        
        if not cleaned:
            empty_pages.append(p_num)
            continue
            
        chapter, section = get_section_for_page(p_num)
        page_obj = DocumentPage(
            document_id=document_id,
            source=source,
            page=p_num,
            text=cleaned,
            metadata={
                "chapter": chapter,
                "section": section,
                "char_count": len(cleaned)
            }
        )
        pages.append(page_obj)

    if empty_pages:
        print(f"[Ingestion] Note: Skipped {len(empty_pages)} empty pages: {empty_pages}")
    print(f"[Ingestion] Loaded {len(pages)} content pages from physical pages {start_page}..{actual_end}")
    return pages
