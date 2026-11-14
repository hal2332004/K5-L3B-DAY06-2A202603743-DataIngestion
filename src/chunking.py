"""Implement fixed-size and structure-aware chunking."""
import re
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split text into overlapping windows and retain page metadata (Strategy A)."""
    text = page.text
    if not text:
        return []

    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    idx = 0

    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue

        chunk_id = f"{page.document_id}_p{page.page}_f{idx:03d}"
        chunk_meta = dict(page.metadata)
        chunk_meta.update({
            "chunk_strategy": "fixed_size",
            "chunk_index": idx,
            "char_length": len(chunk_text)
        })

        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=chunk_text,
            metadata=chunk_meta
        ))
        idx += 1

        if end >= len(text):
            break

    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    """Group paragraphs, split oversized paragraphs and retain metadata (Strategy B & C)."""
    text = page.text
    if not text:
        return []

    # Split by double newline or significant paragraph breaks
    raw_paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not raw_paras:
        raw_paras = [text.strip()]

    # Break any oversized paragraph that exceeds max_size into sub-paragraphs by sentence
    processed_paras: list[str] = []
    for p in raw_paras:
        if len(p) <= max_size:
            processed_paras.append(p)
        else:
            # Split by sentence ending (.!?)
            sentences = re.split(r'(?<=[.!?])\s+', p)
            current_sub = ""
            for s in sentences:
                if not current_sub:
                    current_sub = s
                elif len(current_sub) + 1 + len(s) <= max_size:
                    current_sub += " " + s
                else:
                    processed_paras.append(current_sub.strip())
                    current_sub = s
            if current_sub.strip():
                processed_paras.append(current_sub.strip())

    # Group paragraphs up to max_size
    chunks: list[Chunk] = []
    current_chunk_text = ""
    idx = 0

    for para in processed_paras:
        if not current_chunk_text:
            current_chunk_text = para
        elif len(current_chunk_text) + 2 + len(para) <= max_size:
            current_chunk_text += "\n\n" + para
        else:
            chunk_id = f"{page.document_id}_p{page.page}_p{idx:03d}"
            chunk_meta = dict(page.metadata)
            chunk_meta.update({
                "chunk_strategy": "by_paragraph",
                "chunk_index": idx,
                "char_length": len(current_chunk_text)
            })
            chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=current_chunk_text,
                metadata=chunk_meta
            ))
            idx += 1
            current_chunk_text = para

    if current_chunk_text:
        chunk_id = f"{page.document_id}_p{page.page}_p{idx:03d}"
        chunk_meta = dict(page.metadata)
        chunk_meta.update({
            "chunk_strategy": "by_paragraph",
            "chunk_index": idx,
            "char_length": len(current_chunk_text)
        })
        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=current_chunk_text,
            metadata=chunk_meta
        ))

    return chunks
