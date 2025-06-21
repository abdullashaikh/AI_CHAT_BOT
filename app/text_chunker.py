from typing import List

def chunk_text(pages: List[str], chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Chunks list of pages into smaller text blocks.
    - chunk_size: Number of characters per chunk
    - overlap: Number of overlapping characters between chunks (for context)
    """
    chunks = []

    for page in pages:
        start = 0
        while start < len(page):
            end = start + chunk_size
            chunk = page[start:end]
            chunks.append(chunk.strip())
            start += chunk_size - overlap

    return [c for c in chunks if c]  # Filter out any empty chunks
