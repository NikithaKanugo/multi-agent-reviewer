"""
Paper Loader & Text Chunker

This module handles the PREPROCESSING step before any agent runs:
1. Load paper text from a file
2. Extract basic metadata (title, word count)
3. Chunk the text into overlapping pieces for FAISS indexing

WHY THIS MATTERS:
- FAISS needs individual chunks, not whole documents
- Overlapping chunks prevent information loss at boundaries
- Good chunking = good retrieval = good agent output
- Bad chunking = the #1 silent killer of RAG systems
"""

import os
import tiktoken


# ---------------------------------------------------------------
# TOKEN COUNTER
# ---------------------------------------------------------------
# We chunk by TOKENS not characters, because LLMs think in tokens.
# "hello world" = 2 tokens, but 11 characters.
# Chunking by characters would give inconsistent sizes to the LLM.
# tiktoken is OpenAI's tokenizer — it counts tokens the same way
# the API does, so our chunk sizes are accurate.
# ---------------------------------------------------------------

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string.

    Why tokens instead of words?
    - LLMs process tokens, not words
    - "unhappiness" = 3 tokens (un + happi + ness)
    - Token count determines API cost and context limits
    """
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


# ---------------------------------------------------------------
# TEXT CHUNKER
# ---------------------------------------------------------------
# This is the CORE function. It splits text into overlapping chunks.
#
# Think of it like a sliding window moving across the text:
#   - Window size = chunk_size (500 tokens)
#   - Step size = chunk_size - overlap (400 tokens)
#   - Each window becomes one chunk stored in FAISS
# ---------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks based on token count.

    Args:
        text: The full paper text to chunk
        chunk_size: Maximum tokens per chunk (default 500)
            - Too small (50): lose paragraph context
            - Too large (2000): lose retrieval precision
            - Sweet spot: 300-800 for academic papers
        chunk_overlap: Tokens shared between consecutive chunks (default 100)
            - Prevents information loss at chunk boundaries
            - Rule of thumb: 10-20% of chunk_size

    Returns:
        List of text chunks, each roughly chunk_size tokens
    """
    # We split by sentences first, not characters.
    # WHY? Splitting mid-sentence destroys meaning.
    # "The result was not significant" split at "not" → "The result was not"
    # and "significant" — completely changes the meaning!
    sentences = _split_into_sentences(text)

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # If adding this sentence would exceed chunk_size, save current chunk
        # and start a new one (with overlap)
        if current_token_count + sentence_tokens > chunk_size and current_chunk_sentences:
            # Save the completed chunk
            chunk_text_joined = " ".join(current_chunk_sentences)
            chunks.append(chunk_text_joined)

            # START OVERLAP: keep the last few sentences for the next chunk
            # Walk backwards through sentences until we have ~overlap tokens
            overlap_sentences = []
            overlap_tokens = 0
            for sent in reversed(current_chunk_sentences):
                sent_tokens = count_tokens(sent)
                if overlap_tokens + sent_tokens > chunk_overlap:
                    break
                overlap_sentences.insert(0, sent)
                overlap_tokens += sent_tokens

            # New chunk starts with the overlapping sentences
            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_tokens

        current_chunk_sentences.append(sentence)
        current_token_count += sentence_tokens

    # Don't forget the last chunk!
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Simple approach: split on period + space, question mark, exclamation.

    WHY not use NLTK or spaCy?
    - They're heavy dependencies for a simple task
    - This works well enough for academic papers
    - In production, you'd use a proper sentence tokenizer
      if you hit edge cases (e.g., "Dr. Smith" splitting on the period)
    """
    # Clean up whitespace (papers often have weird formatting)
    text = " ".join(text.split())

    sentences = []
    current = ""

    for i, char in enumerate(text):
        current += char
        # End of sentence: period/question/exclamation followed by space or end
        if char in ".?!" and (i + 1 == len(text) or text[i + 1] == " "):
            stripped = current.strip()
            if stripped:
                sentences.append(stripped)
            current = ""

    # Catch any remaining text (last sentence without punctuation)
    if current.strip():
        sentences.append(current.strip())

    return sentences


# ---------------------------------------------------------------
# PAPER LOADER
# ---------------------------------------------------------------
# Loads a paper from a .txt file and returns everything the system
# needs: raw text, chunks, and metadata.
# ---------------------------------------------------------------

def load_paper(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> dict:
    """Load a paper from a text file and prepare it for the pipeline.

    This is the ENTRY POINT for getting a paper into the system.
    It returns everything needed to populate the initial state.

    Args:
        file_path: Path to the paper (.txt file)
        chunk_size: Tokens per chunk for FAISS indexing
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        dict with keys:
            - paper_text: the full raw text
            - paper_chunks: list of overlapping text chunks
            - metadata: dict with title, source, word_count, chunk_count
    """
    # Validate the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Paper not found: {file_path}")

    # Read the raw text
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError(f"Paper is empty: {file_path}")

    # Chunk the text for FAISS
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # Extract basic metadata
    # The first line is often the title in academic papers
    lines = text.strip().split("\n")
    title = lines[0].strip() if lines else "Untitled"

    metadata = {
        "title": title,
        "source": file_path,
        "word_count": len(text.split()),
        "token_count": count_tokens(text),
        "chunk_count": len(chunks),
    }

    return {
        "paper_text": text,
        "paper_chunks": chunks,
        "metadata": metadata,
    }
