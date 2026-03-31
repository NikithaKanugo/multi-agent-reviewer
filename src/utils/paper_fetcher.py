"""
Paper Fetcher — Download AI/ML Research Papers from arXiv

Searches arXiv for papers on specified topics, downloads PDFs,
extracts text, and saves them ready for the multi-agent pipeline.

USAGE:
    python -m src.utils.paper_fetcher                    # fetch default AI topics
    python -m src.utils.paper_fetcher --topic "transformers" --max 5
    python -m src.utils.paper_fetcher --all              # fetch all default topics
"""

import os
import re
import time
import argparse

import arxiv
import pymupdf


# ---------------------------------------------------------------
# DEFAULT SEARCH TOPICS
# ---------------------------------------------------------------
# Each topic maps to an arXiv search query.
# These cover the major areas of modern AI research.
# ---------------------------------------------------------------

SEARCH_TOPICS = {
    "large_language_models": "large language models LLM",
    "transformers": "transformer architecture attention mechanism",
    "reinforcement_learning": "reinforcement learning from human feedback RLHF",
    "diffusion_models": "diffusion models image generation",
    "multimodal_ai": "multimodal AI vision language models",
    "ai_agents": "AI agents autonomous tool use",
    "ai_safety": "AI safety alignment interpretability",
    "neural_architecture": "neural architecture search efficient inference",
    "retrieval_augmented": "retrieval augmented generation RAG",
    "fine_tuning": "fine tuning parameter efficient LoRA",
    "prompt_engineering": "prompt engineering in-context learning",
    "ai_reasoning": "chain of thought reasoning LLM",
    "code_generation": "code generation AI programming",
    "constitutional_ai": "constitutional AI RLAIF anthropic",
    "scaling_laws": "scaling laws neural networks compute",
}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file.

    Uses PyMuPDF for fast, reliable text extraction.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text as a string
    """
    text_parts = []
    doc = pymupdf.open(pdf_path)
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def sanitize_filename(name: str, max_length: int = 80) -> str:
    """Create a safe filename from a paper title."""
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:max_length]


def fetch_papers(
    query: str,
    max_results: int = 5,
    output_dir: str = "papers",
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
) -> list[dict]:
    """Search arXiv and download papers as text files.

    Args:
        query: Search query string
        max_results: Maximum number of papers to download
        output_dir: Directory to save paper text files
        sort_by: How to sort results (Relevance or SubmittedDate)

    Returns:
        List of dicts with paper metadata and file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Search arXiv
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
    )

    client = arxiv.Client()
    papers = []

    for result in client.results(search):
        title = result.title
        safe_title = sanitize_filename(title)
        txt_path = os.path.join(output_dir, f"{safe_title}.txt")

        # Skip if already downloaded
        if os.path.exists(txt_path):
            print(f"   ⏭️  Already exists: {safe_title}")
            papers.append({
                "title": title,
                "file_path": txt_path,
                "authors": [a.name for a in result.authors],
                "published": str(result.published.date()),
                "url": result.entry_id,
                "abstract": result.summary,
                "skipped": True,
            })
            continue

        print(f"   📥 Downloading: {title[:60]}...")

        # Download PDF to temp location
        pdf_path = os.path.join(output_dir, f"{safe_title}.pdf")
        try:
            result.download_pdf(dirpath=output_dir, filename=f"{safe_title}.pdf")

            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)

            if len(text.strip()) < 100:
                print(f"   ⚠️  Skipping (no extractable text): {title[:60]}")
                os.remove(pdf_path)
                continue

            # Build the text file with metadata header
            header = (
                f"Title: {title}\n"
                f"Authors: {', '.join(a.name for a in result.authors)}\n"
                f"Published: {result.published.date()}\n"
                f"URL: {result.entry_id}\n"
                f"Abstract: {result.summary}\n"
                f"\n{'='*60}\n\n"
            )

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(header + text)

            # Clean up PDF
            os.remove(pdf_path)

            papers.append({
                "title": title,
                "file_path": txt_path,
                "authors": [a.name for a in result.authors],
                "published": str(result.published.date()),
                "url": result.entry_id,
                "abstract": result.summary,
                "skipped": False,
            })

            print(f"   ✅ Saved: {safe_title}.txt")

            # Be nice to arXiv — rate limit
            time.sleep(1)

        except Exception as e:
            print(f"   ❌ Failed: {title[:60]} — {e}")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            continue

    return papers


def fetch_all_topics(
    max_per_topic: int = 3,
    output_dir: str = "papers",
    topics: dict = None,
) -> dict:
    """Fetch papers for all default AI/ML topics.

    Args:
        max_per_topic: Papers to download per topic
        output_dir: Directory to save papers
        topics: Custom topic dict (defaults to SEARCH_TOPICS)

    Returns:
        Dict mapping topic names to lists of paper metadata
    """
    if topics is None:
        topics = SEARCH_TOPICS

    all_papers = {}
    total = 0

    print(f"\n🔍 Fetching AI/ML papers across {len(topics)} topics")
    print(f"   Max {max_per_topic} papers per topic\n")

    for topic_name, query in topics.items():
        print(f"\n📚 Topic: {topic_name}")
        print(f"   Query: \"{query}\"")

        papers = fetch_papers(
            query=query,
            max_results=max_per_topic,
            output_dir=output_dir,
        )

        all_papers[topic_name] = papers
        new_count = sum(1 for p in papers if not p.get("skipped"))
        total += new_count
        print(f"   Downloaded {new_count} new paper(s)")

    print(f"\n{'='*60}")
    print(f"📊 Total new papers downloaded: {total}")
    print(f"📁 Saved to: {output_dir}/")
    print(f"{'='*60}\n")

    return all_papers


# ---------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch AI/ML papers from arXiv")
    parser.add_argument("--topic", type=str, help="Search query (e.g. 'transformers')")
    parser.add_argument("--max", type=int, default=3, help="Max papers to fetch (default: 3)")
    parser.add_argument("--all", action="store_true", help="Fetch all default AI topics")
    parser.add_argument("--output", type=str, default="papers", help="Output directory")

    args = parser.parse_args()

    if args.all:
        fetch_all_topics(max_per_topic=args.max, output_dir=args.output)
    elif args.topic:
        print(f"\n🔍 Searching arXiv for: \"{args.topic}\"")
        papers = fetch_papers(args.topic, max_results=args.max, output_dir=args.output)
        print(f"\n✅ Downloaded {len(papers)} paper(s) to {args.output}/")
    else:
        # Default: fetch a small sample across key topics
        sample_topics = {
            k: SEARCH_TOPICS[k]
            for k in ["large_language_models", "ai_agents", "ai_safety", "transformers", "retrieval_augmented"]
        }
        fetch_all_topics(max_per_topic=2, output_dir=args.output, topics=sample_topics)
