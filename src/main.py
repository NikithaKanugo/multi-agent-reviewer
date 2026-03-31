"""
Main Entry Point — Multi-Agent Research Assistant

This is where everything comes together. Run this file to process a paper.

EXECUTION FLOW:
1. Load environment (API key)
2. Load and chunk the paper
3. Initialize FAISS shared memory
4. Build the LangGraph workflow
5. Run the pipeline (Researcher → Writer → Reviewer → loop or end)
6. Save the final summary

USAGE:
    python -m src.main                          # uses default sample paper
    python -m src.main papers/your_paper.txt    # specify a paper

SYSTEM DESIGN PARALLEL:
This is the "composition root" — the one place where all components
are assembled and the application starts. In a web app, this would be
app.py or server.py. No business logic lives here — just bootstrapping.
"""

import os
import sys

from dotenv import load_dotenv

from src.utils.paper_loader import load_paper
from src.memory.shared_memory import SharedMemory
from src.graph import build_graph


def main(paper_path: str = "papers/sample_paper.txt") -> dict:
    """Run the full multi-agent research pipeline.

    Args:
        paper_path: Path to the paper text file

    Returns:
        Final state dict with all fields populated
    """

    # -------------------------------------------------------
    # STEP 1: Load environment variables
    # -------------------------------------------------------
    # load_dotenv() reads the .env file and sets the values
    # as environment variables. The OpenAI client automatically
    # reads OPENAI_API_KEY from the environment.
    #
    # WHY .env instead of hardcoding?
    # - Security: keys never end up in git
    # - Flexibility: different keys for dev/staging/production
    # - Standard practice: every production app does this
    # -------------------------------------------------------
    load_dotenv()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found.")
        print("Create a .env file with: GROQ_API_KEY=your-key-here")
        print("(Copy .env.example to .env and add your key)")
        sys.exit(1)

    # -------------------------------------------------------
    # STEP 2: Load and chunk the paper
    # -------------------------------------------------------
    # load_paper() does three things:
    #   1. Reads the text file
    #   2. Splits text into overlapping chunks (for FAISS)
    #   3. Extracts metadata (title, word count, etc.)
    #
    # This is the PREPROCESSING step. Everything downstream
    # depends on good chunking. Bad chunks → bad retrieval →
    # bad research notes → bad summary. Garbage in, garbage out.
    # -------------------------------------------------------
    print(f"\n📄 Loading paper: {paper_path}")
    paper_data = load_paper(paper_path)

    print(f"   Title: {paper_data['metadata']['title']}")
    print(f"   Words: {paper_data['metadata']['word_count']}")
    print(f"   Tokens: {paper_data['metadata']['token_count']}")
    print(f"   Chunks: {paper_data['metadata']['chunk_count']}")

    # -------------------------------------------------------
    # STEP 3: Initialize FAISS shared memory
    # -------------------------------------------------------
    # Create the SharedMemory instance and load all chunks.
    # This calls the OpenAI Embedding API to convert each chunk
    # into a vector, then stores them in the FAISS index.
    #
    # After this step, any agent can call memory.search("query")
    # to find relevant chunks. This is the SHARED KNOWLEDGE BASE.
    #
    # COST: ~$0.0001 for a typical paper (basically free)
    # -------------------------------------------------------
    print("\n🧠 Initializing shared memory (FAISS)...")
    memory = SharedMemory()
    memory.add_chunks(paper_data["paper_chunks"])
    print(f"   Indexed {len(paper_data['paper_chunks'])} chunks into FAISS")

    # -------------------------------------------------------
    # STEP 4: Build the LangGraph workflow
    # -------------------------------------------------------
    # build_graph() creates the full workflow:
    #   START → Researcher → Writer → Reviewer → (END or → Writer)
    #
    # The memory instance is passed in so the Researcher node
    # can query FAISS. The graph is compiled and ready to run.
    # -------------------------------------------------------
    print("\n🔧 Building agent workflow...")
    workflow = build_graph(memory)
    print("   Graph compiled: Researcher → Writer → Reviewer (with review loop)")

    # -------------------------------------------------------
    # STEP 5: Create initial state and run
    # -------------------------------------------------------
    # The initial state has:
    #   - Input data (paper_text, chunks, metadata) from the loader
    #   - Empty agent outputs (will be filled as agents run)
    #   - Control fields (revision_count=0, max_revisions=3)
    #
    # max_revisions=3 means: if the Reviewer rejects 3 times,
    # force-accept and stop. This is our circuit breaker.
    #
    # graph.invoke() runs the entire workflow synchronously:
    #   → Researcher runs, updates state
    #   → Writer runs, updates state
    #   → Reviewer runs, updates state
    #   → If rejected: back to Writer (up to 3 times)
    #   → Returns the final state with everything filled in
    # -------------------------------------------------------
    initial_state = {
        # Input data
        "paper_text": paper_data["paper_text"],
        "paper_chunks": paper_data["paper_chunks"],
        "metadata": paper_data["metadata"],

        # Agent outputs (empty — agents will fill these)
        "research_notes": "",
        "draft_summary": "",
        "review_feedback": "",

        # Control flow
        "is_approved": False,
        "revision_count": 0,
        "max_revisions": 3,
    }

    print("\n🚀 Running multi-agent pipeline...")
    print("   Step 1/3: Researcher extracting key information...")
    print("   (This makes multiple FAISS queries + OpenAI calls)\n")

    # --- RUN THE FULL PIPELINE ---
    final_state = workflow.invoke(initial_state)

    # -------------------------------------------------------
    # STEP 6: Display results and save output
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    # Show review status
    if final_state.get("is_approved"):
        revision_count = final_state.get("revision_count", 0)
        if revision_count == 0:
            print("✅ Summary approved on first draft!")
        else:
            print(f"✅ Summary approved after {revision_count} revision(s)")
    else:
        print("⚠️  Summary accepted (max revisions reached)")

    # Save the final summary
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Create a clean filename from the paper title
    title = final_state.get("metadata", {}).get("title", "untitled")
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    safe_title = safe_title.strip().replace(" ", "_")[:50]
    output_path = os.path.join(output_dir, f"{safe_title}_summary.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_state["draft_summary"])

    print(f"\n📝 Summary saved to: {output_path}")

    # Also save the research notes (useful for debugging)
    notes_path = os.path.join(output_dir, f"{safe_title}_notes.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(final_state["research_notes"])

    print(f"📋 Research notes saved to: {notes_path}")

    # Show a preview of the summary
    print(f"\n{'=' * 60}")
    print("SUMMARY PREVIEW (first 500 chars)")
    print("=" * 60)
    print(final_state["draft_summary"][:500])
    print("...\n")

    return final_state


# -------------------------------------------------------
# SCRIPT ENTRY POINT
# -------------------------------------------------------
# This block runs only when you execute the file directly:
#   python -m src.main
#   python -m src.main papers/my_paper.txt
#
# It does NOT run when the file is imported:
#   from src.main import main  ← this won't trigger it
#
# WHY? So you can import main() in tests without it auto-running.
# This is a Python best practice for all entry point files.
# -------------------------------------------------------
def batch_process(papers_dir: str = "papers") -> None:
    """Process all .txt papers in a directory.

    Args:
        papers_dir: Directory containing paper .txt files
    """
    import glob

    txt_files = sorted(glob.glob(os.path.join(papers_dir, "*.txt")))
    if not txt_files:
        print(f"No .txt files found in {papers_dir}/")
        return

    print(f"\n📚 Batch processing {len(txt_files)} paper(s) from {papers_dir}/\n")

    results = []
    for i, paper_path in enumerate(txt_files, 1):
        print(f"\n{'='*60}")
        print(f"Paper {i}/{len(txt_files)}: {os.path.basename(paper_path)}")
        print(f"{'='*60}")
        try:
            state = main(paper_path)
            results.append({"file": paper_path, "status": "success"})
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({"file": paper_path, "status": f"failed: {e}"})

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {sum(1 for r in results if r['status'] == 'success')}/{len(results)} succeeded")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        papers_dir = sys.argv[2] if len(sys.argv) > 2 else "papers"
        batch_process(papers_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "--fetch":
        # Fetch papers from arXiv then process them
        from src.utils.paper_fetcher import fetch_all_topics
        fetch_all_topics(max_per_topic=2)
    elif len(sys.argv) > 1:
        paper_file = sys.argv[1]
        main(paper_file)
    else:
        paper_file = "papers/sample_paper.txt"
        main(paper_file)
