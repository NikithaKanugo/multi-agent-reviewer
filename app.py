"""
Web App — Multi-Agent Research Assistant

Flask web UI with a 3-panel layout: paper upload, chat, and agent response.

USAGE:
    python app.py
    Then open http://localhost:8080 in your browser
"""

import os
import glob
import json
import threading
import time

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
app.secret_key = "multiagent-research-assistant"
app.config["UPLOAD_FOLDER"] = "papers"
app.config["OUTPUT_FOLDER"] = "output"

# Track background jobs
processing_status = {}

# Cache FAISS memory per paper to avoid rebuilding on every chat message
paper_memories = {}

# Store chat history per paper
chat_histories = {}

# Store pipeline progress for live updates
pipeline_progress = {}


def get_papers():
    """Get list of available paper files."""
    files = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], "*.txt"))
    papers = []
    for f in sorted(files):
        basename = os.path.basename(f)
        paper_id = basename.replace(".txt", "")
        name = paper_id.replace("_", " ")
        papers.append({
            "id": paper_id,
            "name": name,
            "path": f,
            "filename": basename,
            "status": processing_status.get(paper_id, "ready"),
            "has_summary": os.path.exists(
                _find_summary_for_paper(paper_id)
            ) if _find_summary_for_paper(paper_id) else False,
        })
    return papers


def _find_summary_for_paper(paper_id):
    """Find summary file matching a paper id."""
    pattern = os.path.join(app.config["OUTPUT_FOLDER"], "*_summary.md")
    for f in glob.glob(pattern):
        return f
    return None


def get_summaries():
    """Get list of generated summaries."""
    files = glob.glob(os.path.join(app.config["OUTPUT_FOLDER"], "*_summary.md"))
    summaries = []
    for f in sorted(files):
        name = os.path.basename(f).replace("_summary.md", "").replace("_", " ")
        with open(f, "r", encoding="utf-8") as fh:
            content = fh.read()
        summaries.append({
            "name": name,
            "path": f,
            "content": content,
            "filename": os.path.basename(f),
        })
    return summaries


def _get_or_build_memory(paper_id):
    """Get cached FAISS memory for a paper, or build it."""
    if paper_id in paper_memories:
        return paper_memories[paper_id]

    paper_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{paper_id}.txt")
    if not os.path.exists(paper_path):
        return None

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from src.utils.paper_loader import load_paper
    from src.memory.shared_memory import SharedMemory

    paper_data = load_paper(paper_path)
    memory = SharedMemory()
    memory.add_chunks(paper_data["paper_chunks"])

    paper_memories[paper_id] = {
        "memory": memory,
        "paper_data": paper_data,
    }
    return paper_memories[paper_id]


def process_paper_background(paper_path, paper_id):
    """Run the pipeline on a paper in a background thread."""
    try:
        processing_status[paper_id] = "processing"
        pipeline_progress[paper_id] = {
            "stage": "starting",
            "detail": "Initializing pipeline...",
            "stages_done": [],
        }
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        from src.utils.paper_loader import load_paper
        from src.memory.shared_memory import SharedMemory
        from src.graph import build_graph

        # Step 1: Load paper
        pipeline_progress[paper_id] = {
            "stage": "loading",
            "detail": "Loading and chunking paper...",
            "stages_done": [],
        }
        paper_data = load_paper(paper_path)

        # Step 2: Build FAISS
        pipeline_progress[paper_id] = {
            "stage": "indexing",
            "detail": f"Indexing {paper_data['metadata']['chunk_count']} chunks into FAISS...",
            "stages_done": ["loading"],
        }
        memory = SharedMemory()
        memory.add_chunks(paper_data["paper_chunks"])

        # Cache the memory for chat
        paper_memories[paper_id] = {
            "memory": memory,
            "paper_data": paper_data,
        }

        # Step 3: Build and run graph
        pipeline_progress[paper_id] = {
            "stage": "researcher",
            "detail": "Researcher agent extracting key information...",
            "stages_done": ["loading", "indexing"],
        }

        workflow = build_graph(memory)

        initial_state = {
            "paper_text": paper_data["paper_text"],
            "paper_chunks": paper_data["paper_chunks"],
            "metadata": paper_data["metadata"],
            "research_notes": "",
            "draft_summary": "",
            "review_feedback": "",
            "is_approved": False,
            "revision_count": 0,
            "max_revisions": 3,
        }

        final_state = workflow.invoke(initial_state)

        # Save outputs
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        title = final_state.get("metadata", {}).get("title", "untitled")
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
        safe_title = safe_title.strip().replace(" ", "_")[:50]

        summary_path = os.path.join(output_dir, f"{safe_title}_summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(final_state["draft_summary"])

        notes_path = os.path.join(output_dir, f"{safe_title}_notes.md")
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(final_state["research_notes"])

        # Store the review data for the UI
        pipeline_progress[paper_id] = {
            "stage": "complete",
            "detail": "Pipeline complete!",
            "stages_done": ["loading", "indexing", "researcher", "writer", "reviewer"],
            "summary_file": os.path.basename(summary_path),
            "notes_file": os.path.basename(notes_path),
            "review_feedback": final_state.get("review_feedback", ""),
            "revision_count": final_state.get("revision_count", 0),
            "is_approved": final_state.get("is_approved", False),
        }

        processing_status[paper_id] = "done"
    except Exception as e:
        processing_status[paper_id] = f"error: {e}"
        pipeline_progress[paper_id] = {
            "stage": "error",
            "detail": str(e),
            "stages_done": pipeline_progress.get(paper_id, {}).get("stages_done", []),
        }


# ---------------------------------------------------------------
# PAGE ROUTES
# ---------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summary/<filename>")
def view_summary(filename):
    filepath = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if not os.path.exists(filepath):
        flash("Summary not found")
        return redirect(url_for("index"))

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    name = filename.replace("_summary.md", "").replace("_", " ")
    return render_template("summary.html", name=name, content=content)


# ---------------------------------------------------------------
# JSON API ENDPOINTS
# ---------------------------------------------------------------

@app.route("/api/papers")
def api_papers():
    """List all papers with status."""
    return jsonify(get_papers())


@app.route("/api/summaries")
def api_summaries():
    """List all generated summaries."""
    summaries = get_summaries()
    # Don't send full content in list view
    for s in summaries:
        s["preview"] = s["content"][:300]
    return jsonify(summaries)


@app.route("/api/summary/<filename>")
def api_summary(filename):
    """Get full summary content."""
    filepath = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "not found"}), 404
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # Also try to load notes
    notes_file = filename.replace("_summary.md", "_notes.md")
    notes_path = os.path.join(app.config["OUTPUT_FOLDER"], notes_file)
    notes = ""
    if os.path.exists(notes_path):
        with open(notes_path, "r", encoding="utf-8") as f:
            notes = f.read()
    return jsonify({
        "filename": filename,
        "content": content,
        "notes": notes,
    })


@app.route("/api/paper/<paper_id>/analysis")
def api_paper_analysis(paper_id):
    """Get summary + notes for a specific paper by matching its title to output files."""
    paper_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{paper_id}.txt")
    if not os.path.exists(paper_path):
        return jsonify({"error": "Paper not found"}), 404

    # Read the paper's first line to get the title (same logic as main.py)
    with open(paper_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    title = first_line

    # Build the safe_title the same way main.py does
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    safe_title = safe_title.strip().replace(" ", "_")[:50]

    summary_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{safe_title}_summary.md")
    notes_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{safe_title}_notes.md")

    # If exact match doesn't work, try fuzzy: find any summary file that
    # starts with a similar prefix from the paper_id
    if not os.path.exists(summary_path):
        # Try matching by paper_id prefix
        all_summaries = glob.glob(os.path.join(app.config["OUTPUT_FOLDER"], "*_summary.md"))
        # Normalize paper_id for matching
        pid_lower = paper_id.lower().replace("_", "").replace("-", "")
        for sf in all_summaries:
            base = os.path.basename(sf).replace("_summary.md", "")
            base_lower = base.lower().replace("_", "").replace("-", "")
            # Check if paper_id is a substring or vice versa
            if pid_lower[:20] in base_lower or base_lower[:20] in pid_lower:
                summary_path = sf
                notes_path = sf.replace("_summary.md", "_notes.md")
                break

    summary = ""
    notes = ""
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read()
    if os.path.exists(notes_path):
        with open(notes_path, "r", encoding="utf-8") as f:
            notes = f.read()

    if not summary and not notes:
        return jsonify({"found": False, "summary": "", "notes": ""})

    return jsonify({"found": True, "summary": summary, "notes": notes})


@app.route("/api/papers/clear", methods=["POST"])
def api_clear_papers():
    """Delete all papers and their outputs."""
    import shutil
    count = 0
    for f in glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], "*.txt")):
        os.remove(f)
        count += 1
    for f in glob.glob(os.path.join(app.config["OUTPUT_FOLDER"], "*.md")):
        os.remove(f)
    # Clear in-memory state
    processing_status.clear()
    pipeline_progress.clear()
    paper_memories.clear()
    chat_histories.clear()
    return jsonify({"success": True, "removed": count})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload a paper file, return JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file selected"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith((".txt", ".pdf")):
        filename = secure_filename(file.filename)

        if filename.endswith(".pdf"):
            pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(pdf_path)
            try:
                import pymupdf
                doc = pymupdf.open(pdf_path)
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                txt_filename = filename.replace(".pdf", ".txt")
                txt_path = os.path.join(app.config["UPLOAD_FOLDER"], txt_filename)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {txt_filename.replace('.txt', '')}\n\n{'='*60}\n\n{text}")
                os.remove(pdf_path)
                return jsonify({"success": True, "filename": txt_filename})
            except Exception as e:
                return jsonify({"error": f"PDF conversion failed: {e}"}), 500
        else:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            return jsonify({"success": True, "filename": filename})
    else:
        return jsonify({"error": "Only .txt and .pdf files are supported"}), 400


@app.route("/api/fetch", methods=["POST"])
def api_fetch():
    """Fetch papers from arXiv."""
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    max_papers = int(data.get("max", 3))

    if not query:
        return jsonify({"error": "Enter a search query"}), 400

    try:
        from src.utils.paper_fetcher import fetch_papers
        papers = fetch_papers(query, max_results=max_papers)
        return jsonify({"success": True, "count": len(papers)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/process/<paper_id>", methods=["POST"])
def api_process(paper_id):
    """Start processing a paper."""
    paper_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{paper_id}.txt")
    if not os.path.exists(paper_path):
        return jsonify({"error": "Paper not found"}), 404

    if processing_status.get(paper_id) == "processing":
        return jsonify({"error": "Already processing"}), 409

    thread = threading.Thread(target=process_paper_background, args=(paper_path, paper_id))
    thread.start()
    return jsonify({"success": True, "paper_id": paper_id})


@app.route("/api/status/<paper_id>")
def api_status(paper_id):
    """Get processing status and progress."""
    return jsonify({
        "status": processing_status.get(paper_id, "ready"),
        "progress": pipeline_progress.get(paper_id, {}),
    })


@app.route("/api/chat/<paper_id>", methods=["POST"])
def api_chat(paper_id):
    """Chat with agents about a paper.

    Routes the question to the appropriate agent persona based on keywords.
    Uses FAISS to retrieve relevant chunks, then sends to LLM.
    """
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Build or get cached memory
    mem_data = _get_or_build_memory(paper_id)
    if not mem_data:
        return jsonify({"error": "Paper not found"}), 404

    memory = mem_data["memory"]
    paper_data = mem_data["paper_data"]
    title = paper_data["metadata"].get("title", "Untitled")

    # Determine which agent should respond based on question content
    q_lower = question.lower()
    if any(w in q_lower for w in ["review", "evaluat", "score", "strength", "weakness", "critique", "quality", "rate", "assess"]):
        agent = "reviewer"
        agent_label = "Reviewer Agent"
        system_prompt = (
            "You are a Senior Peer Reviewer evaluating an academic paper. "
            "Provide critical, constructive evaluation. Assess strengths and weaknesses. "
            "Be specific with scores and feedback. Reference specific claims from the text."
        )
    elif any(w in q_lower for w in ["write", "summarize", "summary", "draft", "rewrite", "format", "explain simply"]):
        agent = "writer"
        agent_label = "Writer Agent"
        system_prompt = (
            "You are a Technical Writer specializing in clear, structured academic summaries. "
            "Synthesize information into well-organized, readable prose. "
            "Use clear language, avoid unnecessary jargon, and structure your response logically."
        )
    else:
        agent = "researcher"
        agent_label = "Researcher Agent"
        system_prompt = (
            "You are a Senior Research Analyst with expertise in academic papers. "
            "Extract and present accurate, structured information. "
            "Only state information explicitly found in the text. "
            "Be precise and cite specific details (numbers, metrics, names)."
        )

    # Retrieve relevant chunks from FAISS
    search_results = memory.search(query=question, top_k=4)
    context = "\n\n---\n\n".join(r["text"] for r in search_results)

    # Also include summary if available
    summary_context = ""
    summary_files = glob.glob(os.path.join(app.config["OUTPUT_FOLDER"], "*_summary.md"))
    for sf in summary_files:
        with open(sf, "r", encoding="utf-8") as f:
            summary_context = f.read()
        break

    # Build messages for LLM
    from src.utils.llm import chat as llm_chat

    user_content = (
        f"Paper: {title}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"RELEVANT EXCERPTS FROM PAPER:\n{context}"
    )
    if summary_context:
        user_content += f"\n\nGENERATED SUMMARY (for reference):\n{summary_context[:1500]}"

    try:
        response = llm_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
    except Exception as e:
        return jsonify({"error": f"LLM error: {e}"}), 500

    # Store in chat history
    if paper_id not in chat_histories:
        chat_histories[paper_id] = []
    chat_histories[paper_id].append({
        "role": "user",
        "content": question,
        "timestamp": time.time(),
    })
    chat_histories[paper_id].append({
        "role": "agent",
        "agent": agent,
        "agent_label": agent_label,
        "content": response,
        "timestamp": time.time(),
    })

    return jsonify({
        "agent": agent,
        "agent_label": agent_label,
        "content": response,
    })


@app.route("/api/chat/<paper_id>/history")
def api_chat_history(paper_id):
    """Get chat history for a paper."""
    return jsonify(chat_histories.get(paper_id, []))


# Keep old routes for backwards compatibility
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file selected")
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("index"))
    if file and file.filename.endswith((".txt", ".pdf")):
        filename = secure_filename(file.filename)
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(pdf_path)
            try:
                import pymupdf
                doc = pymupdf.open(pdf_path)
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                txt_filename = filename.replace(".pdf", ".txt")
                txt_path = os.path.join(app.config["UPLOAD_FOLDER"], txt_filename)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {txt_filename.replace('.txt', '')}\n\n{'='*60}\n\n{text}")
                os.remove(pdf_path)
                flash(f"PDF converted and saved as {txt_filename}")
            except Exception as e:
                flash(f"PDF conversion failed: {e}")
                return redirect(url_for("index"))
        else:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            flash(f"Uploaded {filename}")
    else:
        flash("Only .txt and .pdf files are supported")
    return redirect(url_for("index"))


@app.route("/process/<filename>")
def process(filename):
    paper_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(paper_path):
        flash("Paper not found")
        return redirect(url_for("index"))
    paper_id = filename.replace(".txt", "")
    if processing_status.get(paper_id) == "processing":
        flash("Already processing this paper")
        return redirect(url_for("index"))
    thread = threading.Thread(target=process_paper_background, args=(paper_path, paper_id))
    thread.start()
    flash(f"Started processing: {filename}")
    return redirect(url_for("index"))


@app.route("/fetch", methods=["POST"])
def fetch():
    query = request.form.get("query", "").strip()
    max_papers = int(request.form.get("max", 3))
    if not query:
        flash("Enter a search query")
        return redirect(url_for("index"))
    try:
        from src.utils.paper_fetcher import fetch_papers
        papers = fetch_papers(query, max_results=max_papers)
        flash(f"Fetched {len(papers)} paper(s) for '{query}'")
    except Exception as e:
        flash(f"Fetch failed: {e}")
    return redirect(url_for("index"))


@app.route("/status/<paper_id>")
def status(paper_id):
    return jsonify({"status": processing_status.get(paper_id, "unknown")})


if __name__ == "__main__":
    os.makedirs("papers", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    print("\n  Multi-Agent Research Assistant")
    print("   Open http://localhost:8080 in your browser\n")
    app.run(debug=False, host="127.0.0.1", port=8080)
