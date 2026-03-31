# Multi-Agent Research Assistant

A LangGraph-orchestrated pipeline with 3 specialized AI agents that automatically analyze academic papers. Upload a PDF, and the system extracts key information, writes a structured summary, and quality-checks it through an iterative review loop.

## How It Works

```
Paper (PDF/TXT) --> Chunk & Embed (FAISS) --> Researcher Agent --> Writer Agent --> Reviewer Agent
                                                                       ^                |
                                                                       |   Reject (max 3x)
                                                                       +----------------+
```

**Researcher Agent** — Queries FAISS with 5 targeted questions, extracts structured facts from the paper using retrieved chunks + LLM.

**Writer Agent** — Synthesizes research notes into a 6-section markdown summary (Overview, Key Findings, Methodology, Results, Limitations, Significance).

**Reviewer Agent** — Scores the summary on accuracy, completeness, clarity, and structure (1-10 each). All scores must be >= 7 to approve. Otherwise loops back to the Writer with specific feedback.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your Groq API key:
# GROQ_API_KEY=your-key-here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 3. Run the web app

```bash
python app.py
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

## Web UI

3-panel interface:

- **Left** — Upload PDFs, fetch from arXiv, manage paper library
- **Center** — Chat with agents about any selected paper
- **Right** — View generated summaries, research notes, and pipeline progress

The chat routes questions to the right agent automatically: evaluation questions go to the Reviewer, summary requests to the Writer, and everything else to the Researcher.

## CLI Usage

```bash
# Process a single paper
python -m src.main papers/your_paper.txt

# Batch process all papers
python -m src.main --batch

# Fetch papers from arXiv then process
python -m src.main --fetch
```

## Project Structure

```
multiagent/
├── app.py                     # Flask web server + REST API
├── requirements.txt           # Python dependencies
├── .env.example               # API key template
├── src/
│   ├── main.py                # Pipeline entry point
│   ├── state.py               # ResearchState schema
│   ├── graph.py               # LangGraph workflow + conditional edges
│   ├── agents/
│   │   ├── researcher.py      # Agent 1: FAISS retrieval + extraction
│   │   ├── writer.py          # Agent 2: Summary synthesis + revision
│   │   └── reviewer.py        # Agent 3: Scoring + approval gate
│   ├── memory/
│   │   └── shared_memory.py   # FAISS vector store (384-dim, MiniLM)
│   └── utils/
│       ├── llm.py             # Groq API client with rate-limit retry
│       ├── paper_loader.py    # Token-based chunker (500 tok, 100 overlap)
│       └── paper_fetcher.py   # arXiv search + PDF text extraction
├── papers/                    # Input paper files
├── output/                    # Generated summaries and notes
└── templates/
    ├── index.html             # Main SPA (3-panel layout)
    └── summary.html           # Summary detail view
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph + conditional edges) |
| LLM | Groq API — llama-3.1-8b-instant |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, local) |
| Vector Search | FAISS (IndexFlatL2, brute-force) |
| Web | Flask + vanilla JS SPA |
| PDF Extraction | PyMuPDF |
| Paper Source | arXiv API |
| Tokenizer | tiktoken |

## Key Design Decisions

- **3 agents, not 1 prompt** — Extraction, synthesis, and evaluation are separate cognitive tasks. Combining them degrades all three.
- **RAG over full-context** — FAISS retrieval sends ~7.5K tokens instead of 25-100K, enabling use with small-context models.
- **Temperature tuning** — Researcher: 0.1 (precise), Writer: 0.3/0.2 (fluent), Reviewer: 0.1 (consistent).
- **Circuit breaker** — Max 3 revision loops prevents infinite API calls.
- **Local embeddings** — all-MiniLM-L6-v2 runs on CPU, no API cost.
