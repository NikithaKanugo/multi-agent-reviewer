"""
Shared state schema for the Multi-Agent Research Assistant.

This is the SINGLE SOURCE OF TRUTH for all data flowing between agents.
Every agent reads from and writes to this state. LangGraph manages it automatically.

Think of it as a form being passed around a team:
- Researcher fills in research_notes
- Writer fills in draft_summary
- Reviewer fills in review_feedback and is_approved
"""

from typing import TypedDict


class ResearchState(TypedDict):
    # --- INPUT ---
    # Raw paper text fed into the system
    paper_text: str

    # Paper split into smaller chunks for FAISS vector search
    # WHY chunks? Full papers are too long for LLM context + we need
    # precise retrieval, not the whole paper every time
    paper_chunks: list[str]

    # Paper metadata (title, authors, etc.) extracted during loading
    metadata: dict

    # --- RESEARCHER OUTPUT ---
    # Structured research notes extracted by the Researcher agent
    # Contains: methodology, key findings, contributions, limitations
    research_notes: str

    # --- WRITER OUTPUT ---
    # The formatted research summary written by the Writer agent
    draft_summary: str

    # --- REVIEWER OUTPUT ---
    # Specific feedback from the Reviewer on what needs improvement
    review_feedback: str

    # Whether the Reviewer approved the summary
    # This field drives the conditional edge in the graph:
    #   True  → workflow ends (summary is good)
    #   False → route back to Writer for revision
    is_approved: bool

    # --- CONTROL FLOW ---
    # How many times the Writer has revised (incremented each loop)
    revision_count: int

    # Maximum allowed revisions before forcing completion
    # This prevents infinite loops and runaway API costs
    max_revisions: int
