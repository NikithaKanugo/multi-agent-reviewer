"""
Writer Agent

ROLE: Synthesize research notes into a polished, structured summary.
POSITION IN PIPELINE: Second agent. Runs after Researcher, before Reviewer.

WHAT IT DOES:
- First run: reads research_notes, writes a structured draft_summary
- Revision runs: reads research_notes + review_feedback, rewrites the summary

WHY THIS AGENT EXISTS:
- Extraction (Researcher) and synthesis (Writer) are different tasks
- A dedicated Writer produces more coherent, better-structured output
- Separation allows us to iterate on writing quality without affecting extraction

KEY DESIGN: Context-aware handoff
- The Writer checks revision_count to know if it's a first draft or revision
- On revision, it reads review_feedback to know WHAT to fix
- This is inter-agent communication through shared state

SYSTEM DESIGN PARALLEL:
- This is like a "Transform" step in an ETL pipeline
- Or a templating service that takes structured data and produces a document
- It consumes output from one service (Researcher) and produces input for another (Reviewer)
"""

from src.utils.llm import chat

from src.state import ResearchState


# ---------------------------------------------------------------
# SYSTEM PROMPT — First Draft
# ---------------------------------------------------------------
# When writing for the FIRST time (no feedback yet).
# Higher temperature (0.3) than Researcher (0.1) because:
#   - Writing benefits from some natural language variety
#   - But not too high — we want accuracy over creativity
#   - 0.3 is the sweet spot: fluent but faithful
# ---------------------------------------------------------------

SYSTEM_PROMPT_FIRST_DRAFT = """You are a Technical Writer specializing in producing \
clear, structured summaries of academic research papers.

GOAL: Transform research notes into a well-organized, readable research summary.

OUTPUT FORMAT (you MUST follow this exact structure):
# [Paper Title]

## Overview
A 2-3 sentence high-level summary of what the paper is about and why it matters.

## Key Findings
- Bullet point 1
- Bullet point 2
- (as many as needed)

## Methodology
A clear explanation of the approach/method used, accessible to someone with a technical \
background but not necessarily in this specific field.

## Results
Specific results with numbers and metrics where available.

## Limitations
What the authors acknowledge as limitations or open questions.

## Significance
Why this work matters and what it enables for future research.

CONSTRAINTS:
- Use ONLY information from the provided research notes
- Do NOT add information not present in the notes
- If a section has no relevant information, write "Not covered in the source material"
- Use clear, concise language — avoid jargon where possible
- Include specific numbers and metrics when available in the notes"""


# ---------------------------------------------------------------
# SYSTEM PROMPT — Revision
# ---------------------------------------------------------------
# When REWRITING based on Reviewer feedback.
# This is a DIFFERENT prompt because the task is different:
#   - First draft: create from scratch
#   - Revision: fix specific issues while keeping what's good
#
# WHY a separate prompt instead of appending "also fix this"?
# Cleaner separation of concerns. The revision prompt explicitly
# tells the LLM to focus on the feedback, not start over.
# This produces targeted fixes instead of complete rewrites.
# ---------------------------------------------------------------

SYSTEM_PROMPT_REVISION = """You are a Technical Writer revising a research summary \
based on reviewer feedback.

GOAL: Improve the existing summary by addressing the specific feedback provided.

APPROACH:
- Read the reviewer's feedback carefully
- Fix ONLY the issues mentioned in the feedback
- Keep everything that was NOT criticized — don't rewrite good sections
- Maintain the same output format and structure

OUTPUT FORMAT (same as original):
# [Paper Title]

## Overview
## Key Findings
## Methodology
## Results
## Limitations
## Significance

CONSTRAINTS:
- Address each piece of feedback specifically
- Do NOT remove correct information while fixing issues
- Do NOT add information not present in the research notes
- Preserve the good parts of the original draft"""


def run_writer(state: ResearchState) -> dict:
    """Execute the Writer agent.

    This function handles BOTH first drafts and revisions.
    It checks the state to determine which mode to operate in.

    Args:
        state: Current workflow state
            Reads: research_notes, review_feedback, revision_count,
                   draft_summary (for revisions), metadata

    Returns:
        dict with "draft_summary" key — the written summary
        LangGraph merges this into state.
    """
    is_revision = state.get("revision_count", 0) > 0
    title = state.get("metadata", {}).get("title", "Untitled")

    if is_revision:
        system_prompt = SYSTEM_PROMPT_REVISION
        user_message = (
            f"Here is the original research summary that needs revision:\n\n"
            f"{state['draft_summary']}\n\n"
            f"---\n\n"
            f"REVIEWER FEEDBACK (address each point):\n"
            f"{state['review_feedback']}\n\n"
            f"---\n\n"
            f"ORIGINAL RESEARCH NOTES (use as source of truth):\n"
            f"{state['research_notes']}"
        )
        temperature = 0.2
    else:
        system_prompt = SYSTEM_PROMPT_FIRST_DRAFT
        user_message = (
            f"Write a structured research summary for the following paper.\n\n"
            f"PAPER TITLE: {title}\n\n"
            f"RESEARCH NOTES:\n{state['research_notes']}"
        )
        temperature = 0.3

    draft_summary = chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )

    # Return only the field we own
    return {"draft_summary": draft_summary}
