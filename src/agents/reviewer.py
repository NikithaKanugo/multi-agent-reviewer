"""
Reviewer Agent

ROLE: Quality gate — evaluate the Writer's summary and approve or reject.
POSITION IN PIPELINE: Third (final) agent. Controls the conditional edge.

WHAT IT DOES:
- Compares draft_summary against research_notes (source of truth)
- Scores on 4 dimensions: accuracy, completeness, clarity, structure
- If all scores >= 7: approve → workflow ends
- If any score < 7: reject with specific feedback → Writer revises

WHY THIS AGENT IS SPECIAL:
- Only agent that controls FLOW (approve/reject = conditional edge)
- Only agent that can send work BACKWARD (reject → back to Writer)
- This creates the CYCLE in the graph — without it, we'd have a simple pipeline

SYSTEM DESIGN PARALLEL:
- CI/CD: tests pass → deploy, tests fail → back to code
- PR review: approved → merge, changes requested → revise
- QA gate in a data pipeline: data quality check before publishing

KEY DESIGN: Structured evaluation rubric
- Not "is this good?" (unreliable, inconsistent)
- Instead: "score accuracy 1-10 where 10 means..." (precise, repeatable)
- This is how production LLM evaluation works (LLM-as-judge pattern)
"""

import json

from src.utils.llm import chat

from src.state import ResearchState


# ---------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------
# The Reviewer has the MOST constrained prompt because its output
# directly controls the workflow. A bad review = bad routing.
#
# CRITICAL: We ask for JSON output.
# WHY? Because we need to PARSE the scores programmatically.
# Free-text like "I'd give accuracy an 8" is hard to parse reliably.
# JSON like {"accuracy": 8} is machine-readable.
#
# Temperature = 0.1 (lowest of all agents)
# WHY? Evaluation must be CONSISTENT. If the same summary gets
# approved on one run and rejected on another, the system is
# unreliable. Low temperature = deterministic evaluation.
# ---------------------------------------------------------------

SYSTEM_PROMPT = """You are a Senior Peer Reviewer with extensive experience evaluating \
academic research summaries. You are meticulous, fair, and specific in your feedback.

GOAL: Evaluate the quality of a research summary by comparing it against the original \
research notes.

You MUST respond in this EXACT JSON format (no other text, just the JSON):
{
    "accuracy": <1-10>,
    "accuracy_feedback": "<specific issues or 'No issues found'>",
    "completeness": <1-10>,
    "completeness_feedback": "<what's missing or 'All key points covered'>",
    "clarity": <1-10>,
    "clarity_feedback": "<what's unclear or 'Well written'>",
    "structure": <1-10>,
    "structure_feedback": "<structural issues or 'Properly structured'>",
    "overall_assessment": "<1-2 sentence overall judgment>",
    "is_approved": <true if ALL scores >= 7, false otherwise>
}

SCORING GUIDE:
- accuracy (1-10): Does every claim in the summary match the research notes? \
Deduct points for hallucinated or unsupported claims.
- completeness (1-10): Are all key findings, methodology, results, and limitations \
from the notes covered? Deduct points for missing information.
- clarity (1-10): Is the summary clear, well-written, and easy to understand? \
Deduct points for jargon without explanation, awkward phrasing, or confusing sentences.
- structure (1-10): Does the summary follow the expected format with all sections? \
Deduct points for missing sections, wrong ordering, or inconsistent formatting.

IMPORTANT:
- Be SPECIFIC in feedback — "Results section is missing" not "could be better"
- Compare against the research notes, not your own knowledge
- A score of 7 is the MINIMUM for approval on each dimension
- If rejecting, your feedback should tell the Writer EXACTLY what to fix"""


def run_reviewer(state: ResearchState) -> dict:
    """Execute the Reviewer agent.

    Evaluates the draft summary against research notes and decides
    whether to approve or send back for revision.

    Args:
        state: Current workflow state
            Reads: draft_summary, research_notes, revision_count, max_revisions

    Returns:
        dict with:
            - "review_feedback": specific feedback for the Writer
            - "is_approved": True/False (drives the conditional edge)
            - "revision_count": incremented if rejected
    """
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    if revision_count >= max_revisions:
        return {
            "review_feedback": "Maximum revisions reached. Accepting current version.",
            "is_approved": True,
            "revision_count": revision_count,
        }

    user_message = (
        f"Evaluate the following research summary against the original research notes.\n\n"
        f"=== DRAFT SUMMARY (evaluate this) ===\n"
        f"{state['draft_summary']}\n\n"
        f"=== ORIGINAL RESEARCH NOTES (source of truth) ===\n"
        f"{state['research_notes']}"
    )

    raw_response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )

    # -------------------------------------------------------
    # PARSE THE JSON RESPONSE
    # -------------------------------------------------------
    # The LLM should return valid JSON because we told it to.
    # But LLMs are not perfect — they might add markdown fences
    # or extra text. We handle common failure modes.
    #
    # In production, you'd use OpenAI's JSON mode:
    #   response_format={"type": "json_object"}
    # This GUARANTEES valid JSON. We parse manually here for
    # educational clarity — you should know what happens under the hood.
    # -------------------------------------------------------
    try:
        # Strip markdown code fences if present (```json ... ```)
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            # Remove first and last lines (the fences)
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])

        review = json.loads(cleaned)
    except json.JSONDecodeError:
        # If parsing fails, reject and ask for revision
        # WHY reject on parse failure? Because we can't verify quality
        # if we can't read the review. Fail safe = reject.
        return {
            "review_feedback": (
                "Review parsing failed. Please revise the summary to ensure "
                "it follows the expected format with all required sections."
            ),
            "is_approved": False,
            "revision_count": revision_count + 1,
        }

    # -------------------------------------------------------
    # BUILD FEEDBACK
    # -------------------------------------------------------
    # Regardless of approval, we compile the feedback.
    # Even approved summaries get feedback — useful for logging.
    #
    # For rejected summaries, this feedback goes to the Writer
    # agent, which reads it to know WHAT to fix.
    # -------------------------------------------------------
    is_approved = review.get("is_approved", False)

    scores = {
        "accuracy": review.get("accuracy", 0),
        "completeness": review.get("completeness", 0),
        "clarity": review.get("clarity", 0),
        "structure": review.get("structure", 0),
    }

    feedback_parts = [
        f"## Review (Revision {revision_count + 1})",
        f"**Scores:** Accuracy={scores['accuracy']}/10, "
        f"Completeness={scores['completeness']}/10, "
        f"Clarity={scores['clarity']}/10, "
        f"Structure={scores['structure']}/10",
        f"**Overall:** {review.get('overall_assessment', 'N/A')}",
    ]

    # Add specific feedback for dimensions that need improvement
    if not is_approved:
        feedback_parts.append("\n**Issues to fix:**")
        for dimension, score in scores.items():
            if score < 7:
                feedback_key = f"{dimension}_feedback"
                feedback_text = review.get(feedback_key, "No details provided")
                feedback_parts.append(f"- **{dimension.title()}** ({score}/10): {feedback_text}")

    review_feedback = "\n".join(feedback_parts)

    # -------------------------------------------------------
    # RETURN UPDATED STATE
    # -------------------------------------------------------
    # Three fields updated:
    #   - review_feedback: what the Writer needs to know
    #   - is_approved: drives the conditional edge (END or loop back)
    #   - revision_count: incremented on rejection (tracks loop count)
    # -------------------------------------------------------
    return {
        "review_feedback": review_feedback,
        "is_approved": is_approved,
        "revision_count": revision_count + 1 if not is_approved else revision_count,
    }
