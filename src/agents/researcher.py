"""
Researcher Agent

ROLE: Extract structured information from academic papers.
POSITION IN PIPELINE: First agent. Runs after paper is loaded into FAISS.

WHAT IT DOES:
1. Asks targeted questions about the paper (methodology, results, etc.)
2. For each question, queries FAISS to retrieve relevant chunks
3. Sends chunks + question to OpenAI to extract structured notes
4. Returns comprehensive research notes for the Writer agent

WHY THIS AGENT EXISTS (separation of concerns):
- Extraction and synthesis are different cognitive tasks
- The Researcher is optimized for FINDING and EXTRACTING facts
- The Writer (next agent) is optimized for ORGANIZING and WRITING
- Combining both in one prompt degrades both tasks

SYSTEM DESIGN PARALLEL:
- This is like an ETL pipeline's "Extract" step
- Or a microservice that reads from a database and normalizes data
- It talks to the data layer (FAISS) and produces structured output
"""

from src.utils.llm import chat

from src.memory.shared_memory import SharedMemory
from src.state import ResearchState


# ---------------------------------------------------------------
# RESEARCH QUESTIONS
# ---------------------------------------------------------------
# These are the targeted queries the Researcher asks FAISS.
# Each question targets a DIFFERENT section of the paper.
#
# WHY these specific questions?
# Academic papers follow a standard structure (IMRAD):
#   Introduction, Methodology, Results, And Discussion
# Our questions map to this structure to ensure full coverage.
#
# In production, you could make these dynamic — have the LLM
# generate questions based on the paper's abstract first.
# ---------------------------------------------------------------

RESEARCH_QUESTIONS = [
    "What is the main research question or objective of this paper?",
    "What methodology or approach did the authors use?",
    "What are the key results and findings?",
    "What are the main contributions of this work?",
    "What are the limitations and future work mentioned?",
]

# ---------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------
# This is the Researcher's IDENTITY. It tells the LLM:
#   - WHO you are (role)
#   - WHAT you optimize for (goal)
#   - HOW you behave (constraints)
#
# CrewAI pattern: role + goal + backstory
# We implement this directly in the system prompt.
#
# WHY so specific? Vague prompts get vague results.
# "Summarize this" → generic output
# "You are a research analyst, extract only stated facts" → precise output
# ---------------------------------------------------------------

SYSTEM_PROMPT = """You are a Senior Research Analyst with expertise in reading and \
extracting information from academic papers.

GOAL: Extract accurate, structured information from the provided paper excerpts.

CONSTRAINTS:
- Only extract information that is EXPLICITLY stated in the text
- Do NOT infer, assume, or add information not present
- If information is not available in the provided context, say "Not found in provided context"
- Be precise and cite specific details (numbers, metrics, names)

You will receive excerpts from a paper (retrieved via semantic search) and a specific \
question. Extract the relevant information to answer that question."""


def run_researcher(state: ResearchState, memory: SharedMemory) -> dict:
    """Execute the Researcher agent.

    This is the function that becomes a NODE in the LangGraph.
    It takes the current state, does its work, and returns
    the fields it wants to update.

    Args:
        state: Current workflow state (reads paper_text, paper_chunks)
        memory: Shared FAISS memory (queries it for relevant chunks)

    Returns:
        dict with "research_notes" key — the extracted information
        LangGraph merges this back into the state automatically.

    FLOW:
        For each research question:
            1. Query FAISS → get top 3 relevant chunks
            2. Send chunks + question to OpenAI
            3. Collect the extracted answer
        Combine all answers into structured research notes
    """
    all_notes = []

    # --- STEP 1: Ask each targeted question ---
    for question in RESEARCH_QUESTIONS:

        # Query FAISS: "find me the chunks most relevant to this question"
        # top_k=3 means return the 3 best matches
        # WHY 3? Good balance — 1 might miss context, 5+ adds noise
        search_results = memory.search(query=question, top_k=3)

        # Combine the retrieved chunks into one context block
        context_parts = []
        for result in search_results:
            context_parts.append(result["text"])

        context = "\n\n---\n\n".join(context_parts)

        # --- STEP 2: Send to LLM for extraction ---
        answer = chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Based on the following excerpts from an academic paper, "
                        f"answer this question:\n\n"
                        f"QUESTION: {question}\n\n"
                        f"PAPER EXCERPTS:\n{context}"
                    ),
                },
            ],
            temperature=0.1,
        )

        # Store as a structured note
        all_notes.append(f"### {question}\n{answer}")

    # --- STEP 3: Combine all notes ---
    # Join all Q&A pairs into one structured document
    # The Writer agent will read this to write the summary
    research_notes = (
        f"# Research Notes: {state.get('metadata', {}).get('title', 'Untitled')}\n\n"
        + "\n\n".join(all_notes)
    )

    # Return ONLY the fields we want to update in state
    # LangGraph merges this into the existing state
    # We don't return paper_text, chunks, etc. — those stay unchanged
    return {"research_notes": research_notes}
