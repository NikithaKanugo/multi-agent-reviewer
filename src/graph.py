"""
LangGraph Workflow Orchestration

This file WIRES EVERYTHING TOGETHER. It defines:
- Nodes: each agent function becomes a node
- Edges: who runs after who
- Conditional edges: Reviewer's approve/reject routing
- The compiled graph: a runnable workflow

THREE LANGGRAPH CONCEPTS:
1. StateGraph — the container. It knows the state schema and holds all nodes/edges.
2. Nodes — functions that take state, do work, return partial state updates.
3. Edges — connections between nodes. Normal (always A→B) or conditional (check state, decide).

SYSTEM DESIGN PARALLEL:
- This is like AWS Step Functions — define a workflow as a state machine
- Or like a CI/CD pipeline definition (GitHub Actions YAML → this is the Python equivalent)
- Or an API gateway that routes requests to different microservices based on conditions

WHY LANGGRAPH OVER PLAIN PYTHON?
Plain Python version of this would be:
    notes = researcher(state)
    state.update(notes)
    while True:
        draft = writer(state)
        state.update(draft)
        review = reviewer(state)
        state.update(review)
        if review["is_approved"]:
            break
This works but:
- No checkpointing (can't resume mid-workflow)
- No visualization (can't see the flow)
- State management is manual (error-prone)
- Adding agents means rewriting the loop
LangGraph gives us all of this for free.
"""

from langgraph.graph import StateGraph, END

from src.state import ResearchState
from src.memory.shared_memory import SharedMemory
from src.agents.researcher import run_researcher
from src.agents.writer import run_writer
from src.agents.reviewer import run_reviewer


def build_graph(memory: SharedMemory) -> StateGraph:
    """Build and compile the multi-agent research workflow.

    This function constructs the entire workflow graph:
        START → Researcher → Writer → Reviewer → (approve? END : Writer)

    Args:
        memory: Shared FAISS memory instance.
            WHY passed in? Because the Researcher needs it to query FAISS.
            We pass it here so the graph "closes over" it — the Researcher
            node function can access it without it being in the state.
            This keeps the state clean (no FAISS objects in the state dict).

    Returns:
        Compiled StateGraph ready to be invoked with .invoke(initial_state)
    """

    # -------------------------------------------------------
    # STEP 1: Create the graph with our state schema
    # -------------------------------------------------------
    # StateGraph(ResearchState) tells LangGraph:
    # "The state flowing through this graph has this exact shape"
    # Every node receives this state and returns partial updates.
    # -------------------------------------------------------
    graph = StateGraph(ResearchState)

    # -------------------------------------------------------
    # STEP 2: Add nodes (each agent becomes a node)
    # -------------------------------------------------------
    # A node is: name (string) + function (callable)
    # The function MUST:
    #   - Accept state (ResearchState) as first argument
    #   - Return a dict with the state fields it wants to update
    #
    # WHY wrap run_researcher in a lambda?
    # Because run_researcher takes TWO args (state, memory)
    # but LangGraph nodes only receive ONE arg (state).
    # The lambda "binds" the memory argument so the node
    # function signature matches what LangGraph expects.
    # This is called a CLOSURE — the function "closes over"
    # the memory variable from the outer scope.
    # -------------------------------------------------------

    # Node 1: Researcher — extracts info from paper via FAISS + LLM
    graph.add_node(
        "researcher",
        lambda state: run_researcher(state, memory),
    )

    # Node 2: Writer — synthesizes research notes into a summary
    graph.add_node(
        "writer",
        lambda state: run_writer(state),
    )

    # Node 3: Reviewer — evaluates quality, approves or rejects
    graph.add_node(
        "reviewer",
        lambda state: run_reviewer(state),
    )

    # -------------------------------------------------------
    # STEP 3: Define edges (the flow between nodes)
    # -------------------------------------------------------

    # Entry point: the graph STARTS at the Researcher
    # This is like setting the first step in a CI/CD pipeline
    graph.set_entry_point("researcher")

    # Researcher → Writer (ALWAYS)
    # After research is done, always go to writing. No conditions.
    graph.add_edge("researcher", "writer")

    # Writer → Reviewer (ALWAYS)
    # After writing is done, always go to review. No conditions.
    graph.add_edge("writer", "reviewer")

    # -------------------------------------------------------
    # STEP 4: Conditional edge after Reviewer
    # -------------------------------------------------------
    # This is THE KEY PIECE — the conditional routing.
    #
    # After the Reviewer runs, we check the state:
    #   - is_approved == True  → go to END (workflow complete)
    #   - is_approved == False → go to "writer" (revision loop)
    #
    # add_conditional_edges takes:
    #   1. Source node: "reviewer"
    #   2. Routing function: checks state, returns next node name
    #   3. Path map: maps return values to actual node names
    #
    # The routing function is WHERE THE DECISION HAPPENS.
    # It's a pure function — reads state, returns a string.
    # No side effects, no API calls, just a decision.
    # -------------------------------------------------------
    graph.add_conditional_edges(
        "reviewer",                     # After this node...
        should_continue,                # ...run this function to decide...
        {
            "writer": "writer",         # if it returns "writer" → go to Writer
            "end": END,                 # if it returns "end" → stop the workflow
        },
    )

    # -------------------------------------------------------
    # STEP 5: Compile the graph
    # -------------------------------------------------------
    # .compile() validates the graph and makes it runnable.
    # After this, you can call: graph.invoke(initial_state)
    #
    # Compilation checks:
    # - All nodes referenced in edges exist
    # - Entry point is set
    # - No orphan nodes (nodes with no edges)
    # - State schema is valid
    #
    # If any check fails, you get a clear error at compile time,
    # not a mystery crash at runtime. This is like type checking
    # for workflows.
    # -------------------------------------------------------
    return graph.compile()


def should_continue(state: ResearchState) -> str:
    """Routing function for the conditional edge after Reviewer.

    This function is the DECISION POINT of the entire workflow.
    It reads the state and returns the name of the next node.

    Logic:
        if approved → "end" (workflow complete, summary is good)
        if not approved → "writer" (revision needed, loop back)

    Args:
        state: Current workflow state (reads is_approved)

    Returns:
        "end" or "writer" — maps to the path_map in add_conditional_edges
    """
    if state.get("is_approved", False):
        return "end"
    return "writer"
