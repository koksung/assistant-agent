import re
import inspect
from typing import List, Dict, Any

from app.tool_registry import get_tools, get_registry, ToolCallContext
from app.belief_system import update_beliefs
from app.users.user_task import UserTask
from app.users.session import ConversationContext
from app.utils.logger import get_logger
from app.utils.tool_utils import format_tool_descriptions

logger = get_logger(__name__)

# ----------------------------- Planning utils --------------------------------

PLAN_LINE_RE = r"Use '([\w\.\-]+)'(?: on (.*))?"

def parse_plan(plan_str: str) -> List[Dict[str, str]]:
    """
    Parse lines like:
      Use 'pdf.extract' on file path
      Use 'summary.generate' on user query + extracted text
    Also supports concrete tool names (e.g., 'local_pdf_extractor').
    """
    steps: List[Dict[str, str]] = []
    for line in str(plan_str).splitlines():
        m = re.search(PLAN_LINE_RE, line)
        if not m:
            continue
        token = m.group(1)
        target = (m.group(2) or "").strip()
        steps.append({"token": token, "input_target": target})
    return steps


def build_tool_input(step: Dict[str, str], task: UserTask, session: ConversationContext) -> Dict[str, Any]:
    """
    Heuristics to map the planner's 'on ...' phrase to kwargs for tools.
    Extend as needed.
    """
    target = step.get("input_target", "").lower()

    if "raw text" in target or "extracted text" in target:
        return {"text": task.raw_text}
    if "user query" in target:
        return {"query": task.user_query}
    if "beliefs" in target:
        return {"beliefs": session.belief_state}
    if "file path" in target or "pdf" in target or target == "":
        # default to pdf_path if not specified
        return {"pdf_path": task.file_path}

    # fall-through: supply a broad context
    return {
        "text": task.raw_text,
        "query": task.user_query,
        "beliefs": session.belief_state,
        "pdf_path": task.file_path,
    }


def build_summary_prompt(
    task: UserTask,
    intermediate_outputs: List[Dict[str, Any]],
    belief_state: Dict[str, Any],
    user_input: str
) -> str:
    belief = (belief_state or {}).get("belief", {})
    archetypes = belief.get("archetype_probs", {}) or {}
    preferences = belief.get("preferences", {}) or {}

    formatted_archetypes = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v:.2f}" for k, v in archetypes.items()
    ) or "Unknown"
    formatted_preferences = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}" for k, v in preferences.items()
    ) or "Not specified"

    intermediate_summary = ""
    tool_trace = ""
    for item in intermediate_outputs:
        cap = item.get("capability", "unknown.capability")
        tool_name = item.get("tool", "unknown_tool")
        output = str(item.get("output", "")).strip()
        tool_trace += f"- {cap} → {tool_name}\n"
        intermediate_summary += f"\n### Output from `{tool_name}` ({cap})\n{output}\n"

    return f"""
    You are an academic research assistant helping summarize scientific papers.
    
    ## User Query
    "{user_input}"
    
    ## User Belief Profile
    
    ### Archetype Probabilities
    {formatted_archetypes}
    
    ### User Preferences
    {formatted_preferences}
    
    ## Tools Chosen by Router
    {tool_trace if tool_trace else "- (none)"}
    
    ## Tool Outputs for Reference
    {intermediate_summary}
    
    ## Your Task
    Write a clean, markdown-formatted summary of the paper. Include:
    - Appropriate section headers (e.g., ## Abstract, ## Methodology)
    - Highlights aligned with the user’s archetype
    - Clear structure, precise explanations, and user-adaptive tone
    """.strip()


# ---------------------------- Routing helpers --------------------------------

def _build_tool_ctx_from_session(session: ConversationContext) -> ToolCallContext:
    """
    Convert the session's belief-derived routing context into ToolCallContext.
    Uses safe defaults in case attributes are missing on session.
    """
    belief_state = session.belief_state or {}
    rc = belief_state.get("routing_ctx", {}) or {}

    return ToolCallContext(
        user_id=getattr(session, "user_id", "anonymous"),
        archetypes=rc.get("archetypes", []) or [],
        preferences=rc.get("preferences", {}) or {},
        allow_remote=getattr(session, "allow_remote", True),
        privacy_level=getattr(session, "privacy_level", "standard"),
        max_latency_ms=getattr(session, "latency_budget_ms", None),
        budget_level=getattr(session, "budget_level", None),
    )


def _enhance_doc_prefs_with_preview(session: ConversationContext, preview_text: str) -> None:
    """
    Cheap doc-aware hints that improve first routing: detect equations/layout references.
    Mutates session.belief_state['routing_ctx']['preferences'].
    """
    belief_state = session.belief_state or {}
    rc = belief_state.get("routing_ctx", {}) or {}
    prefs = dict(rc.get("preferences", {}) or {})
    text = (preview_text or "").lower()

    eq_hint = any(w in text for w in ("equation", "latex", "proof", "derivation"))
    layout_hint = any(w in text for w in ("table", "figure", "diagram", "two-column", "layout"))

    prefs["doc_has_equations"] = prefs.get("doc_has_equations") or eq_hint
    prefs["has_complex_layout"] = prefs.get("has_complex_layout") or layout_hint

    # write back
    rc["preferences"] = prefs
    belief_state["routing_ctx"] = rc
    session.belief_state = belief_state


def _format_capability_menu(tools: Dict[str, Any]) -> str:
    """
    Present both: (a) high-level capabilities, (b) concrete tools for reference.
    """
    capability_hints = [
        "- pdf.extract: Extracts structured text/sections from PDFs",
        "- equations: Extracts/retains LaTeX and equations where applicable",
        "- layout: Emphasizes structural fidelity in complex layouts",
        "- summary.generate: Produce an adaptive summary from gathered context",
    ]
    return (
        "## Available Capabilities (router will choose the tool):\n"
        + "\n".join(capability_hints)
        + "\n\n## Concrete Tools (for your reference only — do not hard-code one):\n"
        + format_tool_descriptions(tools)
    )


# Robustly invoke a LangChain StructuredTool across versions
async def _ainvoke_structured_tool(tool, kwargs: Dict[str, Any]):
    if hasattr(tool, "ainvoke"):
        return await tool.ainvoke(kwargs)
    # sync path
    if hasattr(tool, "invoke"):
        res = tool.invoke(kwargs)
    else:
        try:
            res = tool.run(kwargs)
        except TypeError:
            res = tool.run(**kwargs)
    if inspect.isawaitable(res):
        return await res
    return res


# --------------------------- Orchestration flow -------------------------------

async def prepare_user_task(file, user_query: str, session: ConversationContext, llms: Dict[str, Any]):
    """
    Entry point when a user uploads a PDF + query.
    Saves file, routes initial extraction, seeds beliefs, runs the conversation loop.
    """
    content = await file.read()
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(content)

    task = UserTask(
        file_path=file_path,
        raw_text="",  # will fill after routed extraction
        user_query=user_query,
        task_type="summarize",
        metadata={"filename": file.filename}
    )

    session.set_task(task)
    session.add_message("user", user_query)

    # Seed beliefs (if absent) using the user's query
    if not session.belief_state:
        insight = update_beliefs(user_query)
        session.update_beliefs(insight)

    # Route the initial PDF extraction (instead of hard-coding local extractor)
    reg = get_registry()
    if not reg:
        # Safety: if registry is not initialized, fall back to a minimal local path
        from app.tools.local.pdf_extractor import extract_pdf_text
        task.raw_text = extract_pdf_text(file_path)
    else:
        # Optional: add doc-aware hints using a filename sniff (cheap) before routing
        _enhance_doc_prefs_with_preview(session, file.filename)

        ctx = _build_tool_ctx_from_session(session)
        prof = reg.resolve("pdf.extract", ctx)
        if not prof:
            raise RuntimeError("No tool supports capability 'pdf.extract'")

        logger.info(f"[Router]: Selected '{prof.tool.name}' for pdf.extract (requires_network={prof.requires_network})")
        task.raw_text = await _ainvoke_structured_tool(prof.tool, {"pdf_path": file_path})

    # Now that we have text, run the conversation loop
    response = await conversation_loop(user_query, task, session, llms)
    return response


async def conversation_loop(user_input: str, task: UserTask, session: ConversationContext, llms: Dict[str, Any]):
    """
    Plan (in terms of capabilities), route each step, execute tools, and summarize.
    Also updates beliefs at the end with the full context.
    """
    tools = get_tools()  # for the LLM’s reference list
    reg = get_registry()
    orchestrator_llm = llms["orchestrator"]
    summarizer_llm = llms["summarizer"]

    # Add doc hints from actual text preview before planning (improves routing on step 1)
    preview = (task.get_full_text() or "")[:1200]
    _enhance_doc_prefs_with_preview(session, preview)

    # Format belief profile for the planner (human-readable)
    belief = (session.belief_state or {}).get("belief", {})
    archetypes = belief.get("archetype_probs", {}) or {}
    preferences = belief.get("preferences", {}) or {}
    formatted_archetypes = "\n".join(f"- {k}: {v:.2f}" for k, v in archetypes.items()) or "Unknown"
    formatted_prefs = "\n".join(f"- {k}: {v}" for k, v in preferences.items()) or "No explicit preferences"

    # -------------------------- Step 1: Plan via CoT -------------------------
    cot_prompt = f"""
    You are a highly capable orchestration agent that can choose the right tools (via capabilities) to help a user
    understand or summarize a scientific paper.
    
    ## User Input
    "{user_input}"
    
    ## Paper Preview
    {preview}
    
    ## User Belief Profile
    ### Archetype Probabilities
    {formatted_archetypes}
    
    ### Known Preferences
    {formatted_prefs}
    
    ## Instructions
    - Think step-by-step and propose a short plan (1–4 steps).
    - Express steps using **capabilities** (NOT specific tools). The router will choose the concrete tool.
    - Be adaptive: if the user is math-oriented, use 'equations' early; if summary-seeker, go straight to 'summary.generate'.
    - Use the available capability names exactly.
    
    {_format_capability_menu(tools)}
    
    ## Output Format
    Write each step on its own line as:
    Use 'capability_token' on <what to operate on>
    
    Examples:
    Use 'pdf.extract' on file path
    Use 'summary.generate' on user query + extracted text
    """.strip()

    plan = orchestrator_llm.invoke(cot_prompt)
    session.add_message("orchestrator", str(plan))

    # ----------------------- Step 2: Execute the plan ------------------------
    intermediate_outputs: List[Dict[str, Any]] = []

    for step in parse_plan(plan):
        token = step.get("token")
        if not token:
            continue

        input_data = build_tool_input(step, task, session)
        capability = token  # treat token as capability by default

        # Resolve: try as concrete tool name first (back-compat), else as capability
        prof = None
        if reg:
            prof = reg.get_tool(token)  # exact tool by name
            if prof is None:
                ctx = _build_tool_ctx_from_session(session)
                prof = reg.resolve(capability, ctx)

        if not prof:
            logger.warning(f"[Router]: No tool found for '{token}' (capability or name). Skipping.")
            continue

        logger.info(f"[Tool]: Executing {prof.tool.name} for token '{token}' with inputs: {input_data}")
        try:
            output = await _ainvoke_structured_tool(prof.tool, input_data)
            intermediate_outputs.append({"tool": prof.tool.name, "capability": token, "output": output})
        except Exception as e:
            logger.exception(f"[Tool]: {prof.tool.name} failed: {e}")
            intermediate_outputs.append({"tool": prof.tool.name, "capability": token, "output": f"[ERROR] {e}"})

    # ----------------------- Step 3: Final summarization ---------------------
    summary_prompt = build_summary_prompt(task, intermediate_outputs, session.belief_state, user_input)
    response = summarizer_llm.invoke(summary_prompt)

    # ----------------------- Step 4: Belief update ---------------------------
    full_context = "\n".join([f"{m['role']}: {m['content']}" for m in session.conversation_history])
    new_insight = update_beliefs(full_context)
    session.update_beliefs(new_insight)
    session.add_message("summarizer", response)

    return response
