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


def build_tool_input(step: Dict, task: UserTask, session: ConversationContext) -> Dict:
    target = step.get("input_target", "").lower()

    # Always try to include pdf_path when available
    base = {"pdf_path": task.file_path} if getattr(task, "file_path", None) else {}

    if "raw text" in target or "extracted text" in target:
        if getattr(task, "cache", None) and "extracted_struct" in task.cache:
            return {**base, "structured": task.cache["extracted_struct"]}
        return {**base, "text": task.raw_text}
    if "user query" in target:
        return {**base, "query": task.user_query}
    if "beliefs" in target:
        return {**base, "beliefs": session.belief_state}
    if "file path" in target or "pdf" in target or target == "":
        return {**base}

    # fall-through: supply a broad context
    return {
        **base,
        "text": task.raw_text,
        "query": task.user_query,
        "beliefs": session.belief_state,
    }


def build_summary_prompt(
        task: UserTask,
        intermediate_outputs: List[Dict[str, Any]],
        belief_state: Dict[str, Any],
        user_input: str
    ) -> str:
    # --- belief formatting ---
    belief = (belief_state or {}).get("belief", {})
    archetypes = belief.get("archetype_probs", {}) or {}
    preferences = belief.get("preferences", {}) or {}

    formatted_archetypes = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v:.2f}" for k, v in archetypes.items()
    ) or "Unknown"
    formatted_preferences = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}" for k, v in preferences.items()
    ) or "Not specified"

    # Pull a compact paper digest if available
    paper_digest = ""
    if getattr(task, "cache", None) and "extracted_text" in task.cache:
        # make/keep a cached digest to avoid re-summarizing every round
        if "paper_digest" not in task.cache:
            raw = task.cache["extracted_text"]
            # super-cheap digest: first 1500 chars (or you can LLM-summarize once)
            task.cache["paper_digest"] = raw[:1500] + (" …[truncated]" if len(raw) > 1500 else "")
        paper_digest = task.cache["paper_digest"]
    paper_digest_block = f"\n## Paper Digest (short)\n{paper_digest}\n" if paper_digest else ""

    # --- helpers ---
    def _to_static_url(p: str) -> str:
        """
        Convert a filesystem path to a URL that FastAPI can serve.
        If you've mounted StaticFiles(directory='data') at '/static',
        'data/latex_equations/foo.png' -> '/static/latex_equations/foo.png'.
        Otherwise, fall back to the raw path.
        """
        s = str(p).replace("\\", "/")
        return "/static/" + s.split("data/", 1)[-1] if "data/" in s else s

    # --- collect tool traces & outputs ---
    intermediate_summary = ""
    tool_trace = ""
    eq_imgs: List[str] = []

    for item in intermediate_outputs:
        cap = item.get("capability", "unknown.capability")
        tool_name = item.get("tool", "unknown_tool")
        out = item.get("output")

        # trace line
        tool_trace += f"- {cap} → {tool_name}\n"

        # pretty-print output (stringify; keep lightweight)
        out_str = str(out).strip()
        intermediate_summary += f"\n### Output from `{tool_name}` ({cap})\n{out_str}\n"

        # collect equation images if present
        if (cap == "equations") or (tool_name == "equation_renderer"):
            if isinstance(out, dict):
                imgs = out.get("images") or []
                if isinstance(imgs, list):
                    eq_imgs.extend(str(p) for p in imgs)

    # --- build a ready-to-paste markdown block for equation previews ---
    eq_section = ""
    if eq_imgs:
        # show up to 8 previews to keep the output tidy
        previews = "\n".join(f"![Equation]({_to_static_url(p)})" for p in eq_imgs[:8])
        eq_section = f"\n## Rendered Equation Previews\n{previews}\n"

    # --- final prompt ---
    return f"""
    You are an academic research assistant helping summarize scientific papers.
    
    ## User Query
    "{user_input}"
    
    ## User Belief Profile
    
    ### Archetype Probabilities
    {formatted_archetypes}
    
    ### User Preferences
    {formatted_preferences}
    
    {paper_digest_block}
    
    ## Tools Chosen by Router
    {tool_trace if tool_trace else "- (none)"}
    
    ## Tool Outputs for Reference
    {intermediate_summary}
    
    ## Your Task
    Write a clean, markdown-formatted summary of the paper. Include:
    - Appropriate section headers (e.g., ## Abstract, ## Methodology)
    - Highlights aligned with the user’s archetype
    - Clear structure, precise explanations, and user-adaptive tone
    
    ### If Present — Paste This Markdown Block At The End (verbatim)
    {eq_section if eq_section else "(no previews provided)"}
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


def _tool_expected_fields(tool) -> set[str]:
    schema = getattr(tool, "args_schema", None)
    if not schema:
        return set()
    fields = getattr(schema, "model_fields", None) or getattr(schema, "__fields__", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    ann = getattr(schema, "__annotations__", None)
    if isinstance(ann, dict):
        return set(ann.keys())
    return set()


def _coerce_payload_for_tool(tool, payload: dict, task: UserTask) -> dict:
    expected = _tool_expected_fields(tool)
    data = dict(payload)

    # If tool expects pdf_path and we have it on the task, inject it.
    if "pdf_path" in expected and "pdf_path" not in data and getattr(task, "file_path", None):
        data["pdf_path"] = task.file_path

    # If tool does NOT expect 'text', drop it (helps Nougat which wants only pdf_path)
    if expected and "text" in data and "text" not in expected:
        data.pop("text", None)

    # Prune to expected keys to avoid extra-field validation issues
    if expected:
        data = {k: v for k, v in data.items() if k in expected}

    return data


def _format_capability_menu(tools: Dict[str, Any]) -> str:
    capability_hints = [
        "- abstract.summarize: Summarize a paper quickly using the abstract (with fallback).",
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


def _clip(s: str, n: int = 1200) -> str:
    s = str(s or "")
    return s if len(s) <= n else (s[:n] + " …[truncated]")


def _is_extractor(tool_name: str) -> bool:
    return tool_name in {"local_pdf_extractor", "advanced_pdf_extractor_1", "advanced_pdf_extractor_2"}


def _normalize_extraction_output(output: Any, task: "UserTask") -> None:
    """
    Ensures:
      - task.raw_text: str (prompt-safe)
      - task.cache["extracted_struct"]: dict (if provided)
      - task.cache["extracted_text"]: str (longform)
    """
    text = ""

    if isinstance(output, dict):
        # Preserve structured form
        task.cache = getattr(task, "cache", {})
        task.cache["extracted_struct"] = output

        # Build readable digest from string-like fields
        parts = []
        for k, v in output.items():
            if isinstance(v, str) and v:
                s = v if len(v) <= 500 else (v[:500] + "…")
                parts.append(f"{str(k).upper()}: {s}")
        # If no string fields were found, fall back to a compact str() of the dict
        text = "\n".join(parts) if parts else str(output)

    else:
        # Coerce None/other types into text
        text = "" if output is None else str(output)

    task.raw_text = text
    task.cache = getattr(task, "cache", {})
    task.cache["extracted_text"] = text


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
        insight = update_beliefs(user_query, llms["belief_updater"])
        session.update_beliefs(insight)

    # Route the initial PDF extraction (instead of hard-coding local extractor)
    reg = get_registry()
    if not reg:
        # Safety: if registry is not initialized, fall back to a minimal local path
        from app.tools.local.pdf_extractor import extract_pdf_text
        output = extract_pdf_text(file_path)
        _normalize_extraction_output(output, task)
    else:
        # Optional: add doc-aware hints using a filename sniff (cheap) before routing
        _enhance_doc_prefs_with_preview(session, file.filename)

        ctx = _build_tool_ctx_from_session(session)
        prof = reg.resolve("pdf.extract", ctx)
        if not prof:
            raise RuntimeError("No tool supports capability 'pdf.extract'")

        logger.info(f"[Router]: Selected '{prof.tool.name}' for pdf.extract (requires_network={prof.requires_network})")
        output = await _ainvoke_structured_tool(prof.tool, {"pdf_path": file_path})
        _normalize_extraction_output(output, task)

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
    session.compact_history(summarizer_llm, max_chars=12000, keep_last=2, summary_target_chars=1200)

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
    
    ## Output Format (must follow)
    Write each step on its own line as:
    Use '<capability_token>' on <what to operate on>
    
    Allowed capability tokens ONLY:
    - pdf.summarize
    - pdf.extract
    - equations
    - layout
    - summary.generate
    
    Examples:
    Use 'pdf.summarize' on paper
    Use 'pdf.extract' on pdf
    Use 'equations' on extracted text
    Use 'summary.generate' on user query + tool outputs
    """.strip()

    plan_obj = orchestrator_llm.invoke(cot_prompt)
    plan_text = getattr(plan_obj, "content", plan_obj)  # use .content if available
    plan_text = str(plan_text)
    session.add_message("orchestrator", plan_text)

    # ----------------------- Step 2: Execute the plan ------------------------
    intermediate_outputs: List[Dict[str, Any]] = []

    for step in parse_plan(plan_text):
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

        # coerce payload for the specific tool’s schema
        input_data = _coerce_payload_for_tool(prof.tool, input_data, task)
        logger.info(f"[Tool]: Executing {prof.tool.name} for token '{token}' with inputs: {input_data}")
        try:
            output = await _ainvoke_structured_tool(prof.tool, input_data)
            digest_output = output
            if _is_extractor(prof.tool.name):
                # store full text off-prompt (no token cost)
                try:
                    # keep the full thing for retrieval/summarization later
                    if not hasattr(task, "cache"):
                        task.cache = {}
                    task.cache["extracted_text"] = str(output)
                except (Exception, ):
                    pass
                # only include a clipped preview in the prompt
                digest_output = _clip(str(output), 1200)

            elif prof.tool.name == "equation_renderer":
                # keep only count + first few paths in prompt
                if isinstance(output, dict):
                    imgs = output.get("images", []) or []
                    digest_output = {
                        "count": output.get("count", len(imgs)),
                        "images": imgs[:8],  # previews; full list is unnecessary
                    }

            intermediate_outputs.append({
                "tool": prof.tool.name,
                "capability": token,
                "output": digest_output
            })
        except Exception as e:
            logger.exception(f"[Tool]: {prof.tool.name} failed: {e}")
            intermediate_outputs.append({"tool": prof.tool.name, "capability": token, "output": f"[ERROR] {e}"})

    # ----------------------- Step 3: Final summarization ---------------------
    summary_prompt = build_summary_prompt(task, intermediate_outputs, session.belief_state, user_input)
    resp_obj = summarizer_llm.invoke(summary_prompt)
    response = getattr(resp_obj, "content", resp_obj)
    response = str(response)

    # ----------------------- Step 4: Belief update ---------------------------
    full_context = "\n".join([f"{m['role']}: {m['content']}" for m in session.conversation_history])
    new_insight = update_beliefs(full_context, llms["belief_updater"])
    session.update_beliefs(new_insight)
    session.add_message("summarizer", response)

    return response
