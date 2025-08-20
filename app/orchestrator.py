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
    token = (step.get("token") or "").lower()

    # Always try to include pdf_path when available
    base = {"pdf_path": task.file_path} if getattr(task, "file_path", None) else {}

    # >>> UPDATED: Treat both equation renderers similarly when extracting a single LaTeX string.
    wants_single_eq = (
        "equations" in token
        or "latex_console_renderer" in token
        or "equation_renderer" in token
    )

    # If the planner asked for an equation, try to extract one:
    if wants_single_eq:
        # 1) Section-targeted extraction (unchanged)
        if "section" in target:
            section_id = None
            if m := re.search(r"section (\d+)", target, re.IGNORECASE):
                section_id = m.group(1)

            if section_id and getattr(task, "cache", None) and "extracted_struct" in task.cache:
                sections = task.cache["extracted_struct"].get("sections", [])
                target_section = next((s for s in sections if s.get("id") == section_id), None)
                if target_section and "equations" in target_section:
                    equations = target_section["equations"]
                    latex = equations[0]["latex"] if equations else ""
                    if latex:
                        return {**base, "latex_string": latex}

        # 2) Generic fallback from raw_text (NOW runs even if no 'section' in target)
        if getattr(task, "raw_text", ""):
            equations = re.findall(r"\$\$([^$]+)\$\$|\$([^$]+)\$", task.raw_text)
            if equations:
                latex = equations[0][0] or equations[0][1]
                if latex:
                    return {**base, "latex_string": latex}
        # If we couldn't extract, we DON'T return yet — let other branches supply broader context.

    if "raw text" in target or "extracted text" in target:
        if getattr(task, "cache", None) and "extracted_struct" in task.cache:
            return {**base, "structured": task.cache["extracted_struct"]}
        return {**base, "text": task.raw_text}

    if "user query" in target:
        # Ensure summary.generate gets real context, not just the query
        if token == "summary.generate":
            return {**base, "text": task.raw_text, "query": task.user_query, "beliefs": session.belief_state}
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
        if "paper_digest" not in task.cache:
            raw = task.cache["extracted_text"]
            task.cache["paper_digest"] = raw[:1500] + (" …[truncated]" if len(raw) > 1500 else "")
        paper_digest = task.cache["paper_digest"]
    paper_digest_block = f"\n## Paper Digest (short)\n{paper_digest}\n" if paper_digest else ""

    def _to_static_url(p: str) -> str:
        s = str(p).replace("\\", "/")
        return "/static/" + s.split("data/", 1)[-1] if "data/" in s else s

    intermediate_summary = ""
    tool_trace = ""
    eq_imgs: List[str] = []

    for item in intermediate_outputs:
        cap = item.get("capability", "unknown.capability")
        tool_name = item.get("tool", "unknown_tool")
        out = item.get("output")

        tool_trace += f"- {cap} → {tool_name}\n"

        out_str = str(out).strip()
        intermediate_summary += f"\n### Output from `{tool_name}` ({cap})\n{out_str}\n"

        # collect equation images if present (image renderer)
        if (cap == "equations") or (tool_name == "equation_renderer"):
            if isinstance(out, dict):
                imgs = out.get("images") or []
                if isinstance(imgs, list):
                    eq_imgs.extend(str(p) for p in imgs)

    eq_section = ""
    if eq_imgs:
        previews = "\n".join(f"![Equation]({_to_static_url(p)})" for p in eq_imgs[:8])
        eq_section = f"\n## Rendered Equation Previews\n{previews}\n"

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


# >>> NEW: light heuristic to nudge console vs image rendering based on the *user query*.
def _hint_equation_render_mode(user_input: str, session: ConversationContext) -> None:
    text = (user_input or "").lower()
    want_console = any(w in text for w in ["console", "cli", "unicode", "plain text", "text-only", "terminal"])
    want_images = any(w in text for w in ["image", "png", "figure", "render", "preview", "visual"])
    belief_state = session.belief_state or {}
    rc = belief_state.get("routing_ctx", {}) or {}
    prefs = dict(rc.get("preferences", {}) or {})
    if want_console:
        prefs["prefer_console_equations"] = True
    if want_images:
        prefs["prefer_image_equations"] = True
    rc["preferences"] = prefs
    belief_state["routing_ctx"] = rc
    session.belief_state = belief_state


def _enhance_doc_prefs_with_preview(session: ConversationContext, preview_text: str) -> None:
    belief_state = session.belief_state or {}
    rc = belief_state.get("routing_ctx", {}) or {}
    prefs = dict(rc.get("preferences", {}) or {})
    text = (preview_text or "").lower()

    eq_hint = any(w in text for w in ("equation", "latex", "proof", "derivation"))
    layout_hint = any(w in text for w in ("table", "figure", "diagram", "two-column", "layout"))

    prefs["doc_has_equations"] = prefs.get("doc_has_equations") or eq_hint
    prefs["has_complex_layout"] = prefs.get("has_complex_layout") or layout_hint

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

    if "pdf_path" in expected and "pdf_path" not in data and getattr(task, "file_path", None):
        data["pdf_path"] = task.file_path

    if expected and "text" in data and "text" not in expected:
        data.pop("text", None)

    if expected:
        data = {k: v for k, v in data.items() if k in expected}

    return data


def _format_capability_menu(tools: Dict[str, Any]) -> str:
    capability_hints = [
        "- pdf.summarize: Handles full text summary of a paper or sections.",
        "- abstract.summarize: Summarize a paper quickly using the abstract (with fallback).",
        "- pdf.extract: Extracts structured text/sections from PDFs",
        "- equations: Extracts/handles LaTeX; can render as PNG (images) or as Unicode for console",
        "- layout: Emphasizes structural fidelity in complex layouts",
        "- summary.generate: Produce an adaptive summary from gathered context",
    ]
    # >>> NEW: teach the planner when it’s OK to name a specific tool
    concrete_guidance = (
        "\n### When to call concrete tools directly\n"
        "- Use 'latex_console_renderer' if the user wants console/CLI/plain-text/Unicode math.\n"
        "- Use 'equation_renderer' if the user wants rendered images/figures/previews.\n"
        "Otherwise, use 'equations' as a capability and let the router decide."
    )
    return (
            "## Available Capabilities (router will choose the tool):\n"
            + "\n".join(capability_hints)
            + "\n\n## Concrete Tools (for your reference only — prefer capabilities):\n"
            + format_tool_descriptions(tools)
            + concrete_guidance
    )


# Robustly invoke a LangChain StructuredTool across versions
async def _ainvoke_structured_tool(tool, kwargs: Dict[str, Any]):
    if hasattr(tool, "ainvoke"):
        return await tool.ainvoke(kwargs)
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
    text = ""

    if isinstance(output, dict):
        task.cache = getattr(task, "cache", {})
        task.cache["extracted_struct"] = output

        parts = []
        for k, v in output.items():
            if isinstance(v, str) and v:
                s = v if len(v) <= 500 else (v[:500] + "…")
                parts.append(f"{str(k).upper()}: {s}")
        text = "\n".join(parts) if parts else str(output)

    else:
        text = "" if output is None else str(output)

    task.raw_text = text
    task.cache = getattr(task, "cache", {})
    task.cache["extracted_text"] = text


def _pick_precomputed_summary(intermediate_outputs: List[Dict[str, Any]]) -> str | None:
    """
    If any prior tool already produced a usable summary, return it to short-circuit
    the final summarizer LLM pass. Prefer the most recent one.
    """
    summary_caps = {"pdf.summarize", "abstract.summarize"}
    summary_tools = {"pdf_summarizer", "abstract_summary"}

    for item in reversed(intermediate_outputs):
        token = (item.get("capability") or "").strip().lower()
        tool = (item.get("tool") or "").strip().lower()
        out = item.get("output")

        if token in summary_caps or tool in summary_tools:
            # We expect a string summary from these tools
            if isinstance(out, str):
                clean = out.strip()
                # Heuristic: ensure it's not trivially short or an [ERROR]
                if clean and "[error]" not in clean.lower() and len(clean) >= 80:
                    return clean
    return None


# --------------------------- Orchestration flow -------------------------------

async def prepare_user_task(file, user_query: str, session: ConversationContext, llms: Dict[str, Any]):
    content = await file.read()
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(content)

    task = UserTask(
        file_path=file_path,
        raw_text="",
        user_query=user_query,
        task_type="summarize",
        metadata={"filename": file.filename}
    )

    session.set_task(task)
    session.add_message("user", user_query)

    if not session.belief_state:
        insight = update_beliefs(user_query, llms["belief_updater"])
        session.update_beliefs(insight)

    reg = get_registry()
    if not reg:
        from app.tools.local.pdf_extractor import extract_pdf_text
        output = extract_pdf_text(file_path)
        _normalize_extraction_output(output, task)
    else:
        _enhance_doc_prefs_with_preview(session, file.filename)
        ctx = _build_tool_ctx_from_session(session)
        prof = reg.resolve("pdf.extract", ctx)
        if not prof:
            raise RuntimeError("No tool supports capability 'pdf.extract'")
        logger.info(f"[Router]: Selected '{prof.tool.name}' for pdf.extract (requires_network={prof.requires_network})")
        output = await _ainvoke_structured_tool(prof.tool, {"pdf_path": file_path})
        _normalize_extraction_output(output, task)

    response = await conversation_loop(user_query, task, session, llms)
    return response


async def conversation_loop(user_input: str, task: UserTask, session: ConversationContext, llms: Dict[str, Any]):
    tools = get_tools()
    reg = get_registry()
    orchestrator_llm = llms["orchestrator"]
    summarizer_llm = llms["summarizer"]
    session.compact_history(summarizer_llm, max_chars=12000, keep_last=2, summary_target_chars=1200)

    preview = (task.get_full_text() or "")[:1200]
    _enhance_doc_prefs_with_preview(session, preview)
    # >>> NEW: nudge rendering mode based on *this round’s* user input
    _hint_equation_render_mode(user_input, session)

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
    - Prefer **capabilities** (NOT specific tools). The router will choose the concrete tool.
    - If the user explicitly wants console/CLI/plain-text/Unicode math, you MAY call `latex_console_renderer` directly.
    - If the user explicitly wants images/previews/figures of equations, you MAY call `equation_renderer` directly.
    - Be adaptive: if the user is math-oriented, use 'equations' early; if summary-seeker, go straight to 'summary.generate'.
    - Use the available capability names exactly.

    {_format_capability_menu(tools)}

    ## Output Format (must follow)
    Write each step on its own line as:
    Use '<capability_token_or_tool_name>' on <what to operate on>

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
    Use 'latex_console_renderer' on the most important equation
    Use 'equation_renderer' on equations in section 2
    Use 'summary.generate' on user query + tool outputs
    """.strip()

    plan_obj = orchestrator_llm.invoke(cot_prompt)
    plan_text = getattr(plan_obj, "content", plan_obj)
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

        prof = None
        if reg:
            # Allow concrete tool names like 'latex_console_renderer'
            prof = reg.get_tool(token)
            if prof is None:
                ctx = _build_tool_ctx_from_session(session)
                prof = reg.resolve(capability, ctx)

        if not prof:
            logger.warning(f"[Router]: No tool found for '{token}' (capability or name). Skipping.")
            continue

        input_data = _coerce_payload_for_tool(prof.tool, input_data, task)
        expected_fields = _tool_expected_fields(prof.tool)
        if (
                prof.tool.name == "latex_console_renderer"
                and "latex_string" in expected_fields
                and not input_data.get("latex_string")
        ):
            logger.info("[Tool]: Skipping latex_console_renderer — no latex_string extracted; planner target was '%s'.",
                        step.get("input_target"))
            continue
        logger.info(f"[Tool]: Executing {prof.tool.name} for token '{token}' with inputs: {input_data}")
        try:
            output = await _ainvoke_structured_tool(prof.tool, input_data)
            digest_output = output
            if _is_extractor(prof.tool.name):
                try:
                    if not hasattr(task, "cache"):
                        task.cache = {}
                    task.cache["extracted_text"] = str(output)
                except (Exception, ):
                    pass
                digest_output = _clip(str(output), 1200)

            elif prof.tool.name == "equation_renderer":
                if isinstance(output, dict):
                    imgs = output.get("images", []) or []
                    digest_output = {
                        "count": output.get("count", len(imgs)),
                        "images": imgs[:8],
                    }

            # >>> NEW: clip console renderer output to keep prompt light
            elif prof.tool.name == "latex_console_renderer":
                digest_output = _clip(str(output), 600)

            intermediate_outputs.append({
                "tool": prof.tool.name,
                "capability": token,
                "output": digest_output
            })
        except Exception as e:
            logger.exception(f"[Tool]: {prof.tool.name} failed: {e}")
            intermediate_outputs.append({"tool": prof.tool.name, "capability": token, "output": f"[ERROR] {e}"})

    # ----------------------- Step 3: Final summarization ---------------------
    precomputed = _pick_precomputed_summary(intermediate_outputs)
    if precomputed:
        response = str(precomputed)
        logger.info("[Orchestrator] Short-circuit: using precomputed summary from a tool.")
    else:
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
