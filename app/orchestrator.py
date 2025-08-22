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
        # 0) Helper: pull first non-empty LaTeX from a list
        def _first_nonempty(xs):
            for x in xs or []:
                s = (x or "").strip()
                if s:
                    return s
            return ""

        # 1) Section-targeted extraction (unchanged but a bit safer)
        if "section" in target and getattr(task, "cache", None) and "extracted_struct" in task.cache:
            section_id = None
            if m := re.search(r"(?:section|sec)\.?\s+(\d+)", target, re.IGNORECASE):
                section_id = m.group(1)

            sections = (task.cache["extracted_struct"] or {}).get("sections", [])
            if section_id and isinstance(sections, list):
                target_section = next((s for s in sections if str(s.get("id")) == str(section_id)), None)
                if isinstance(target_section, dict):
                    # Prefer explicit LaTeX list if present
                    if "equations" in target_section and isinstance(target_section["equations"], list):
                        # Support both [{"latex": "..."}] and ["..."]
                        eq_objs = target_section["equations"]
                        latexs = [e.get("latex") if isinstance(e, dict) else str(e) for e in eq_objs]
                        latex = _first_nonempty(latexs)
                        if latex:
                            return {**base, "latex_string": latex}

                    # Fallback: regex from section text/content
                    sec_text = target_section.get("text") or target_section.get("content") or ""
                    # Support $, $$, \( \), \[ \]
                    m_inline = re.search(r"\$(.+?)\$", sec_text, re.DOTALL)
                    m_disp = re.search(r"\$\$(.+?)\$\$", sec_text, re.DOTALL)
                    m_paren = re.search(r"\\\((.+?)\\\)", sec_text, re.DOTALL)  # \( ... \)
                    m_brack = re.search(r"\\\[(.+?)\\\]", sec_text, re.DOTALL)  # \[ ... \]
                    latex = ""
                    for m_ in (m_disp, m_inline, m_brack, m_paren):
                        if m_ and m_.group(1).strip():
                            latex = m_.group(1).strip()
                            break
                    if latex:
                        return {**base, "latex_string": latex}

        # 2) Pull from full extracted_struct (anywhere), if available
        if getattr(task, "cache", None) and "extracted_struct" in task.cache:
            struct = task.cache["extracted_struct"] or {}
            # (a) global equations list
            if isinstance(struct.get("equations"), list):
                eq_objs = struct["equations"]
                latexs = [e.get("latex") if isinstance(e, dict) else str(e) for e in eq_objs]
                latex = _first_nonempty(latexs)
                if latex:
                    return {**base, "latex_string": latex}
            # (b) scan section texts for math
            if isinstance(struct.get("sections"), list):
                for s in struct["sections"]:
                    sec_text = (s.get("text") or s.get("content") or "") if isinstance(s, dict) else ""
                    if not sec_text:
                        continue
                    m_disp = re.search(r"\$\$(.+?)\$\$", sec_text, re.DOTALL)
                    m_inline = re.search(r"\$(.+?)\$", sec_text, re.DOTALL)
                    m_paren = re.search(r"\\\((.+?)\\\)", sec_text, re.DOTALL)
                    m_brack = re.search(r"\\\[(.+?)\\\]", sec_text, re.DOTALL)
                    for m_ in (m_disp, m_inline, m_brack, m_paren):
                        if m_ and m_.group(1).strip():
                            return {**base, "latex_string": m_.group(1).strip()}

        # 3) Fallback: scan cached plain extracted text (Nougat/Docling markdown) if present
        if getattr(task, "cache", None) and "extracted_text" in task.cache:
            t = task.cache["extracted_text"] or ""
            # Support $, $$, \( \), \[ \]  (Nougat often escapes as \\( \\))
            m_disp = re.search(r"\$\$(.+?)\$\$", t, re.DOTALL)
            m_inline = re.search(r"(?<!\$)\$(.+?)\$(?!\$)", t, re.DOTALL)
            m_paren = re.search(r"\\\((.+?)\\\)", t, re.DOTALL)
            m_brack = re.search(r"\\\[(.+?)\\\]", t, re.DOTALL)
            for m_ in (m_disp, m_inline, m_brack, m_paren):
                if m_ and m_.group(1).strip():
                    return {**base, "latex_string": m_.group(1).strip()}

        # 4) Last resort: scan raw_text
        if getattr(task, "raw_text", ""):
            t = task.raw_text
            m_disp = re.search(r"\$\$(.+?)\$\$", t, re.DOTALL)
            m_inline = re.search(r"(?<!\$)\$(.+?)\$(?!\$)", t, re.DOTALL)
            m_paren = re.search(r"\\\((.+?)\\\)", t, re.DOTALL)
            m_brack = re.search(r"\\\[(.+?)\\\]", t, re.DOTALL)
            for m_ in (m_disp, m_inline, m_brack, m_paren):
                if m_ and m_.group(1).strip():
                    return {**base, "latex_string": m_.group(1).strip()}

    # ----------------- NEW: targeted text analysis (flexible narrowing) -----------------
    if token in {"text.analyse", "text_analyser"}:
        # Prefer richest text available
        full_text = ""
        if getattr(task, "cache", None) and "extracted_text" in task.cache:
            full_text = task.cache["extracted_text"] or ""
        if not full_text:
            full_text = getattr(task, "raw_text", "") or ""

        narrowed = full_text  # default
        target_norm = (target or "").lower().strip()

        # numeric like "section 3", "sec. 3"
        section_hint_num = None
        m_num = re.search(r"(?:section|sec)\.?\s+(\d+)", target_norm)
        if m_num:
            section_hint_num = m_num.group(1)

        # semantic like "abstract", "related work", etc.
        key_map = {
            "abstract": "abstract",
            "introduction": "introduction",
            "related work": "related_work",
            "background": "background",
            "method": "method",
            "methods": "method",
            "methodology": "method",
            "approach": "method",
            "experiment": "experiments",
            "experiments": "experiments",
            "evaluation": "experiments",
            "results": "experiments",
            "discussion": "discussion",
            "analysis": "discussion",
            "conclusion": "conclusion",
            "summary": "conclusion",
            "references": "references",
            "bibliography": "references",
            "appendix": "appendix",
        }
        section_hint_key = None
        for human, norm in key_map.items():
            if human in target_norm:
                section_hint_key = norm
                break

        # Try to narrow using structured output if present
        if getattr(task, "cache", None) and "extracted_struct" in task.cache:
            struct = task.cache["extracted_struct"]
            if isinstance(struct, dict):
                # Case A: remote extractors with list of sections and id/header/text
                if isinstance(struct.get("sections"), list):
                    if section_hint_num is not None:
                        sec = next(
                            (s for s in struct["sections"] if str(s.get("id")) == str(section_hint_num)),
                            None
                        )
                        if sec:
                            narrowed = (sec.get("text") or sec.get("content") or "").strip() or narrowed
                    elif section_hint_key is not None:
                        def _norm_hdr(h):
                            h = (h or "").lower()
                            for human, norm in key_map.items():
                                if human in h:
                                    return norm
                            return h

                        cand = next(
                            (s for s in struct["sections"]
                             if _norm_hdr(s.get("header") or s.get("title")) == section_hint_key),
                            None
                        )
                        if cand:
                            narrowed = (cand.get("text") or cand.get("content") or "").strip() or narrowed

                # Case B: local extractor with normalized top-level keys
                elif section_hint_key and isinstance(struct.get(section_hint_key), str):
                    sec_txt = struct.get(section_hint_key, "").strip()
                    if sec_txt:
                        narrowed = sec_txt

        # Build a concise context string from beliefs/preferences
        belief = (session.belief_state or {}).get("belief", {}) or {}
        prefs = (session.belief_state or {}).get("routing_ctx", {}).get("preferences", {}) or {}
        ctx_lines = []
        if isinstance(belief.get("preferences"), dict):
            for pk, pv in belief["preferences"].items():
                ctx_lines.append(f"{pk}: {pv}")
        for pk, pv in prefs.items():
            ctx_lines.append(f"{pk}: {pv}")
        context_str = "\n".join(ctx_lines)

        return {
            **base,
            "text": narrowed or full_text,
            "query": task.user_query,
            "context": context_str,
        }

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
        user_input: str,
        max_words: int=250
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
    if getattr(task, "cache", None) and "extracted_text" in task.cache:
        if task.raw_text:
            text = task.raw_text
            paper_digest = text[:4000] + (" …[truncated]" if len(text) > 4000 else "")
            paper_digest_block = f"\n## Paper Digest\n{paper_digest}\n"
        else:
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
    Write a clear, flowing summary of this academic paper in natural English prose. 
    
    Focus on:
    - What problem the paper addresses and why it matters
    - The main approach or methodology used
    - Key findings and contributions
    - Significance and implications
    
    Write as a coherent narrative without section headers or bullet points. 
    Aim for {max_words} words maximum.

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

    # --- NEW: auto-fill for text_analyser-like tools ---
    if expected:
        if "text" in expected and not data.get("text"):
            text = ""
            if getattr(task, "cache", None) and "extracted_text" in task.cache:
                text = task.cache["extracted_text"] or ""
            if not text:
                text = getattr(task, "raw_text", "") or ""
            if text:
                data["text"] = text

        if "query" in expected and not data.get("query"):
            data["query"] = getattr(task, "user_query", "") or ""

        if "context" in expected and not data.get("context"):
            data["context"] = ""

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
        "- text.analyse: Answer targeted questions about specific parts of the text (e.g., most important equation, gist of section N, clarity of derivations, coverage of related work).",
    ]
    concrete_guidance = (
        "\n### When to call concrete tools directly\n"
        "- Use 'latex_console_renderer' if the user wants console/CLI/plain-text/Unicode math.\n"
        "- Use 'equation_renderer' if the user wants rendered images/figures/previews.\n"
        "- Use 'text_analyser' if you need a targeted analysis on a specific subset of text.\n"
        "Otherwise, use capabilities and let the router decide."
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


def build_orchestration_prompt(
    user_input: str,
    preview: str,
    tools_desc: str,
    formatted_archetypes: str,
    formatted_prefs: str,
    routing_tags: list[str] | None,
    latency_budget_ms: int | None = None,
    allow_remote: bool = True,
    privacy_level: str = "standard",
) -> str:
    routing_tags = routing_tags or []
    budget_line = (
        f"- Latency budget: ≤ {latency_budget_ms} ms.\n" if latency_budget_ms is not None else ""
    )
    remote_line = (
        "- Remote/network tools DISALLOWED (privacy or user setting).\n"
        if (not allow_remote or privacy_level == "strict")
        else "- Remote/network tools ALLOWED.\n"
    )

    return f"""
    You are the MASTER ORCHESTRATOR for an academic paper assistant. Plan 1–4 actions (capabilities or specific tools) that best satisfy the user, **grounded in the belief profile** (archetype priors + preferences) and system constraints.
    
    ## Inputs
    ### User Input
    "{user_input}"
    
    ### Paper Preview (truncated)
    {preview}
    
    ### Belief Profile (Priors)
    - Archetypes (P(archetype)): 
    {formatted_archetypes}
    
    - Stated/learned preferences:
    {formatted_prefs}
    
    - Routing tags: {", ".join(routing_tags) if routing_tags else "(none)"}
    
    ### System Constraints
    {budget_line}{remote_line}
    
    ## Capabilities (router will choose the concrete tool unless a concrete tool is explicitly allowed)
    - pdf.summarize
    - pdf.extract
    - equations
    - layout
    - summary.generate
    - text.analyse
    
    ### Concrete tools you MAY name directly (only if clearly needed):
    - latex_console_renderer (console/Unicode equations)
    - equation_renderer (image previews of equations)
    - text_analyser (targeted analysis)
    - pdf_summarizer (full-text summarization)
    - abstract_summary (abstract-first summarization)
    - local_pdf_extractor / advanced_pdf_extractor_1 / advanced_pdf_extractor_2 (only if you **must** force a specific extractor)
    
    ## Decision Policy (bind decisions to beliefs)
    1) **Math-oriented (high prior ≥ 0.35 or top-1):**
       - Early 'equations'. If user wants text/CLI → 'latex_console_renderer'; if images/previews → 'equation_renderer'.
       - Then 'text.analyse' for key derivations, optionally 'summary.generate'.
    2) **Summary-seeker (≥ 0.35 or top-1):**
       - Prefer quick overview: 'abstract_summary' **or** 'pdf.summarize' (choose one).
       - Then 'summary.generate' to tailor to preferences.
    3) **Deep-dive analyst (≥ 0.35 or top-1):**
       - 'pdf.extract' → 'text.analyse' on specific sections (methods/experiments).
       - Consider 'equations' if document likely has math (beliefs or preview suggest it).
    4) **Explorer (default/uncertain):**
       - Start broad: 'pdf.summarize' or 'abstract_summary', then refine with 'text.analyse' if needed.
    
    ## Preference Overrides (if present, they trump archetype defaults)
    - preferred_focus=math → include 'equations' early.
    - preferred_focus=methods → include 'text.analyse' on method/approach section.
    - depth_preference=high-level → skip deep tools; go straight to summary path.
    - tone=concise → keep plan minimal (1–2 steps).
    - routing tags:
      - math_heavy → include 'equations'
      - detail_oriented → include 'text.analyse'
      - prefers_summary → prioritize 'abstract_summary' or 'pdf.summarize'
    - doc_has_equations=True → include 'equations'
    - has_complex_layout=True → lean on 'pdf.extract' with better layout handling.
    
    ## Cost/Latency/Privacy Rules
    - If remote tools disallowed or privacy=strict → avoid tools requiring network.
    - If latency budget is tight → prefer fewer steps and low-latency capabilities.
    - If both a capability and a concrete tool fit, **prefer the capability** (router picks the tool).
    - Avoid redundant steps (don’t call both 'abstract_summary' and 'pdf.summarize' unless justified).
    
    ## Output Format (STRICT)
    Output ONLY the plan lines (no explanations, no numbering, no extra text).  
    Each step on its own line:
    Use '<capability_or_tool>' on <target>
    
    Allowed capability tokens ONLY:
    - pdf.summarize
    - pdf.extract
    - equations
    - layout
    - summary.generate
    - text.analyse
    
    Examples:
    Use 'pdf.summarize' on paper
    Use 'pdf.extract' on pdf
    Use 'equations' on extracted text
    Use 'latex_console_renderer' on the most important equation
    Use 'equation_renderer' on equations in section 2
    Use 'text.analyse' on related work section (coverage/comprehensiveness)
    Use 'summary.generate' on user query + tool outputs
    
    Remember: **Use beliefs to justify your choices internally, but output ONLY the plan lines.**
    """.strip()


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
        insight = update_beliefs(user_query, session.belief_state, llms["belief_updater"])
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

    # -------------------------- Step 1: Plan via CoT -------------------------
    belief = (session.belief_state or {}).get("belief", {})
    archetypes = belief.get("archetype_probs", {}) or {}
    preferences = belief.get("preferences", {}) or {}
    formatted_archetypes = "\n".join(f"- {k}: {v:.2f}" for k, v in archetypes.items()) or "Unknown"
    formatted_prefs = "\n".join(f"- {k}: {v}" for k, v in preferences.items()) or "No explicit preferences"
    routing_ctx = (session.belief_state or {}).get("routing_ctx", {}) or {}
    routing_tags = routing_ctx.get("archetypes", []) or []
    latency_budget = getattr(session, "latency_budget_ms", None)
    allow_remote = getattr(session, "allow_remote", True)
    privacy_level = getattr(session, "privacy_level", "standard")

    cot_prompt = build_orchestration_prompt(
        user_input=user_input,
        preview=preview,
        tools_desc=_format_capability_menu(tools),
        formatted_archetypes=formatted_archetypes,
        formatted_prefs=formatted_prefs,
        routing_tags=routing_tags,
        latency_budget_ms=latency_budget,
        allow_remote=allow_remote,
        privacy_level=privacy_level,
    )

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
    logger.info(f"Belief distribution: {session.belief_state}")
    full_context = "\n".join([f"{m['role']}: {m['content']}" for m in session.conversation_history])
    new_insight = update_beliefs(full_context, session.belief_state, llms["belief_updater"])
    session.update_beliefs(new_insight)
    session.add_message("summarizer", response)

    return response
