from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import json, re
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextAnalysisInput(BaseModel):
    text: str = Field(..., description="Paper text content to analyze.")
    query: str = Field(..., description="Specific question to answer about the text.")
    context: str = Field(default="", description="Optional additional context to guide analysis.")


# ----------------- helpers -----------------

EQ_INLINE = re.compile(r"\\\((.+?)\\\)")
EQ_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
EQ_DOLLAR_DISP = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
EQ_DOLLAR_INL = re.compile(r"(?<!\$)\$(.+?)\$(?!\$)", re.DOTALL)

def _looks_like_equation_intent(q: str) -> bool:
    q = (q or "").lower()
    return any(
        phrase in q
        for phrase in [
            "most important equation",
            "main equation",
            "key equation",
            "single equation",
            "one equation",
            "cool equation",
            "highlight equation",
        ]
    )

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def _first_equation_from_text(text: str) -> Optional[str]:
    """Fallback: try to pull one LaTeX equation from the text/excerpt."""
    if not text:
        return None
    for rx in (EQ_DISPLAY, EQ_DOLLAR_DISP, EQ_INLINE, EQ_DOLLAR_INL):
        m = rx.search(text)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None


# ----------------- tool entrypoints -----------------

def analyze_text_adapter(text: str, query: str, context: str = "") -> Dict[str, Any]:
    """Adapter for LangChain StructuredTool: accepts schema fields directly."""
    from app.main import llms  # reuse your configured LLMs
    inp = TextAnalysisInput(text=text, query=query, context=context)
    return analyze_text_content(inp, llms["summarizer"])


def analyze_text_content(input_data: TextAnalysisInput, llm) -> Dict[str, Any]:
    """
    Analyze text content to answer specific questions.
    If the question is about a single/most-important/cool equation, attempt to return exactly one LaTeX snippet.
    """
    want_single_eq = _looks_like_equation_intent(input_data.query)

    if want_single_eq:
        # JSON-only contract so the orchestrator can auto-chain to the renderer.
        analysis_prompt = f"""
You are an expert academic assistant. From the provided paper excerpt, identify ONE equation that best matches the user's intent
(e.g., most important / main / key / cool). Then explain briefly why.

Respond ONLY as JSON (no extra text, no code fences):

{{
  "answer": "Brief natural language explanation (≤ 120 words).",
  "latex_string": "LaTeX of the single chosen equation (omit surrounding $$ or \\[\\]).",
  "confidence": 0.0
}}

User Question:
{input_data.query}

Additional Context:
{input_data.context}

Paper Content (excerpt):
{input_data.text}
""".strip()
    else:
        # Generic analysis (plain text answer; keep your previous style)
        analysis_prompt = f"""
You are an expert academic assistant. Analyze the paper text to answer the user's question.

User Question:
{input_data.query}

Additional Context:
{input_data.context}

Paper Content (excerpt):
{input_data.text}

Instructions:
- Answer directly and specifically (≤ 200 words).
- If discussing equations, cite them in LaTeX if present.
""".strip()

    try:
        resp = llm.invoke(analysis_prompt)
        raw = getattr(resp, "content", str(resp))

        if want_single_eq:
            # Parse JSON; fall back to regex if needed
            data = raw if isinstance(raw, dict) else _safe_json_loads(_strip_code_fences(str(raw))) or {}
            answer = (data.get("answer") or "").strip()
            latex = (data.get("latex_string") or "").strip()

            if not latex:
                # Fallback: try to grab one from the text excerpt
                latex = _first_equation_from_text(input_data.text) or ""

            return {
                "analysis": answer if answer else "Identified a key equation.",
                "latex_string": latex,
                "confidence": float(data.get("confidence", 0.0)) if isinstance(data.get("confidence"), (int, float)) else 0.0,
                "query": input_data.query,
                "success": True,
            }

        # Non-equation queries: just return analysis text
        return {"analysis": str(raw).strip(), "query": input_data.query, "success": True}

    except Exception as e:
        logger.exception(f"[text_analyser] Analysis failed: {e}")
        return {"analysis": f"Analysis failed: {str(e)}", "query": input_data.query, "success": False}
