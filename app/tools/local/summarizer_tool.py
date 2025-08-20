# app/tools/local/summarizer_tool.py

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from app.prompts.summarizer_prompt import CHUNK_USER_TEMPLATE, ONE_SHOT_USER_TEMPLATE


class GenerateSummaryInput(BaseModel):
    """
    Flexible schema for the 'generate_summary' tool.
    You may pass a single prebuilt 'context' string, OR provide pieces and let the tool compose it.
    """
    # EITHER provide this:
    context: Optional[str] = Field(
        None, description="Pre-composed context to summarize (takes precedence if provided)."
    )

    # OR provide any of these pieces (the tool will compose a context string):
    text: Optional[str] = Field(None, description="Extracted paper text or relevant snippet.")
    query: Optional[str] = Field(None, description="User's current query / instruction.")
    beliefs: Optional[Dict[str, Any]] = Field(
        None, description="Belief state/preferences to adapt tone or focus."
    )

    # Optional knobs (safe defaults)
    target_words: int = Field(250, ge=50, le=1000, description="Approx target length for the summary.")
    style: str = Field("neutral technical", description="Style hint for summarization.")
    format_hint: str = Field(
        "short paragraphs or bullets",
        description="Format hint for the output."
    )


def summarize_abstract(text_to_be_summarized: str, summarizer_llm) -> str:
    """
    Summarize ONLY the abstract (or intro fallback) — quick high-level pass.
    Retained for compatibility with any existing callers.
    """
    user_prompt = ONE_SHOT_USER_TEMPLATE.format(
        target_words=150,
        style="neutral technical",
        format_hint="3–5 sentences",
        text=text_to_be_summarized,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful & competent research assistant. "
                       "Summarise the abstract of this academic paper.",
        },
        {"role": "user", "content": user_prompt},
    ]

    result = summarizer_llm.invoke(messages)
    return getattr(result, "content", result)


def _compose_context_from_pieces(text: Optional[str], query: Optional[str], beliefs: Optional[Dict[str, Any]]) -> str:
    parts = []
    if query:
        parts.append(f"## User Query\n{query}")
    if beliefs:
        parts.append(f"## Beliefs / Preferences\n{beliefs}")
    if text:
        parts.append(f"## Extracted Text\n{text}")
    return "\n\n".join(parts).strip()


def generate_summary(
    context: str = None,
    text: str = None,
    query: str = None,
    beliefs: Dict[str, Any] = None,
    target_words: int = 250,
    style: str = "neutral technical",
    format_hint: str = "short paragraphs or bullets"
) -> str:
    """
    Adaptive summarizer used by the 'summary.generate' capability.
    Accepts either a full 'context' string or separate pieces (text, query, beliefs).
    """
    from app.main import llms
    summarizer_llm = llms["summarizer"]

    # Compose context if not explicitly provided
    if context:
        context_chunk = context
    else:
        parts = []
        if query:
            parts.append(f"## User Query\n{query}")
        if beliefs:
            parts.append(f"## Beliefs / Preferences\n{beliefs}")
        if text:
            parts.append(f"## Extracted Text\n{text}")
        context_chunk = "\n\n".join(parts).strip()

    if not context_chunk:
        return "No sufficient context was provided to summarize."

    user_prompt = CHUNK_USER_TEMPLATE.format(chunk=context_chunk)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful & competent research assistant. "
                f"Provide a {style} summary (~{target_words} words), "
                f"using {format_hint} where helpful."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    result = summarizer_llm.invoke(messages)
    return getattr(result, "content", result)
