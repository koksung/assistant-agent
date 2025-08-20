from pydantic import BaseModel, Field


class PdfSummarizeInput(BaseModel):
    """
    Provide raw text to summarize; optional size control.
    """
    text: str = Field(..., description="Raw extracted text to summarize.")
    max_words: int = Field(default=250, ge=50, le=1000, description="Target length for the summary (approx).")


def pdf_summarize_adapter(text: str, max_words: int = 250) -> str:
    """
    Summarize arbitrary full text using the project's summarizer LLM.
    Keeps prompt self-contained so it's usable standalone.
    """
    from app.main import llms
    summarizer_llm = llms["summarizer"]

    prompt = f"""
    You are a scientific writing assistant. Summarize the following text into a clear, faithful overview.
    - Focus on the paper's problem, approach, key results, and notable limitations.
    - Keep it concise and coherent; avoid marketing language.
    - Target length: ~{max_words} words.
    - Use markdown with short paragraphs or bullets when helpful.
    
    TEXT:
    \"\"\"{text}\"\"\"
    """.strip()

    resp = summarizer_llm.invoke(prompt)
    return getattr(resp, "content", resp)
