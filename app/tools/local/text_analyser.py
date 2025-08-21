from typing import Dict, Any
from pydantic import BaseModel, Field
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextAnalysisInput(BaseModel):
    text: str = Field(..., description="Paper text content to analyze.")
    query: str = Field(..., description="Specific question to answer about the text.")
    context: str = Field(default="", description="Optional additional context to guide analysis.")


def analyze_text_adapter(text: str, query: str, context: str = "") -> Dict[str, Any]:
    """Adapter for LangChain StructuredTool: accepts schema fields directly."""
    from app.main import llms  # reuse your configured LLMs
    inp = TextAnalysisInput(text=text, query=query, context=context)
    return analyze_text_content(inp, llms["summarizer"])


def analyze_text_content(input_data: TextAnalysisInput, llm) -> Dict[str, Any]:
    """
    Analyze text content to answer specific questions about importance,
    relationships, key concepts, etc.
    """

    analysis_prompt = f"""
    You are an expert academic assistant. Analyze the paper text to answer the user’s question.

    ## User Question
    {input_data.query}

    ## Additional Context
    {input_data.context}

    ## Paper Content (excerpt)
    {input_data.text}

    ### Instructions
    - Answer the question directly and specifically.
    - If the question is about equations:
      - Identify the most important equation(s).
      - Provide the LaTeX form and explain its role in the paper.
    - If the question is about a section (e.g. "gist of section 3"):
      - Summarize only that section, in a few clear sentences.
    - If the question is about derivations or clarity:
      - Comment on whether derivations are detailed, easy to follow, or complex.
    - If the question is about related work/background:
      - Evaluate whether it is comprehensive, relevant, or superficial.
    - Always back up claims with direct references to the text.
    - Keep your response focused (200 words max).
    """.strip()

    try:
        response = llm.invoke(analysis_prompt)
        content = getattr(response, "content", str(response))
        return {"analysis": content, "query": input_data.query, "success": True}
    except Exception as e:
        logger.exception(f"[text_analyser] Analysis failed: {e}")
        return {"analysis": f"Analysis failed: {str(e)}", "query": input_data.query, "success": False}
