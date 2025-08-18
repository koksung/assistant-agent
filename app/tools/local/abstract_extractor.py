from app.tools.local.pdf_extractor import extract_pdf_text
from app.tools.local.summarizer_tool import summarize_abstract
from pydantic import BaseModel, Field


class AbstractSummaryInput(BaseModel):
    pdf_path: str = Field(..., description="Path to the PDF to summarize (abstract-first).")


def abstract_extraction(pdf_path: str):
    result = extract_pdf_text(pdf_path)
    if "abstract" in result:
        return result["abstract"]
    else:
        return result.get("introduction", "")


def abstract_summary(pdf_path: str, summarizer_llm):
    abstract_text = abstract_extraction(pdf_path)
    return summarize_abstract(abstract_text, summarizer_llm)
