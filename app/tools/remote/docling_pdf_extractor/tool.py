import requests

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool


def call_docling_pdf_extractor_remote(pdf_path: str) -> str:
    """
    Wrapper that calls the remote FastAPI service to extract Markdown from a PDF.
    """
    response = requests.post(
        "http://127.0.0.1:8110/extract",  # Replace with actual server address if deployed
        params={"file_path": pdf_path},   # FastAPI expects file_path as query param
        timeout=30  # Adjust as needed
    )
    if response.status_code != 200:
        raise RuntimeError(f"Docling extractor failed: {response.status_code} - {response.text}")
    return response.text


class DoclingExtractorInput(BaseModel):
    pdf_path: str = Field(..., description="Absolute or relative path to the PDF file")

docling_pdf_tool = StructuredTool.from_function(
    func=call_docling_pdf_extractor_remote,
    name="docling_pdf_extractor",
    description=(
        "Extract structured markdown text from academic PDFs using Docling. "
        "Especially useful for PDFs with complex layouts, math, or figures."
    ),
    args_schema=DoclingExtractorInput,
    return_direct=True
)
