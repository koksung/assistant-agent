import requests

from pydantic import BaseModel, Field


def call_docling_pdf_extractor_remote(file_path: str) -> str:
    """
    Wrapper that calls the remote FastAPI service to extract Markdown from a PDF.
    """
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "application/pdf")}
        response = requests.post("http://127.0.0.1:8110/extract", files=files, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"Docling extractor failed: {response.status_code} - {response.text}")
    return response.text


class DoclingExtractorInput(BaseModel):
    pdf_path: str = Field(..., description="Absolute or relative path to the PDF file")
