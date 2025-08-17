import re
import requests

from pydantic import BaseModel, Field
from typing import Dict
from app.tools.local.pdf_extractor import normalize_section_name


def split_markdown_sections(md: str) -> Dict[str, str]:
    # Matches lines like: ## Abstract, ## 1 Introduction
    header_pattern = re.compile(r"^##\s+(.*)", re.MULTILINE)
    matches = list(header_pattern.finditer(md))

    sections = {}

    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()

        normalized = normalize_section_name(header)
        sections[normalized] = body

    return sections


def call_docling_pdf_extractor_remote(file_path: str) -> dict:
    """
    Wrapper that calls the remote FastAPI service to extract Markdown from a PDF.
    """
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "application/pdf")}
        response = requests.post("http://127.0.0.1:8110/extract", files=files, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"Docling extractor failed: {response.status_code} - {response.text}")
    markdown = response.text
    return split_markdown_sections(markdown)


class DoclingExtractorInput(BaseModel):
    pdf_path: str = Field(..., description="Absolute or relative path to the PDF file")
