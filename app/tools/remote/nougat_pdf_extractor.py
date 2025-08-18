import re
import requests
from typing import Dict
from pydantic import BaseModel, Field

from app.utils.logger import get_logger
from app.tools.local.pdf_extractor import normalize_section_name
logger = get_logger(__name__)


class NougatPdfExtractorInput(BaseModel):
    pdf_path: str = Field(..., description="Absolute or relative path to the PDF file")


def split_markdown_sections(md: str) -> Dict[str, str]:
    """
    Split a markdown string into sections based on ## headers.
    Returns a dict: {section_name: section_body}
    """
    # Convert literal "\n" into actual newlines
    md = md.encode().decode("unicode_escape")

    header_pattern = re.compile(r"^#{2,6}\s+(.*)", re.MULTILINE)
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


def nougat_pdf_extraction(pdf_path: str):
    result = call_nougat_api(pdf_path)
    return split_markdown_sections(result)


def call_nougat_api(pdf_path: str, start: int = None, stop: int = None):
    url = "http://127.0.0.1:8503/predict/"

    params = {}
    if start is not None:
        params["start"] = start
    if stop is not None:
        params["stop"] = stop

    with open(pdf_path, "rb") as file:
        files = {
            "file": (pdf_path, file, "application/pdf")
        }
        response = requests.post(url, files=files, params=params)

    if response.status_code != 200:
        raise RuntimeError(f"Nougat API failed ({response.status_code}): {response.text}")

    return response.text
