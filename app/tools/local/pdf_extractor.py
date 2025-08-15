import fitz  # PyMuPDF
import re
from typing import Dict


def split_into_sections(text: str) -> Dict[str, str]:
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Ensure Abstract header is present
    if "abstract" not in text[:500].lower():
        text = "Abstract\n" + text

    section_pattern = re.compile(
        r"(?P<header>^\s*(\d{0,2}[\.]?\s*)?(abstract|introduction|related work|background|method(?:s|ology)?|experiments?|results?|discussion|conclusion|references))",
        re.IGNORECASE | re.MULTILINE
    )

    matches = list(section_pattern.finditer(text))

    sections = {}

    # Step 1: Capture everything BEFORE first section as "preamble"
    if matches:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections["preamble"] = preamble

    # Step 2: Capture each section
    for i, match in enumerate(matches):
        section_title = match.group("header").strip().lower().replace('.', '')
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        normalized_title = normalize_section_name(section_title)
        sections[normalized_title] = section_text

    return sections


def normalize_section_name(raw: str) -> str:
    """Maps noisy section headers to normalized keys."""
    name = raw.lower()
    if "abstract" in name: return "abstract"
    if "introduction" in name: return "introduction"
    if "related" in name or "background" in name: return "related_work"
    if "method" in name or "approach" in name or "model" in name: return "method"
    if "experiment" in name or "evaluation" in name or "result" in name: return "experiments"
    if "discussion" in name or "analysis" in name: return "discussion"
    if "conclusion" in name or "summary" in name: return "conclusion"
    if "reference" in name or "bibliography" in name: return "references"
    return re.sub(r'\W+', '_', name).strip("_")


def extract_pdf_text(path: str) -> Dict[str, str]:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        # noinspection PyUnresolvedReferences
        text += page.get_text() + "\n"
    return split_into_sections(text.strip())
