import os
from app.tools.local.pdf_extractor import extract_pdf_text

def test_extract_pdf_text():
    test_path = "data/ddpm-short.pdf"
    assert os.path.exists(test_path), f"Test PDF not found at {test_path}"

    extracted = extract_pdf_text(test_path)

    # New checks for dict-based structure
    assert isinstance(extracted, dict), "Output should be a dictionary"
    assert "abstract" in extracted, "Missing 'abstract' section"
    assert isinstance(extracted["abstract"], str), "'abstract' should be a string"
    assert len(extracted["abstract"]) > 50, "Abstract seems too short"
