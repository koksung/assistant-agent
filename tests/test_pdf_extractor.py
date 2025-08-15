import os
from app.tools.local.pdf_extractor import extract_pdf_text

def test_extract_pdf_text():
    test_path = "data/ddpm.pdf"
    assert os.path.exists(test_path), f"Test PDF not found at {test_path}"

    extracted_text = extract_pdf_text(test_path)

    # Basic checks
    assert isinstance(extracted_text, str), "Output should be a string"
    assert len(extracted_text) > 100, "Extracted text is too short — possibly failed extraction"
    assert "abstract" in extracted_text.lower(), "Expected 'abstract' not found in extracted text"
