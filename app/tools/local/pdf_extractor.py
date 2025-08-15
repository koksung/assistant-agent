import fitz  # PyMuPDF

def extract_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        # noinspection PyUnresolvedReferences
        text += page.get_text() + "\n"
    return text.strip()
