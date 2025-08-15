from app.utils.device import configure_torch_device
from docling.document_converter import DocumentConverter

configure_torch_device()
# Load once — it's expensive
converter = DocumentConverter()


def extract_pdf_with_docling(file_path: str) -> str:
    """
    Extracts text (Markdown format) from a PDF using Docling's model.
    Returns a Markdown string.
    """
    result = converter.convert(file_path)
    return result.document.export_to_markdown()
