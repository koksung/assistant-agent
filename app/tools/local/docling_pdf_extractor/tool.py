from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from docling_pdf_extractor import extract_pdf_with_docling

class DoclingExtractorInput(BaseModel):
    pdf_path: str = Field(..., description="Absolute or relative path to the PDF file")

docling_pdf_tool = StructuredTool.from_function(
    func=extract_pdf_with_docling,
    name="docling_pdf_extractor",
    description=(
        "Extract structured markdown text from academic PDFs using Docling. "
        "Especially useful for PDFs with complex layouts, math, or figures."
    ),
    args_schema=DoclingExtractorInput,
    return_direct=True
)
