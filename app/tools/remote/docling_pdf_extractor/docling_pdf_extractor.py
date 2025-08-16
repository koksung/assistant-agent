import re
import html

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager

from app.utils.device import configure_torch_device
from app.utils.logger import get_logger

from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    TableStructureOptions,
    PdfPipelineOptions
)
pipeline_options = PdfPipelineOptions(
    do_ocr = False,
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        do_cell_matching=True
    ),
    ocr_options=EasyOcrOptions(
        lang=["en"],
        force_full_page_ocr=True
    ),
    accelerator_options=AcceleratorOptions(
        num_threads = 4,
        device = AcceleratorDevice.CUDA
    ),
    generate_page_images=False,
    generate_picture_images=False
)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(fapp: FastAPI):
    # STARTUP
    configure_torch_device()

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        }
    )
    fapp.state.converter = converter
    print("Docling converter loaded.")

    yield  # The application runs during this point

    # SHUTDOWN (if any cleanup needed)
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


def clean_docling_markdown(text: str) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        # Remove known placeholders
        if "<!-- image -->" in line or "<!-- formula-not-decoded -->" in line:
            continue

        # Remove likely base64-like encoded LaTeX images
        if re.match(r'^[A-Za-z0-9+/=]{30,}$', line.strip()):
            continue

        # Remove isolated single characters (dribble lines)
        if len(line.strip()) <= 2:
            continue

        # Remove lines with decoded LaTeX base64 blobs
        if re.match(r'^[<>\\_a-zA-Z0-9=/:;+"\'\-\s]{10,}$', line.strip()) and len(set(line.strip())) > 10:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def final_clean(markdown: str) -> str:
    # Unescape HTML (e.g., &lt; becomes <)
    markdown = html.unescape(markdown)

    # Optionally, remove lonely lines with just < or > if they still exist
    lines = []
    for line in markdown.splitlines():
        if line.strip() in ("<", ">"):
            continue
        lines.append(line)
    return "\n".join(lines)


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.post("/extract", response_class=PlainTextResponse)
async def extract_pdf_with_docling(file_path: str):
    # Access the converter from app state
    converter: DocumentConverter = app.state.converter
    try:
        result = converter.convert(file_path)
        markdown = result.document.export_to_markdown()
        markdown = clean_docling_markdown(markdown)
        return final_clean(markdown)
    except Exception as e:
        logger.exception(f"Extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

# run it
# uvicorn app.tools.remote.docling_pdf_extractor.docling_pdf_extractor:app --host 127.0.0.1 --port 8110
