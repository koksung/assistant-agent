from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
from app.utils.device import configure_torch_device
from docling.document_converter import DocumentConverter


@asynccontextmanager
async def lifespan(fapp: FastAPI):
    # STARTUP
    configure_torch_device()
    converter = DocumentConverter()
    fapp.state.converter = converter
    print("Docling converter loaded.")

    yield  # The application runs during this point

    # SHUTDOWN (if any cleanup needed)
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


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
        return markdown
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

# run it
# uvicorn main:app --host 127.0.0.1 --port 8110
