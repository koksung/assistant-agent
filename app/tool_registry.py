from app.prompts.summarizer_prompt import get_prompt
from app.tools.local import summarizer_tool
from app.tools.remote.docling_pdf_extractor.tool import docling_pdf_tool

def get_tools():
    return {
        "summarizer": summarizer_tool.Summarizer(get_prompt()),
        "advanced_pdf_extractor": docling_pdf_tool
    }
