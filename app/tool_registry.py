from langchain.tools import StructuredTool
from typing import Optional

from torch import initial_seed

from app.tools.remote.docling_pdf_extractor.tool import call_docling_pdf_extractor_remote, DoclingExtractorInput
from app.tools.local.pdf_extractor import extract_pdf_text, LocalPdfExtractorInput


class ToolProfile:
    def __init__(
        self,
        tool: StructuredTool,
        purpose: str,
        strengths: str,
        limitations: str,
        cost: Optional[str] = None,
        latency: Optional[str] = None,
    ):
        self.tool = tool  # This is a LangChain StructuredTool
        self.purpose = purpose
        self.strengths = strengths
        self.limitations = limitations
        self.cost = cost
        self.latency = latency

    def describe(self) -> str:
        return (
            f"{self.tool.name}: {self.purpose} | Inputs: {self.tool.args_schema.__annotations__} "
            f"| Outputs: [see tool description] | Strengths: {self.strengths} | Limitations: {self.limitations}"
        )


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register_tool(self, rich_tool: ToolProfile):
        self.tools[rich_tool.tool.name] = rich_tool

    def get_tool(self, name: str) -> Optional[ToolProfile]:
        return self.tools.get(name)

    def list_all_tools(self) -> dict:
        return self.tools


def initialize_tool_registry() -> ToolRegistry:
    initial_registry = ToolRegistry()

    local_pdf_extractor = StructuredTool.from_function(
        func=extract_pdf_text,
        name="local_pdf_extractor",
        description="Extract structured markdown text from academic PDFs using PyMuPDF.",
        args_schema=LocalPdfExtractorInput,
        return_direct=True
    )
    initial_registry.register_tool(ToolProfile(
        tool=local_pdf_extractor,
        purpose="Extract structured academic content from PDFs with simple layouts.",
        strengths="Extracts PDF text paragraph fairly fast.",
        limitations="Might miss out certain sections, blocks of text.",
        cost="Low",
        latency="Low"
    ))

    remote_docling_tool = StructuredTool.from_function(
        func=call_docling_pdf_extractor_remote,
        name="advanced_pdf_extractor",
        description="Extract structured markdown text from academic PDFs using Docling.",
        args_schema=DoclingExtractorInput,
        return_direct=True
    )
    initial_registry.register_tool(ToolProfile(
        tool=remote_docling_tool,
        purpose="Extract structured academic content from PDFs with complex layouts.",
        strengths="Handles much better pdf text extraction for sections.",
        limitations="Requires internet and external API availability.",
        cost="Moderate (external API)",
        latency="High"
    ))

    return initial_registry

# This will hold the global ToolRegistry instance
registry: Optional[ToolRegistry] = None

def set_global_registry(reg):
    global registry
    registry = reg


def get_registry() -> Optional[ToolRegistry]:
    return registry


def get_tools():
    return registry.list_all_tools() if registry else {}
