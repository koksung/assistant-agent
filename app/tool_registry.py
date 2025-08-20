from langchain.tools import StructuredTool
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# --- Enums / hints -----------------------------------------------------------
class CostHint(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class LatencyClass(Enum):
    LOW = 100       # ~< 200ms
    MID = 800       # ~0.2–1s
    HIGH = 1500     # >1s


@dataclass
class ToolProfile:
    """
    Wraps a LangChain StructuredTool with routing-friendly metadata.
    """
    tool: StructuredTool
    purpose: str
    strengths: str
    limitations: str
    # routing fields
    capabilities: List[str] = field(default_factory=list)  # e.g. ["pdf.extract","equations"]
    requires_network: bool = False
    cost_hint: CostHint = CostHint.LOW
    latency_hint_ms: int = LatencyClass.LOW.value
    # Human-facing (optional) mirrors for UI/logs
    cost_label: Optional[str] = None
    latency_label: Optional[str] = None

    def _inputs_dict(self) -> Dict[str, str]:
        """
        Robustly extract arg annotations across Pydantic/LC versions.
        """
        schema = getattr(self.tool, "args_schema", None)
        ann = getattr(schema, "__annotations__", None)
        if isinstance(ann, dict):
            return {k: getattr(v, "__name__", str(v)) for k, v in ann.items()}
        # Fallback: try pydantic model_fields (v2) or __fields__ (v1)
        fields = getattr(schema, "model_fields", None) or getattr(schema, "__fields__", None)
        if isinstance(fields, dict):
            return {k: str(v.annotation) for k, v in fields.items()}
        return {}

    def to_string(self) -> str:
        return (
            f"{self.tool.name}: {self.purpose} | "
            f"Inputs: {self._inputs_dict()} | Outputs: [see tool description] | "
            f"Strengths: {self.strengths} | Limitations: {self.limitations}"
        )

    def describe(self) -> dict:
        return {
            "name": self.tool.name,
            "purpose": self.purpose,
            "inputs": self._inputs_dict(),
            "outputs": "Structured markdown and/or LaTeX",
            "strengths": self.strengths,
            "limitations": self.limitations,
            "capabilities": self.capabilities,
            "requires_network": self.requires_network,
            "cost_hint": self.cost_hint.name,
            "latency_hint_ms": self.latency_hint_ms,
            "cost_label": self.cost_label,
            "latency_label": self.latency_label,
        }


@dataclass
class ToolCallContext:
    user_id: str
    archetypes: List[str] = field(default_factory=list)  # e.g. ["privacy_sensitive","math_heavy","visual_learner"]
    preferences: Dict[str, Any] = field(default_factory=dict)  # e.g. {"prefer_local": True}
    allow_remote: bool = True
    privacy_level: str = "standard"  # or "strict"
    max_latency_ms: Optional[int] = None
    budget_level: Optional[CostHint] = None  # constrain to <= level


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolProfile] = {}

    # --- registration / retrieval -------------------------------------------
    def register_tool(self, rich_tool: ToolProfile, *, override: bool = False):
        name = rich_tool.tool.name
        if not override and name in self.tools:
            raise ValueError(f"Tool '{name}' already registered")
        self.tools[name] = rich_tool

    def get_tool(self, name: str) -> Optional[ToolProfile]:
        return self.tools.get(name)

    def list_all_tools(self) -> Dict[str, ToolProfile]:
        return dict(self.tools)

    def by_capability(self, capability: str) -> List[ToolProfile]:
        return [t for t in self.tools.values() if capability in t.capabilities]

    # --- routing -------------------------------------------------------------
    def resolve(self, capability: str, ctx: ToolCallContext) -> Optional[ToolProfile]:
        candidates = self.by_capability(capability)
        if not candidates:
            return None

        def score(t: ToolProfile) -> float:
            s = 0.0
            # privacy / remote allowance
            if t.requires_network and (not ctx.allow_remote or ctx.privacy_level == "strict"):
                s -= 100.0
            # hard budget cap
            if ctx.budget_level is not None and t.cost_hint.value > ctx.budget_level.value:
                s -= 50.0
            # latency budget
            if ctx.max_latency_ms is not None:
                over = max(0, t.latency_hint_ms - ctx.max_latency_ms)
                s -= over / 100.0

            # archetype-guided nudges
            if "privacy_sensitive" in ctx.archetypes and not t.requires_network:
                s += 5.0
            if "math_heavy" in ctx.archetypes and "equations" in t.capabilities:
                s += 3.0
            if "visual_learner" in ctx.archetypes and "layout" in t.capabilities:
                s += 2.0

            # user preferences
            prefs = ctx.preferences or {}
            if prefs.get("prefer_console_equations") and t.tool.name == "latex_console_renderer":
                s += 6.0
            if prefs.get("prefer_image_equations") and t.tool.name == "equation_renderer":
                s += 5.0
            if ctx.preferences.get("prefer_local") and not t.requires_network:
                s += 2.0

            # general tie-breakers: cheaper & faster first
            s -= t.cost_hint.value * 0.5
            s -= t.latency_hint_ms / 2000.0  # normalize
            return s

        return sorted(candidates, key=score, reverse=True)[0]

# --- Initialization of actual tools -------------------------------------

# Imports kept here to avoid circulars in some runners
from app.tools.remote.nougat_pdf_extractor import nougat_pdf_extraction, NougatPdfExtractorInput
from app.tools.remote.docling_pdf_extractor.tool import call_docling_pdf_extractor_remote, DoclingExtractorInput
from app.tools.local.pdf_extractor import extract_pdf_text, LocalPdfExtractorInput
from app.tools.local.equation_renderer import EquationRendererInput, LatexConsoleRendererInput, parse_latex_equation, render_equation
from app.tools.local.summarizer_tool import GenerateSummaryInput, generate_summary
from app.tools.local.abstract_extractor import abstract_summary as _abstract_summary_fn, AbstractSummaryInput
from app.tools.local.pdf_summarizer import PdfSummarizeInput, pdf_summarize_adapter

def _latex_console_renderer_adapter(latex_string: str) -> str:
    return parse_latex_equation(latex_string)


def _extract_pdf_text_adapter(pdf_path: str, **kwargs):
    # Call the original function positionally so its param name doesn’t matter
    return extract_pdf_text(pdf_path)


def _equation_renderer_adapter(
    markdown_text: str = "",
    pdf_path: str = "",
    include_inline: bool = False,
    out_dir: str = "data/latex_equations",
    dpi: int = 300,
    fontsize: int = 20,
):
    payload = EquationRendererInput(
        markdown_text=markdown_text,
        pdf_path=pdf_path,
        include_inline=include_inline,
        out_dir=out_dir,
        dpi=dpi,
        fontsize=fontsize,
    )
    return render_equation(payload)


def _nougat_adapter(**kwargs) -> Any:
    """
    Adapter so StructuredTool can pass validated kwargs.
    Prefer calling Nougat with a `str pdf_path`, but fall back to model or kwargs.
    """
    model = NougatPdfExtractorInput(**kwargs)

    # primary: most implementations accept a plain path string
    if hasattr(model, "pdf_path"):
        try:
            return nougat_pdf_extraction(model.pdf_path)  # type: ignore[arg-type]
        except TypeError:
            pass

    # fallback 1: function accepts the Pydantic model
    try:
        return nougat_pdf_extraction(model)  # type: ignore[arg-type]
    except TypeError:
        # fallback 2: function accepts kwargs
        return nougat_pdf_extraction(**model.model_dump())


def _docling_adapter(**kwargs) -> Any:
    """
    Adapter so StructuredTool can pass validated kwargs.
    Prefer calling docling with a `str pdf_path`, but fall back to model or kwargs.
    """
    model = DoclingExtractorInput(**kwargs)

    # primary: most implementations want a plain path string
    if hasattr(model, "pdf_path"):
        try:
            return call_docling_pdf_extractor_remote(model.pdf_path)  # type: ignore[arg-type]
        except TypeError:
            pass

    # fallback 1: function accepts the Pydantic model
    try:
        return call_docling_pdf_extractor_remote(model)  # type: ignore[arg-type]
    except TypeError:
        # fallback 2: function accepts kwargs
        return call_docling_pdf_extractor_remote(**model.model_dump())


def _abstract_summary_adapter(pdf_path: str) -> str:
    """
    Extracts the abstract (or falls back to introduction) and returns a concise summary.
    Uses the project's configured summarizer LLM.
    """
    from app.main import llms
    summarizer_llm = llms["summarizer"]   # your existing map of LLMs
    return _abstract_summary_fn(pdf_path, summarizer_llm)


def initialize_tool_registry() -> ToolRegistry:
    r = ToolRegistry()

    # Local PyMuPDF extractor
    local_pdf_extractor = StructuredTool.from_function(
        func=_extract_pdf_text_adapter,
        name="local_pdf_extractor",
        description="Extract structured markdown text from academic PDFs using PyMuPDF.",
        args_schema=LocalPdfExtractorInput,
        return_direct=True
    )
    r.register_tool(ToolProfile(
        tool=local_pdf_extractor,
        purpose="Extract structured academic content from simpler PDFs.",
        strengths="Fast, local, good paragraph extraction.",
        limitations="May miss some sections/blocks; limited equation fidelity.",
        capabilities=["pdf.extract"],
        requires_network=False,
        cost_hint=CostHint.LOW,
        latency_hint_ms=LatencyClass.LOW.value,
        latency_label="Low",
    ))

    # Remote Nougat extractor (equations focus)
    remote_nougat_tool = StructuredTool.from_function(
        func=_nougat_adapter,
        name="advanced_pdf_extractor_1",
        description="Extract structured markdown and LaTeX equations from academic PDFs using Nougat.",
        args_schema=NougatPdfExtractorInput,
        return_direct=True
    )
    r.register_tool(ToolProfile(
        tool=remote_nougat_tool,
        purpose="Handle complex layouts with equations.",
        strengths="Better section structure and LaTeX equations.",
        limitations="Requires internet and external API availability.",
        capabilities=["pdf.extract", "equations"],
        requires_network=True,
        cost_hint=CostHint.LOW,
        latency_hint_ms=LatencyClass.MID.value,
        latency_label="Mid",
    ))

    # Remote Docling extractor (layout/structure focus)
    remote_docling_tool = StructuredTool.from_function(
        func=_docling_adapter,
        name="advanced_pdf_extractor_2",
        description="Extract structured markdown text from academic PDFs using Docling.",
        args_schema=DoclingExtractorInput,
        return_direct=True
    )
    r.register_tool(ToolProfile(
        tool=remote_docling_tool,
        purpose="Robust extraction on complex layouts and sections.",
        strengths="Good sectioning and structural text extraction.",
        limitations="Requires internet and external API availability.",
        capabilities=["pdf.extract", "layout"],
        requires_network=True,
        cost_hint=CostHint.MEDIUM,
        latency_hint_ms=LatencyClass.HIGH.value,
        latency_label="High",
    ))

    # Equation Renderer (local → images)
    equation_renderer_tool = StructuredTool.from_function(
        func=_equation_renderer_adapter,
        name="equation_renderer",
        description="Render LaTeX equations (from markdown or via Nougat) into PNG images. Returns list of file paths.",
        args_schema=EquationRendererInput,
        return_direct=True
    )
    r.register_tool(ToolProfile(
        tool=equation_renderer_tool,
        purpose="Convert LaTeX math into rendered images for math- or visual-oriented users.",
        strengths="High-fidelity LaTeX rendering (requires local TeX); flexible input (markdown or PDF).",
        limitations="Requires a LaTeX installation for usetex; invalid LaTeX may fail to render.",
        capabilities=["equations", "visualization"],
        requires_network=False,  # local rendering
        cost_hint=CostHint.LOW,
        latency_hint_ms=LatencyClass.LOW.value,
        latency_label="Low",
    ))

    # LaTeX Console Renderer (local → Unicode text)
    latex_console_renderer_tool = StructuredTool.from_function(
        func=_latex_console_renderer_adapter,
        name="latex_console_renderer",
        description="Parse a single LaTeX equation into console-friendly Unicode text using Sympy, ideal for CLI display of specific equations from queries like 'Which equation is most important?'.",
        args_schema=LatexConsoleRendererInput,
        return_direct=True
    )
    r.register_tool(ToolProfile(
        tool=latex_console_renderer_tool,
        purpose="Render a single LaTeX equation as Unicode text for console-based display, suitable for math-oriented queries in a CLI environment.",
        strengths="Fast, local parsing with Sympy; produces clean Unicode for CLI; handles specific equation queries.",
        limitations="Limited to valid LaTeX equations; complex equations may have imperfect Unicode rendering.",
        capabilities=["equations"],
        requires_network=False,
        cost_hint=CostHint.LOW,
        latency_hint_ms=LatencyClass.LOW.value,
        latency_label="Low",
    ))

    # Abstract-first summary (local extract + LLM summarize)
    abstract_summary_tool = StructuredTool.from_function(
        func=_abstract_summary_adapter,
        name="abstract_summary",
        description="Summarize a paper using its Abstract (fallback to Introduction if no Abstract found). "
                    "Returns a concise bullet/paragraph summary. Excellent for providing an initial summary.",
        args_schema=AbstractSummaryInput,
        return_direct=True
    )
    r.register_tool(ToolProfile(
        tool=abstract_summary_tool,
        purpose="Give users a fast, low-latency entry point: extract Abstract → summarize.",
        strengths="Very quick perceived results; robust fallback to Introduction; good for first-touch overview.",
        limitations="If the paper's abstract is unusually sparse or promotional, the summary may miss deeper method/results details.",
        capabilities=["abstract.summarize"],  # <- new capability tag
        requires_network=True,  # set True if your summarizer LLM is remote; set False if local
        cost_hint=CostHint.LOW,  # LLM call cost
        latency_hint_ms=LatencyClass.MID.value,  # typically ~0.2–1s
        latency_label="Mid",
        cost_label="Low",
    ))

    # Generate summary
    generate_summary_tool = StructuredTool.from_function(
        func=generate_summary,
        name="generate_summary",
        description=" Produce an adaptive summary from gathered context.",
        args_schema=GenerateSummaryInput,
        return_direct=True
    )
    r.register_tool(ToolProfile(
        tool=generate_summary_tool,
        purpose="Provides a good high-level summary of the current state of things via the context.",
        strengths="Provides a good high-level summary of the current state of things via the context.",
        limitations="If context is off, summary might be off too.",
        capabilities=["summary.generate"],  # <- new capability tag
        requires_network=True,  # set True if your summarizer LLM is remote; set False if local
        cost_hint=CostHint.MEDIUM,  # LLM call cost
        latency_hint_ms=LatencyClass.MID.value,  # typically ~0.2–1s
        latency_label="Mid",
        cost_label="Mid",
    ))

    # --- Full-text PDF summarizer (capability: pdf.summarize) -----------
    pdf_summarizer_tool = StructuredTool.from_function(
        func=pdf_summarize_adapter,
        name="pdf_summarizer",
        description="Summarize arbitrary full text (already extracted from the PDF) into a concise overview.",
        args_schema=PdfSummarizeInput,
        return_direct=True
    )
    r.register_tool(ToolProfile(
        tool=pdf_summarizer_tool,
        purpose="Provide a concise, faithful summary of the full paper text.",
        strengths="Works on any supplied text; yields consistent markdown summaries; controllable length.",
        limitations="Quality depends on the quality/coverage of the provided text; may miss tables/figures not in text.",
        capabilities=["pdf.summarize"],
        requires_network=True,  # uses your remote summarizer LLM
        cost_hint=CostHint.MEDIUM,  # adjust as you see fit
        latency_hint_ms=LatencyClass.MID.value,
        latency_label="Mid",
        cost_label="Mid",
    ))

    return r


registry: Optional[ToolRegistry] = None

def set_global_registry(reg: ToolRegistry):
    global registry
    registry = reg

def get_registry() -> Optional[ToolRegistry]:
    return registry

def get_tools() -> Dict[str, ToolProfile]:
    return registry.list_all_tools() if registry else {}
