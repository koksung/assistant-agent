import os
import re
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from app.utils.logger import get_logger
from app.tools.remote.nougat_pdf_extractor import call_nougat_api

logger = get_logger(__name__)

# Enable LaTeX rendering using actual LaTeX (requires TeX installation)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{mathtools}"
})


# ---------------------------- Pydantic schema -----------------------------

class EquationRendererInput(BaseModel):
    """
    Args schema for the StructuredTool.
    - If `markdown_text` is empty and you pass `pdf_path`, we will call Nougat to get markdown first.
    """
    markdown_text: str = Field("", description="Markdown text potentially containing LaTeX math (\\[...\\] or \\(...\\)).")
    pdf_path: str = Field("", description="Optional: PDF path to run through Nougat when markdown_text is empty.")
    include_inline: bool = Field(default=False, description="Also extract inline equations \\(...\\).")
    out_dir: str = Field(default="data/latex_equations", description="Output directory for rendered PNGs.")
    dpi: int = Field(default=300, ge=72, le=600, description="PNG DPI.")
    fontsize: int = Field(default=20, ge=6, le=64, description="Font size for rendering.")


# ------------------------------ Utilities --------------------------------

def _latex_str_hash(latex_str: str) -> str:
    return hashlib.md5(latex_str.encode("utf-8")).hexdigest()[:8]

def _sanitize_latex_for_matplotlib(eqn: str) -> str:
    eqn = re.sub(r"\\tag\{[^}]*}", "", eqn)         # remove \tag{...}
    eqn = eqn.replace(r"\coloneqq", r":=")           # normalize coloneqq
    eqn = eqn.rstrip("\\").strip()                   # trailing slashes/spaces
    eqn = re.sub(r"_\{\s*(\\\w+)\s*}", r"_{\1}", eqn)
    eqn = re.sub(r"\\\s+", r"\\", eqn)               # cleanup stray spaces after backslashes
    return eqn

def _extract_latex_equations(markdown_text: str, include_inline: bool = False) -> List[str]:
    """
    Extract LaTeX from Nougat-style markdown:
      - Display: \\[ ... \\]   (default)
      - Inline:  \\( ... \\)   (optional)
    Unescapes double backslashes.
    """
    inline_eqs = re.findall(r"\\\((.+?)\\\)", markdown_text) if include_inline else []
    display_eqs = re.findall(r"\\\[(.+?)\\]", markdown_text, re.DOTALL)
    all_eqs = inline_eqs + display_eqs
    cleaned = [eq.replace("\\\\", "\\").strip() for eq in all_eqs]
    return cleaned

def _render_single_equation(eqn: str, out_dir: str, dpi: int, fontsize: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    latex_eq = r"$" + _sanitize_latex_for_matplotlib(eqn) + r"$"
    ax.text(0.5, 0.5, latex_eq, fontsize=fontsize, ha="center", va="center")
    filename = os.path.join(out_dir, f"equation_{_latex_str_hash(eqn)}.png")
    plt.savefig(filename, bbox_inches="tight", dpi=dpi, transparent=True)
    plt.close(fig)
    normalized_fn = os.path.normpath(filename)
    logger.info(f"[equation_renderer] Saved: {normalized_fn}")
    return normalized_fn


# ------------------------------ Public API --------------------------------

def render_equation_images(
    markdown_text: str,
    out_dir: str = "data/latex_equations",
    include_inline: bool = False,
    dpi: int = 300,
    fontsize: int = 20
) -> Dict[str, Any]:
    """
    Core renderer (callable directly if you don't need Nougat).
    Returns: {"count": int, "images": [paths], "failed": [latex_snippets]}
    """
    equations = _extract_latex_equations(markdown_text or "", include_inline=include_inline)
    images, failed = [], []
    for eq in equations:
        try:
            images.append(_render_single_equation(eq, out_dir=out_dir, dpi=dpi, fontsize=fontsize))
        except Exception as e:
            logger.exception(f"[equation_renderer] Failed to render: {eq} | {e}")
            failed.append(eq)
    return {"count": len(images), "images": images, "failed": failed}


def render_equation(md_input: EquationRendererInput) -> Dict[str, Any]:
    """
    StructuredTool entrypoint.
    If `markdown_text` is empty and `pdf_path` is provided, calls Nougat first to get markdown.
    """
    md = md_input.markdown_text
    if not md and md_input.pdf_path:
        try:
            md = call_nougat_api(md_input.pdf_path)
        except Exception as e:
            logger.exception(f"[equation_renderer] Nougat call failed for {md_input.pdf_path}: {e}")
            return {"count": 0, "images": [], "failed": [], "error": f"Nougat failed: {e}"}

    if not md:
        return {"count": 0, "images": [], "failed": [], "error": "No markdown_text provided and no pdf_path to derive it."}

    return render_equation_images(
        markdown_text=md,
        out_dir=md_input.out_dir,
        include_inline=md_input.include_inline,
        dpi=md_input.dpi,
        fontsize=md_input.fontsize,
    )
