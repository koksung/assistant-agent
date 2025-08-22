import os
import re
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from sympy import pretty
from sympy.parsing.latex import parse_latex
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.utils.logger import get_logger
from app.tools.remote.nougat_pdf_extractor import call_nougat_api

logger = get_logger(__name__)

# Enable LaTeX rendering using actual LaTeX (requires TeX installation)
# Added amsfonts + physics for \mathbb and \qty, etc.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}\usepackage{mathtools}\usepackage{physics}"
})


# ---------------------------- Pydantic schema -----------------------------

# --- add to the Pydantic schema ---
class EquationRendererInput(BaseModel):
    markdown_text: str = Field("", description="Markdown with LaTeX (\\[...\\] or \\(...\\)).")
    pdf_path: str = Field("", description="Optional PDF path to run Nougat if markdown_text is empty.")
    include_inline: bool = Field(default=False, description="Also extract \\(...\\).")
    out_dir: str = Field(default="data/latex_equations")
    dpi: int = Field(default=300, ge=72, le=600)
    fontsize: int = Field(default=20, ge=6, le=64)
    latex_string: Optional[str] = Field(default=None, description="Render only this single LaTeX equation.")
    max_images: int = Field(default=8, ge=1, le=50, description="Maximum number of images to return.")
    equations_top_k: int = Field(default=0, ge=0, le=50,
                                 description="If >0 and latex_string is not set, only render the top-k extracted equations.")


class LatexConsoleRendererInput(BaseModel):
    latex_string: str = Field(..., description="Single LaTeX equation to render as console-friendly Unicode text.")


# ------------------------------ Utilities --------------------------------

def enhance_unicode(expr_str: str) -> str:
    """Enhance sympy's Unicode output with better symbols"""
    replacements = {
        'alpha': 'α',
        'beta': 'β',
        'gamma': 'γ',
        'epsilon': 'ε',
        'theta': 'θ',
        'sigma': 'σ',
        'mu': 'μ',
        'pi': 'π',
        'Sum': '∑',
        'Product': '∏',
        'sqrt': '√',
        'int': '∫',
        'infinity': '∞',
        'partial': '∂',
        'nabla': '∇',
        'Element': '∈',
        'ForAll': '∀',
        'Exists': '∃',
        '\\mathbb{E}': '𝔼',
        '\\mathbb{R}': 'ℝ',
    }
    result = expr_str
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def _latex_str_hash(latex_str: str) -> str:
    return hashlib.md5(latex_str.encode("utf-8")).hexdigest()[:8]


def _sanitize_latex_for_matplotlib(eqn: str) -> str:
    """
    Sanitize/normalize Nougat-extracted LaTeX for Matplotlib's usetex/mathtools.
    - Remove \\tag{...}
    - Normalize \\coloneqq
    - Collapse stray spaces around subscripts/superscripts
    - Normalize \\mathbf{ ... } spacing
    - Replace \\qty delimiters (physics) with \\left...\\right... so it also works if physics is absent
    - Trim trailing slashes/spaces and collapse stray backslashes
    """
    if not isinstance(eqn, str):
        eqn = str(eqn or "")

    # Remove \tag{...}
    eqn = re.sub(r"\\tag\{[^}]*\}", "", eqn)

    # Normalize \coloneqq
    eqn = eqn.replace(r"\coloneqq", r":=")

    # Collapse spaces inside subscripts/superscripts: _{ x } -> _{x}
    eqn = re.sub(r"(_|\^)\s*\{\s*(.*?)\s*\}", r"\1{\2}", eqn)

    # Tighten \mathbf{ ... } etc.
    eqn = re.sub(r"\\mathbf\{\s*(.*?)\s*\}", r"\\mathbf{\1}", eqn)

    # Replace \qty(...) / \qty[...] / \qty{...} with \left..\right..
    eqn = re.sub(r"\\qty\s*\(\s*(.*?)\s*\)", r"\\left( \1 \\right)", eqn)
    eqn = re.sub(r"\\qty\s*\[\s*(.*?)\s*\]", r"\\left[ \1 \\right]", eqn)
    eqn = re.sub(r"\\qty\s*\{\s*(.*?)\s*\}", r"\\left\\{ \1 \\right\\}", eqn)

    # Remove double spaces after backslashes like "\\ " -> "\"
    eqn = re.sub(r"\\\s+", r"\\", eqn)

    # Common stray space before braces in commands: \mathbf{x }_{t} -> \mathbf{x}_{t}
    eqn = re.sub(r"(\\[A-Za-z]+)\{\s*([^}]*)\s*\}\s*(_|\^)", r"\1{\2}\3", eqn)

    # Trim trailing backslashes/spaces
    eqn = eqn.rstrip("\\").strip()

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


def _fallback_for_mathtext(eqn: str) -> str:
    """
    Prepare a LaTeX snippet for mathtext rendering (no system LaTeX).
    - Replace some macros not supported by mathtext.
    """
    s = eqn
    # Blackboard bold often limited: map to \mathrm
    s = s.replace(r"\mathbb{E}", r"\mathrm{E}")
    s = s.replace(r"\mathbb{R}", r"\mathrm{R}")
    # Remove color/phantom if present
    s = re.sub(r"\\color\{[^}]*\}", "", s)
    s = re.sub(r"\\phantom\{[^}]*\}", "", s)
    # \text{...} is limited in mathtext; approximate with \mathrm{...}
    s = re.sub(r"\\text\{([^}]*)\}", r"\\mathrm{\1}", s)
    return s


def _render_single_equation(eqn: str, out_dir: str, dpi: int, fontsize: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sanitized = _sanitize_latex_for_matplotlib(eqn)
    latex_eq = r"$" + sanitized + r"$"

    # First try with full LaTeX (usetex=True)
    try:
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, latex_eq, fontsize=fontsize, ha="center", va="center")
        filename = os.path.join(out_dir, f"equation_{_latex_str_hash(eqn)}.png")
        plt.savefig(filename, bbox_inches="tight", dpi=dpi, transparent=True)
        plt.close(fig)
        normalized_fn = os.path.normpath(filename)
        logger.info(f"[equation_renderer] Saved: {normalized_fn}")
        return normalized_fn
    except Exception as e:
        logger.exception(f"[equation_renderer] LaTeX render failed (usetex=True). Trying mathtext fallback. Eq: {sanitized} | Error: {e}")

    # Fallback: try to render with mathtext (usetex=False)
    fallback = None
    try:
        fallback = _fallback_for_mathtext(sanitized)
        mt_eq = r"$" + fallback + r"$"
        with plt.rc_context({"text.usetex": False}):
            fig2, ax2 = plt.subplots(figsize=(14, 3))
            ax2.axis("off")
            ax2.text(0.5, 0.5, mt_eq, fontsize=fontsize, ha="center", va="center")
            filename = os.path.join(out_dir, f"equation_{_latex_str_hash(eqn)}.png")
            plt.savefig(filename, bbox_inches="tight", dpi=dpi, transparent=True)
            plt.close(fig2)
            normalized_fn = os.path.normpath(filename)
            logger.info(f"[equation_renderer] Saved (mathtext fallback): {normalized_fn}")
            return normalized_fn
    except Exception as e2:
        logger.exception(f"[equation_renderer] Mathtext fallback failed. Eq: {fallback} | Error: {e2}")

    # If both fail, re-raise a clean error for the caller to record in 'failed'
    raise RuntimeError("Equation render failed for snippet: " + eqn)


# ------------------------------ Public API --------------------------------

def parse_latex_equation(latex_str: str) -> str:
    """Parse LaTeX string and return enhanced Unicode output"""
    try:
        expr = parse_latex(latex_str)
        unicode_eq = pretty(expr, use_unicode=True)
        return enhance_unicode(str(unicode_eq))
    except Exception as e:
        return f"Parsing error: {str(e)}"



def render_equation_images(markdown_text: str, out_dir: str = "data/latex_equations",
                           include_inline: bool = False, dpi: int = 300, fontsize: int = 20,
                           max_images: int = 8) -> Dict[str, Any]:
    equations = _extract_latex_equations(markdown_text or "", include_inline=include_inline)
    images, failed = [], []
    for eq in equations:
        if len(images) >= max_images:   # 👈 cap
            break
        try:
            images.append(_render_single_equation(eq, out_dir=out_dir, dpi=dpi, fontsize=fontsize))
        except Exception as e:
            logger.exception(f"[equation_renderer] Failed to render: {eq} | {e}")
            failed.append(eq)
    return {"count": len(images), "images": images, "failed": failed}


def render_equation(md_input: EquationRendererInput) -> Dict[str, Any]:
    # Single-equation fast path
    if md_input.latex_string:
        single_md = f"\\[{_sanitize_latex_for_matplotlib(md_input.latex_string)}\\]"
        return render_equation_images(
            markdown_text=single_md,
            out_dir=md_input.out_dir,
            include_inline=False,
            dpi=md_input.dpi,
            fontsize=md_input.fontsize,
            max_images=1,  # force single
        )

    md = md_input.markdown_text
    if not md and md_input.pdf_path:
        try:
            md = call_nougat_api(md_input.pdf_path)
        except Exception as e:
            logger.exception(f"[equation_renderer] Nougat call failed for {md_input.pdf_path}: {e}")
            return {"count": 0, "images": [], "failed": [], "error": f"Nougat failed: {e}"}

    if not md:
        return {"count": 0, "images": [], "failed": [], "error": "No markdown_text provided and no pdf_path to derive it."}

    # If we have full markdown and want only top-k, trim before rendering
    if md and not md_input.latex_string and getattr(md_input, "equations_top_k", 0) > 0:
        eqs_all = _extract_latex_equations(md, include_inline=md_input.include_inline)

        # Simple importance heuristic: losses/objectives first, then longer eqs
        def _score(s: str) -> int:
            s2 = s.lower()
            return (
                    5 * any(k in s2 for k in ["\\mathcal{l}", "loss", "objective", "\\mathrm{kl}", "elbo"])
                    + 3 * any(k in s2 for k in ["\\sum", "\\prod", "\\int", "\\log", "\\mathbb{e}"])
                    + min(len(s), 120) // 20
            )

        topk = sorted(eqs_all, key=_score, reverse=True)[: md_input.equations_top_k]
        # Rebuild minimal markdown containing only the chosen equations
        md = "".join([f"\\[{_sanitize_latex_for_matplotlib(e)}\\]\n" for e in topk])

    return render_equation_images(
        markdown_text=md,
        out_dir=md_input.out_dir,
        include_inline=md_input.include_inline,
        dpi=md_input.dpi,
        fontsize=md_input.fontsize,
        max_images=md_input.max_images,  # respect cap
    )

