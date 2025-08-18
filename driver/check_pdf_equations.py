import json

from app.tools.local.equation_renderer import render_equation, EquationRendererInput


def check_extract_equations_from_pdf(pdf_path, include_inline=False):
    payload = EquationRendererInput(
        pdf_path=pdf_path,  # if you already have markdown_text, pass it instead
        markdown_text="",  # leave empty to auto-call Nougat
        include_inline=include_inline,
        out_dir="../data/latex_equations",
        dpi=300,
        fontsize=20,
    )
    result = render_equation(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    file_path = "C:/Users/Zeus/PycharmProjects/assistant-agent/data/ddpm-short.pdf"
    check_extract_equations_from_pdf(file_path)
