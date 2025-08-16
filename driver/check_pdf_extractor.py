from app.tools.local.pdf_extractor import extract_pdf_text
from app.tools.remote.docling_pdf_extractor import DoclingExtractorTool


def check_local_pdf_extraction(pdf_path):
    sections = extract_pdf_text(pdf_path)
    assert len(sections) > 0, "No text extracted — is the PDF scanned or empty?"

    for name, content in sections.items():
        print(f"\n--- {name.upper()} ---\n")
        # print(content)


def check_mcp_pdf_extraction(pdf_path):
    output = DoclingExtractorTool.run(pdf_path)
    print(output[:1000])


if __name__ == "__main__":
    check_local = False
    file_path = "../data/ddpm.pdf"
    if check_local:
        check_local_pdf_extraction(file_path)
    else:
        check_mcp_pdf_extraction(file_path)
