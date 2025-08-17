from app.tools.local.pdf_extractor import extract_pdf_text
from app.tools.remote.docling_pdf_extractor.tool import call_docling_pdf_extractor_remote


def check_local_pdf_extraction(pdf_path):
    sections = extract_pdf_text(pdf_path)
    assert len(sections) > 0, "No text extracted — is the PDF scanned or empty?"

    for name, content in sections.items():
        print(f"\n--- {name.upper()} ---\n")
        # print(content)


def check_mcp_pdf_extraction(pdf_path):
    output_dict = call_docling_pdf_extractor_remote(pdf_path)
    output = "\n\n".join(
                f"# {section_name.replace('_', ' ').title()}\n{section_text.strip()}"
                for section_name, section_text in output_dict.items()
            )
    print(output)


if __name__ == "__main__":
    check_local = False
    file_path = "C:/Users/Zeus/PycharmProjects/assistant-agent/data/ddpm-short.pdf"
    if check_local:
        check_local_pdf_extraction(file_path)
    else:
        check_mcp_pdf_extraction(file_path)
