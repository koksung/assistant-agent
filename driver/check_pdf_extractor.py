from app.tools.local.pdf_extractor import extract_pdf_text
from app.tools.remote.docling_pdf_extractor.tool import call_docling_pdf_extractor_remote
from app.tools.remote.nougat_pdf_extractor import nougat_pdf_extraction


def check_local_pdf_extraction(pdf_path):
    sections = extract_pdf_text(pdf_path)
    assert len(sections) > 0, "No text extracted — is the PDF scanned or empty?"

    for name, content in sections.items():
        print(f"\n--- {name.upper()} ---\n")
        # print(content)


def check_remote_docling_pdf_extraction(pdf_path):
    output_dict = call_docling_pdf_extractor_remote(pdf_path)
    output = "\n\n".join(
                f"# {section_name.replace('_', ' ').title()}\n{section_text.strip()}"
                for section_name, section_text in output_dict.items()
            )
    print(output)


def check_remote_nougat_pdf_extraction(pdf_path):
    output_dict = nougat_pdf_extraction(pdf_path)
    output = "\n\n".join(
                f"# {section_name.replace('_', ' ').title()}\n{section_text.strip()}"
                for section_name, section_text in output_dict.items()
            )
    print(output)


if __name__ == "__main__":
    check_flag = "3"
    file_path = "C:/Users/Zeus/PycharmProjects/assistant-agent/data/ddpm-short.pdf"
    match check_flag:
        case "1":
            check_local_pdf_extraction(file_path)
        case "2":
            check_remote_docling_pdf_extraction(file_path)
        case "3":
            check_remote_nougat_pdf_extraction(file_path)
