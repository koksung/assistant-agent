from app.tools.local.pdf_extractor import extract_pdf_text

def check_pdf_extraction():
    path = "../data/ddpm.pdf"
    sections = extract_pdf_text(path)
    assert len(sections) > 0, "No text extracted — is the PDF scanned or empty?"

    for name, content in sections.items():
        print(f"\n--- {name.upper()} ---\n")
        # print(content)


if __name__ == "__main__":
    check_pdf_extraction()
