from app.tools.local.pdf_extractor import extract_pdf_text

def test_pdf_extraction():
    path = "../data/ddpm.pdf"
    text = extract_pdf_text(path)

    print("Extracted text:")
    print("-" * 50)
    print(text[:1000])  # Print the first 1000 characters
    print("-" * 50)

    assert len(text) > 0, "No text extracted — is the PDF scanned or empty?"


if __name__ == "__main__":
    test_pdf_extraction()
